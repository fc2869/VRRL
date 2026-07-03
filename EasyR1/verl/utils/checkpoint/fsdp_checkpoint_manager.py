# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

from .checkpoint_manager import BaseCheckpointManager

try:
    from peft import PeftModel
except Exception:  # pragma: no cover - PEFT is optional
    PeftModel = None  # type: ignore


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    A checkpoint manager that saves and loads
    - model
    - optimizer
    - lr_scheduler
    - extra_states
    in a SPMD way.

    We save
    - sharded model states and optimizer states
    - full lr_scheduler states
    - huggingface tokenizer and config for ckpt merge
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin],
    ):
        super().__init__(model, optimizer, lr_scheduler, processing_class)

    def load_checkpoint(self, path: Optional[str] = None):
        if path is None:
            return

        # every rank download its own checkpoint
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[rank-{self.rank}]: model checkpoint not found: {model_path}")

        has_optim = os.path.exists(optim_path)
        has_extra = os.path.exists(extra_path)

        print(f"[rank-{self.rank}]: Loading model from {os.path.abspath(model_path)}.")
        model_state_dict = torch.load(model_path, weights_only=False)

        state_dict_options = StateDictOptions(cpu_offload=True)

        if has_optim:
            print(f"[rank-{self.rank}]: Loading optimizer from {os.path.abspath(optim_path)}.")
            optim_state_dict = torch.load(optim_path, weights_only=False)
            set_state_dict(
                model=self.model,
                optimizers=self.optimizer,
                model_state_dict=model_state_dict,
                optim_state_dict=optim_state_dict,
                options=state_dict_options,
            )
        else:
            # Checkpoint was saved with save_model_only=True. Load model
            # weights only; optimizer state remains its fresh-init value.
            # The student/resumed run continues from the saved weights but
            # with a clean Adam moments tensor — this is a degraded form
            # of true resume but is necessary when optim wasn't saved.
            print(
                f"[rank-{self.rank}]: optimizer checkpoint not found at "
                f"{optim_path}; loading model weights only and starting "
                f"optimizer state fresh."
            )
            set_model_state_dict(
                model=self.model,
                model_state_dict=model_state_dict,
                options=state_dict_options,
            )

        if has_extra:
            print(f"[rank-{self.rank}]: Loading extra_state from {os.path.abspath(extra_path)}.")
            extra_state_dict = torch.load(extra_path, weights_only=False)
            self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])
            # recover random state
            if "rng" in extra_state_dict:
                self.load_rng_state(extra_state_dict["rng"])
        else:
            print(
                f"[rank-{self.rank}]: extra_state checkpoint not found at "
                f"{extra_path}; lr_scheduler and rng state will start fresh."
            )

    def save_checkpoint(self, path: str, save_model_only: bool = False):
        path = self.local_mkdir(path)
        dist.barrier()

        # every rank will save its own model and optim shard
        model_path = os.path.join(path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        optim_path = os.path.join(path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        extra_path = os.path.join(path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

        state_dict_options = StateDictOptions(cpu_offload=True)
        if save_model_only:
            model_state_dict = get_model_state_dict(self.model, options=state_dict_options)
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            torch.save(model_state_dict, model_path)
        else:
            model_state_dict, optim_state_dict = get_state_dict(self.model, self.optimizer, options=state_dict_options)
            extra_state_dict = {
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "rng": self.get_rng_state(),
            }
            print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}.")
            print(f"[rank-{self.rank}]: Saving optimizer to {os.path.abspath(optim_path)}.")
            print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}.")
            torch.save(model_state_dict, model_path)
            torch.save(optim_state_dict, optim_path)
            torch.save(extra_state_dict, extra_path)

        # wait for everyone to dump to local
        dist.barrier()

        if self.rank == 0:
            hf_path = os.path.join(path, "huggingface")
            os.makedirs(hf_path, exist_ok=True)

            wrapped_module = self.model._fsdp_wrapped_module
            base_model: Optional[PreTrainedModel] = None

            if isinstance(wrapped_module, PreTrainedModel):
                base_model = wrapped_module
            elif PeftModel is not None and isinstance(wrapped_module, PeftModel):
                # LoRA / PEFT wrappers expose the underlying HF model via get_base_model()
                candidate = wrapped_module.get_base_model()
                if isinstance(candidate, PreTrainedModel):
                    base_model = candidate
                # Persist adapter weights/config so checkpoints remain reloadable
                adapter_dir = os.path.join(hf_path, "adapter")
                wrapped_module.save_pretrained(adapter_dir)

            if base_model is None:
                raise TypeError(
                    "FSDPCheckpointManager only supports saving HuggingFace PreTrainedModel or PEFT-wrapped models; "
                    f"got {type(self.model._fsdp_wrapped_module)}."
                )

            base_model.config.save_pretrained(hf_path)
            if getattr(base_model, "generation_config", None) is not None:
                base_model.generation_config.save_pretrained(hf_path)

            self.processing_class.save_pretrained(hf_path)

        dist.barrier()
