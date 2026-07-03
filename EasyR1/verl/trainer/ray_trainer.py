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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_random_restart_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                # `answer` is what MultiTurnRLHFDataset emits, `ground_truth` is
                # what RLHFDataset emits (it renames answer -> ground_truth at
                # __getitem__). Pop both so the rollout sees the map spec no
                # matter which dataset class the val_dataloader is using.
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "answer", "ground_truth"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        # Slice indicators accumulated alongside `all_metrics` across OLF
        # retries — needed so the post-loop slice computation can attribute
        # each entry in `all_metrics` to {prefix_buffer, non_prefix_buffer}.
        # Tracks the FULL pre-OLF superset (every rollout we generated,
        # including those filtered out), which is the truth about policy
        # behavior at sampling time. The post-OLF (kept-rollouts-only) view
        # is still emitted separately in fit() to show what gradient was
        # actually applied.
        all_is_pb: list = []
        all_pb_types: list = []
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "video_fps": self.config.data.video_fps,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "answer"],
                meta_info_keys=["min_pixels", "max_pixels", "video_fps"],
            )

            # generate a batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            # Remove placeholder responses and response_mask from new_batch before unioning with generated responses
            if "responses" in new_batch.batch:
                new_batch.batch.pop("responses")
            if "response_mask" in new_batch.batch:
                new_batch.batch.pop("response_mask")
            new_batch = new_batch.union(gen_batch_output)

            # filter group
            if self.config.algorithm.online_filtering:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                new_batch.batch["token_level_scores"] = reward_tensor
                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)
                # Accumulate slice indicators in lockstep with `all_metrics`.
                # Each new_batch may have a different is_prefix_buffer_rollout
                # composition (e.g. when buffer min_size not yet reached, all
                # rollouts are non-PB). Default to all-False when absent so
                # the accumulator stays length-aligned with `all_metrics`.
                nb_len = len(new_batch)
                nb_is_pb = new_batch.non_tensor_batch.get(
                    "is_prefix_buffer_rollout", None)
                nb_pb_types = new_batch.non_tensor_batch.get(
                    "prefix_buffer_type", None)
                if nb_is_pb is not None and len(nb_is_pb) == nb_len:
                    all_is_pb.extend([bool(x) for x in nb_is_pb])
                else:
                    all_is_pb.extend([False] * nb_len)
                if nb_pb_types is not None and len(nb_pb_types) == nb_len:
                    all_pb_types.extend([x for x in nb_pb_types])
                else:
                    all_pb_types.extend([None] * nb_len)

                # Compute void traces (format reward == 0) for this batch
                try:
                    fmt_scores = reward_metrics.get("format", [])
                    if len(fmt_scores) == len(new_batch):
                        void_traces = np.asarray([(
                            (s is None) or (float(s) == 0.0)
                        ) for s in fmt_scores], dtype=object)
                        new_batch.non_tensor_batch["void_traces"] = void_traces
                    else:
                        # Length mismatch; skip attaching void_traces for safety
                        pass
                except Exception:
                    # Be robust; do not crash training due to metrics parsing
                    pass

                filter_scores = reward_metrics[self.config.algorithm.filter_key]
                uids = new_batch.non_tensor_batch["uid"]
                uid2scores = defaultdict(list)
                for uid, score in zip(uids, filter_scores):
                    uid2scores[uid].append(score)

                uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                kept_uids = [
                    uid
                    for uid, avg_score in uid2mean.items()
                    if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
                ]
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    print(
                        f"Warning: all {len(uid2mean)} uids filtered out in this sub-batch "
                        f"(scores: {list(uid2mean.values())}). Skipping and retrying..."
                    )
                    # Don't update batch; fall through to the retry logic below
                else:
                    new_batch = new_batch[kept_sample_idxs]
                    batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            else:
                batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n if batch is not None else 0
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                if self.config.algorithm.online_filtering:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
                    # Pre-OLF slice metrics: ground truth across every rollout
                    # we generated this step (including OLF-rejected groups).
                    # Same key conventions as the post-OLF slicing below, just
                    # with the `_unfiltered` suffix on the prefix so the two
                    # views can be compared side-by-side in wandb.
                    try:
                        from .prefix_buffer_metrics import (
                            compute_prefix_buffer_quality_metrics,
                            compute_non_pb_quality_metrics,
                            compute_all_quality_metrics)
                        metrics.update(compute_prefix_buffer_quality_metrics(
                            all_is_pb, all_pb_types, all_metrics,
                            prefix="prefix_buffer_unfiltered"))
                        metrics.update(compute_non_pb_quality_metrics(
                            all_is_pb, all_metrics,
                            prefix="non_prefix_buffer_unfiltered"))
                        metrics.update(compute_all_quality_metrics(
                            all_metrics, prefix="all_unfiltered"))
                    except Exception:
                        # Don't fail training if slice metrics blow up
                        pass

                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor

                        # Attach void traces (format reward == 0) per-sample to non-tensor batch
                        try:
                            fmt_scores = reward_metrics.get("format", [])
                            if len(fmt_scores) == len(batch):
                                void_traces = np.asarray([(
                                    (s is None) or (float(s) == 0.0)
                                ) for s in fmt_scores], dtype=object)
                                batch.non_tensor_batch["void_traces"] = void_traces
                            else:
                                # Length mismatch; skip attaching void_traces for safety
                                pass
                        except Exception:
                            pass
                            
                        # Save per-sample EM rewards BEFORE reduce_metrics aggregates them to scalars
                        # (FrozenLake's primary correctness signal; pointing-task used "coordinate" here)
                        per_sample_em_rewards = reward_metrics.get("em", None)
                        
                        # Filter out prefix buffer samples from reward metrics so that "reward/" 
                        # section only shows random_start trace performance (prefix buffer samples
                        # are already tracked separately in the "prefix_buffer/" section)
                        is_pb_flags = batch.non_tensor_batch.get("is_prefix_buffer_rollout", None)
                        if is_pb_flags is not None and len(is_pb_flags) > 0:
                            # Filter reward_metrics to only include non-prefix-buffer samples
                            non_pb_indices = [i for i, is_pb in enumerate(is_pb_flags) if not is_pb]
                            if len(non_pb_indices) > 0:
                                filtered_reward_metrics = {}
                                for k, v in reward_metrics.items():
                                    if isinstance(v, list) and len(v) == len(is_pb_flags):
                                        filtered_reward_metrics[k] = [v[i] for i in non_pb_indices]
                                    else:
                                        # Keep as-is if not a per-sample list
                                        filtered_reward_metrics[k] = v
                                reward_metrics_for_logging = filtered_reward_metrics
                            else:
                                # All samples are prefix buffer, skip logging reward metrics
                                reward_metrics_for_logging = {}
                        else:
                            reward_metrics_for_logging = reward_metrics

                        # Post-OLF per-slice quality metrics (truth on the
                        # rollouts the gradient actually saw). Pre-OLF
                        # `*_unfiltered/` variants are emitted in
                        # _make_batch_data; this block emits the post-OLF
                        # `prefix_buffer/{key}_{wrong,right,all}_mean`,
                        # `non_prefix_buffer/{key}_mean`, and `all/{key}_mean`
                        # keys with the full DEFAULT_KEYS set (em_turn1,
                        # em_final, pr_turn1, pr_final, reflection, etc.). Fires
                        # even when buffer mode is disabled (is_pb may be None
                        # or all-False).
                        try:
                            from .prefix_buffer_metrics import (
                                compute_prefix_buffer_quality_metrics,
                                compute_non_pb_quality_metrics,
                                compute_all_quality_metrics)
                            _is_pb_postolf = batch.non_tensor_batch.get("is_prefix_buffer_rollout", None)
                            _pb_types_postolf = batch.non_tensor_batch.get("prefix_buffer_type", None)
                            metrics.update(compute_prefix_buffer_quality_metrics(
                                _is_pb_postolf, _pb_types_postolf, reward_metrics))
                            metrics.update(compute_non_pb_quality_metrics(
                                _is_pb_postolf, reward_metrics))
                            metrics.update(compute_all_quality_metrics(
                                reward_metrics))
                        except Exception:
                            # Slice metrics are advisory; don't crash training.
                            pass
                        
                        if reward_metrics_for_logging:
                            reduced_reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics_for_logging).items()}
                            metrics.update(reduced_reward_metrics)
                        
                        # Compute prefix buffer accuracy by prefix turn count
                        try:
                            is_pb = batch.non_tensor_batch.get("is_prefix_buffer_rollout", None)
                            pb_turns = batch.non_tensor_batch.get("prefix_buffer_num_turns", None)
                            pb_types = batch.non_tensor_batch.get("prefix_buffer_type", None)
                            pb_imm_term = batch.non_tensor_batch.get("prefix_buffer_immediate_terminate", None)
                            em_rewards = per_sample_em_rewards

                            if is_pb is not None and pb_turns is not None and em_rewards is not None:
                                from collections import defaultdict
                                turn_to_total = defaultdict(int)
                                turn_to_correct = defaultdict(int)
                                type_to_total = defaultdict(int)
                                type_to_correct = defaultdict(int)
                                total_pb = 0
                                total_correct = 0
                                
                                for i in range(len(is_pb)):
                                    if not is_pb[i]:
                                        continue
                                    turn = pb_turns[i]
                                    if turn is None or (isinstance(turn, (int, float)) and int(turn) <= 0):
                                        continue
                                    
                                    pb_type = pb_types[i] if pb_types is not None and i < len(pb_types) else None
                                    em_reward = em_rewards[i] if i < len(em_rewards) else 0.0
                                    is_correct = float(em_reward) >= 0.99  # em == 1.0 means strict-EM correct
                                    
                                    total_pb += 1
                                    turn_to_total[int(turn)] += 1
                                    if is_correct:
                                        total_correct += 1
                                        turn_to_correct[int(turn)] += 1
                                    
                                    if pb_type:
                                        type_to_total[pb_type] += 1
                                        if is_correct:
                                            type_to_correct[pb_type] += 1
                                
                                # Log overall accuracy
                                if total_pb > 0:
                                    metrics["prefix_buffer/acc_overall"] = total_correct / total_pb
                                    metrics["prefix_buffer/acc_count"] = total_pb
                                
                                # Log accuracy by prefix turn count
                                for turn in sorted(turn_to_total.keys()):
                                    total = turn_to_total[turn]
                                    correct = turn_to_correct.get(turn, 0)
                                    metrics[f"prefix_buffer/acc_turns_{turn}"] = correct / total if total > 0 else 0.0
                                    metrics[f"prefix_buffer/acc_turns_{turn}_count"] = total
                                
                                # Log accuracy by prefix type (wrong vs right)
                                for ptype in type_to_total.keys():
                                    total = type_to_total[ptype]
                                    correct = type_to_correct.get(ptype, 0)
                                    metrics[f"prefix_buffer/acc_type_{ptype}"] = correct / total if total > 0 else 0.0
                                    metrics[f"prefix_buffer/acc_type_{ptype}_count"] = total

                                # Log immediate-terminate ratio for right-type prefix buffer samples
                                if pb_imm_term is not None:
                                    right_imm_total = 0
                                    right_imm_count = 0
                                    for i in range(len(is_pb)):
                                        if not is_pb[i]:
                                            continue
                                        pb_type = pb_types[i] if pb_types is not None and i < len(pb_types) else None
                                        if pb_type != 'right':
                                            continue
                                        flag = pb_imm_term[i] if i < len(pb_imm_term) else None
                                        if flag is None:
                                            continue
                                        right_imm_total += 1
                                        if flag:
                                            right_imm_count += 1
                                    if right_imm_total > 0:
                                        metrics["prefix_buffer/right_prefix_immediate_terminate_ratio"] = right_imm_count / right_imm_total
                                        metrics["prefix_buffer/right_prefix_immediate_terminate_count"] = right_imm_count

                            # Log reflection score stats for prefix buffer samples
                            pb_reflection_rewards = reward_metrics.get("reflection", None)
                            if is_pb is not None and pb_reflection_rewards is not None:
                                turn_to_reflection = defaultdict(list)
                                type_to_reflection = defaultdict(list)
                                all_pb_reflections = []

                                for i in range(len(is_pb)):
                                    if not is_pb[i]:
                                        continue
                                    refl = pb_reflection_rewards[i] if i < len(pb_reflection_rewards) else 0.0
                                    all_pb_reflections.append(float(refl))

                                    turn = pb_turns[i] if pb_turns is not None and i < len(pb_turns) else None
                                    if turn is not None and isinstance(turn, (int, float)) and int(turn) > 0:
                                        turn_to_reflection[int(turn)].append(float(refl))

                                    pb_type = pb_types[i] if pb_types is not None and i < len(pb_types) else None
                                    if pb_type:
                                        type_to_reflection[pb_type].append(float(refl))

                                if all_pb_reflections:
                                    metrics["prefix_buffer/reflection_mean"] = float(np.mean(all_pb_reflections))

                                for turn in sorted(turn_to_reflection.keys()):
                                    vals = turn_to_reflection[turn]
                                    if vals:
                                        metrics[f"prefix_buffer/reflection_turns_{turn}_mean"] = float(np.mean(vals))

                                for ptype, vals in type_to_reflection.items():
                                    if vals:
                                        metrics[f"prefix_buffer/reflection_type_{ptype}_mean"] = float(np.mean(vals))

                            # Per-slice quality metrics are emitted
                            # unconditionally just above. The
                            # `prefix_buffer/reflection_mean` aggregation above
                            # this point stays inside the gate because it
                            # depends on pb_turns / pb_types being non-None.

                        except Exception as e:
                            # Don't fail training if accuracy logging fails
                            pass

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))
            metrics.update(compute_random_restart_metrics(batch=batch))
            
            # Update prefix buffer step and collect buffer stats
            try:
                self.actor_rollout_ref_wg.update_prefix_buffer_step(self.global_step)
                buffer_stats = self.actor_rollout_ref_wg.get_prefix_buffer_stats()
                if isinstance(buffer_stats, (list, tuple)):
                    # ONE_TO_ALL dispatch may return per-rank stats; use the first non-empty dict
                    buffer_stats = next((s for s in buffer_stats if s), {})
                if isinstance(buffer_stats, dict) and buffer_stats:
                    # Prefix buffer stats with "prefix_buffer/" prefix
                    for k, v in buffer_stats.items():
                        metrics[f"prefix_buffer/{k}"] = v
            except Exception:
                # Silently fail if prefix buffer methods are not available
                pass

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
