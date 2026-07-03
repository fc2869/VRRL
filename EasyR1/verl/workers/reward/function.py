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

import importlib.util
import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        ...


class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        multi_turn_mask = data.batch["multi_turn_mask"]

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "ground_truth": data.non_tensor_batch["answer" if "answer" in data.non_tensor_batch else "ground_truth"][i],
            }

            # Add additional training data fields if available
            if "table_id" in data.non_tensor_batch:
                reward_input["table_id"] = data.non_tensor_batch["table_id"][i]

            if "question_id" in data.non_tensor_batch:
                reward_input["question_id"] = data.non_tensor_batch["question_id"][i]

            if "related_data" in data.non_tensor_batch:
                reward_input["related_data"] = data.non_tensor_batch["related_data"][i]

            if "meta_data_path" in data.non_tensor_batch:
                reward_input["meta_data_path"] = data.non_tensor_batch["meta_data_path"][i]

            if "crop_paths_data" in data.non_tensor_batch:
                reward_input["crop_paths_data"] = data.non_tensor_batch["crop_paths_data"][i]

            score = self.reward_fn(reward_input)

            # Find the last assistant token position for reward assignment
            assistant_token_positions = torch.where(multi_turn_mask[i] == 1)[0]
            if len(assistant_token_positions) > 0:
                # Assign reward to the last assistant token
                last_assistant_pos = assistant_token_positions[-1].item()
                reward_tensor[i, last_assistant_pos] = score["overall"]
            else:
                # Fallback: assign to the last response token if no assistant tokens found
                reward_tensor[i, cur_response_length - 1] = score["overall"]
            
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)

        multi_turn_mask = data.batch["multi_turn_mask"]

        for i in range(len(data)):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            reward_input = {
                "response": response_str,
                "response_length": cur_response_length,
                "response_token_ids": valid_response_ids.tolist(),
                "ground_truth": data.non_tensor_batch["answer" if "answer" in data.non_tensor_batch else "ground_truth"][i],
            }

            # Add full conversation sequence if available
            if "raw_prompt_ids" in data.non_tensor_batch:
                try:
                    raw_ids = data.non_tensor_batch["raw_prompt_ids"][i]
                    reward_input["sequence"] = self.tokenizer.decode(raw_ids, skip_special_tokens=False)
                except Exception:
                    pass

            # Add additional training data fields if available
            if "table_id" in data.non_tensor_batch:
                reward_input["table_id"] = data.non_tensor_batch["table_id"][i]
            
            if "question_id" in data.non_tensor_batch:
                reward_input["question_id"] = data.non_tensor_batch["question_id"][i]
            
            if "related_data" in data.non_tensor_batch:
                reward_input["related_data"] = data.non_tensor_batch["related_data"][i]
            
            if "meta_data_path" in data.non_tensor_batch:
                reward_input["meta_data_path"] = data.non_tensor_batch["meta_data_path"][i]
            
            if "task" in data.non_tensor_batch:
                reward_input["task"] = data.non_tensor_batch["task"][i]
            
            # Start turn for random restart training (used for reflection reward calculation)
            if "start_turn" in data.non_tensor_batch:
                reward_input["start_turn"] = int(data.non_tensor_batch["start_turn"][i])

            # Per-sample prefix-buffer metadata for the optional decision-bonus
            # path in frozenlake_score (prefix_buffer_decision_bonus kwarg).
            # All three fields populated by MultiTurnRolloutFrozenLake on pb
            # rollouts; absent on rs / normal rollouts (decision bonus is a no-op).
            if "is_prefix_buffer_rollout" in data.non_tensor_batch:
                try:
                    reward_input["is_prefix_buffer_rollout"] = bool(
                        data.non_tensor_batch["is_prefix_buffer_rollout"][i])
                except Exception:
                    pass
            if "prefix_buffer_type" in data.non_tensor_batch:
                pbt = data.non_tensor_batch["prefix_buffer_type"][i]
                if pbt is not None:
                    reward_input["prefix_buffer_type"] = str(pbt)
            if "prefix_buffer_num_turns" in data.non_tensor_batch:
                pbn = data.non_tensor_batch["prefix_buffer_num_turns"][i]
                if pbn is not None:
                    try:
                        reward_input["prefix_buffer_num_turns"] = int(pbn)
                    except Exception:
                        pass

            # if "crop_paths_data" in data.non_tensor_batch:
            #     reward_input["crop_paths_data"] = data.non_tensor_batch["crop_paths_data"][i]
            # Image size for Qwen3 [0-1000] coordinate normalization
            if "image_sizes" in data.non_tensor_batch:
                try:
                    sz = data.non_tensor_batch["image_sizes"][i]
                    reward_input["image_size"] = (int(sz[0]), int(sz[1]))
                except Exception:
                    pass
            
            if "crop_paths_data" in data.non_tensor_batch:
                reward_input["crop_paths_data"] = data.non_tensor_batch["crop_paths_data"][i]

            reward_inputs.append(reward_input)

        # Propagate reflection flags for logging (reflection vs normal) if available
        reward_kwargs = {}
        # Forward global_step so reward functions can apply step-dependent schedules
        # (e.g. lambda_deg warm-up, reflect_weight ramps). Source: data.meta_info,
        # populated by ray_trainer.py before each compute_reward call.
        try:
            reward_kwargs["global_step"] = int(data.meta_info.get("global_step", 0))
        except Exception:
            reward_kwargs["global_step"] = 0
        if "is_random_init_rollout" in data.non_tensor_batch:
            try:
                flags = data.non_tensor_batch["is_random_init_rollout"]
                if isinstance(flags, np.ndarray):
                    flags = flags.tolist()
                reward_kwargs["is_random_init_rollout"] = flags
                # also store per-input for completeness
                for _ri, _flag in zip(reward_inputs, flags):
                    _ri["is_random_init_rollout"] = bool(_flag)
            except Exception:
                pass

        # Propagate prefix buffer flags for logging prefix buffer traces
        if "is_prefix_buffer_rollout" in data.non_tensor_batch:
            try:
                pb_flags = data.non_tensor_batch["is_prefix_buffer_rollout"]
                if isinstance(pb_flags, np.ndarray):
                    pb_flags = pb_flags.tolist()
                for _ri, _flag in zip(reward_inputs, pb_flags):
                    _ri["is_prefix_buffer_rollout"] = bool(_flag)
            except Exception:
                pass
        if "prefix_buffer_type" in data.non_tensor_batch:
            try:
                pb_types = data.non_tensor_batch["prefix_buffer_type"]
                if isinstance(pb_types, np.ndarray):
                    pb_types = pb_types.tolist()
                for _ri, _pt in zip(reward_inputs, pb_types):
                    _ri["prefix_buffer_type"] = _pt if _pt is not None else ""
            except Exception:
                pass
        if "prefix_buffer_num_turns" in data.non_tensor_batch:
            try:
                pb_turns = data.non_tensor_batch["prefix_buffer_num_turns"]
                if isinstance(pb_turns, np.ndarray):
                    pb_turns = pb_turns.tolist()
                for _ri, _pt in zip(reward_inputs, pb_turns):
                    _ri["prefix_buffer_num_turns"] = int(_pt) if _pt is not None else 0
            except Exception:
                pass

        scores = self.reward_fn(reward_inputs, **reward_kwargs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            # Find the last assistant token position for reward assignment
            assistant_token_positions = torch.where(multi_turn_mask[i] == 1)[0]
            if len(assistant_token_positions) > 0:
                # Assign reward to the last assistant token
                last_assistant_pos = assistant_token_positions[-1].item()
                reward_tensor[i, last_assistant_pos] = score["overall"]
            else:
                # Fallback: assign to the last response token if no assistant tokens found
                response_length = torch.sum(data.batch["response_mask"][i], dim=-1)
                cur_response_length = int(response_length.item())
                reward_tensor[i, cur_response_length - 1] = score["overall"]
            
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
