# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License").

"""Multi-turn vLLM rollout for FrozenLake GRPO training.

Drives the PG loop, vLLM batching, attention-mask construction, and DataProto
packaging for the FrozenLake route/answer protocol: stop string, finish check,
per-turn feedback-image rendering, and initial image source.
"""

import copy
import os
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed
from PIL import Image
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig
from .frozenlake_helpers import (
    FEEDBACK_USER_PROMPT_REFLECTION,
    is_finished,
    parse_route_terminate,
    render_feedback_pil,
    save_batch_logs,
)


# Module-level base-map image cache.

_BASE_MAP_CACHE: dict[str, "Image.Image"] = {}


def _load_base_image_cached(origin_image):
    """Return a PIL.Image RGB copy for `origin_image`.

    If it's already a PIL.Image (the eval path passes Images directly), return
    as-is. If it's a string path, look up in `_BASE_MAP_CACHE` and load+decode
    only on miss. Cache keys are str paths only — PIL.Image inputs are not
    cached because they're already in memory.
    """
    if isinstance(origin_image, str):
        cached = _BASE_MAP_CACHE.get(origin_image)
        if cached is None:
            cached = Image.open(origin_image).convert("RGB")
            _BASE_MAP_CACHE[origin_image] = cached
        return cached
    return origin_image


def _repeat_interleave(value, repeats):
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    return np.repeat(value, repeats, axis=0)


class MultiTurnRolloutFrozenLake(BaseRollout):
    def __init__(self, model_path, config, tokenizer, processor):
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        self.tokenizer = tokenizer
        self.processor = processor

        self.max_turns = getattr(config, "max_turns", 8)
        self.num_llm_calls_available = getattr(config, "num_llm_calls_available", 8)
        self.single_turn_response_length = getattr(
            config, "single_turn_response_length", 500
        )
        # Optional rollout logging: when temp_dir is set, each generate_sequences call
        # writes a per-batch subdir containing feedback PNGs + transcripts.jsonl.
        self.temp_dir = getattr(config, "temp_dir", None)

        # Output format expected from the rolled-out policy: "tag" (default,
        # <route>/<answer>) or "json" (function_call schema). Threaded through
        # every parse_route_terminate / is_finished call so the rollout-time
        # parser matches the model.
        self.fmt = getattr(config, "fmt", "tag")

        # --- Mode selection + prefix-buffer + random-start configuration ---
        self.per_question_mode_selection = getattr(
            config, "per_question_mode_selection", False
        )
        self.normal_mode_weight = getattr(config, "normal_mode_weight", 1.0)
        self.random_start_mode_weight = getattr(config, "random_start_mode_weight", 0.0)
        self.prefix_buffer_mode_weight = getattr(
            config, "prefix_buffer_mode_weight", 0.0
        )

        self.random_start_turn_min = getattr(config, "random_start_turn_min", 1)
        self.random_start_mask_all_until_final = getattr(
            config, "random_start_mask_all_until_final", False
        )

        self.prefix_buffer_size = getattr(config, "prefix_buffer_size", 1000)
        self.prefix_buffer_wrong_ratio = getattr(
            config, "prefix_buffer_wrong_ratio", 0.8
        )
        self.prefix_buffer_max_staleness_steps = getattr(
            config, "prefix_buffer_max_staleness_steps", 100
        )
        self.prefix_buffer_min_size = getattr(config, "prefix_buffer_min_size", 100)
        self.prefix_buffer_max_per_question = getattr(
            config, "prefix_buffer_max_per_question", 3
        )
        self.prefix_buffer_force_route = getattr(
            config, "prefix_buffer_force_pointing", False
        )  # name reuse
        self.prefix_buffer_recycle = getattr(config, "prefix_buffer_recycle", False)
        self.prefix_buffer_max_route_turns = getattr(
            config, "prefix_buffer_max_pointing_turns", 6
        )  # name reuse
        self.prefix_buffer_correct_replay = getattr(
            config, "prefix_buffer_correct_replay", False
        )
        # OLF-style gate at collection: only collect buffer entries from
        # questions whose group em-mean lies in (em_lower, em_upper). Filters
        # out "hopeless" wrong rollouts (group em=0, model can't recover) AND
        # trivially-easy questions (group em=1, no useful wrong entries
        # available anyway). Defaults (0.0, 1.0) preserve prior behavior.

        self.prefix_buffer_collect_em_lower = float(
            getattr(config, "prefix_buffer_collect_em_lower", 0.0)
        )
        self.prefix_buffer_collect_em_upper = float(
            getattr(config, "prefix_buffer_collect_em_upper", 1.0)
        )
        # When True, prefix-buffer prefix turns are masked from the loss
        # (only the model's NEW generation after the prefix gets gradient).
        # Default False preserves the historical behavior in which the
        # buffered prefix's assistant content (incl. wrong <think>/<ANSWER>)
        # also received gradient. See config.py for the conflict rationale.
        self.mask_buffer_prefix = bool(getattr(config, "mask_buffer_prefix", False))

        self.prefix_buffer = None
        if self.prefix_buffer_mode_weight > 0:
            from .frozenlake_prefix_buffer import PrefixBuffer

            self.prefix_buffer = PrefixBuffer(
                max_size=self.prefix_buffer_size,
                wrong_ratio=self.prefix_buffer_wrong_ratio,
                max_staleness_steps=self.prefix_buffer_max_staleness_steps,
                min_size=self.prefix_buffer_min_size,
                max_per_question=self.prefix_buffer_max_per_question,
            )

        self.batch_counter = 0
        self.current_batch_uuid = str(uuid.uuid4())

        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")
        if (
            config.max_num_batched_tokens
            < config.prompt_length + config.response_length
        ):
            raise ValueError(
                "max_num_batched_tokens should be greater than prompt_length + response_length."
            )

        engine_kwargs = {}
        if processor is not None:
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len
            or (config.prompt_length + config.response_length) * 2,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=20000,
            max_num_seqs=32,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": self.single_turn_response_length,
            "detokenize": True,
            "stop": ["<|im_end|>"],  # FrozenLake assistant turns end at <|im_end|>
            "include_stop_str_in_output": True,
            "logprobs": 5,
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if key == "seed":
                # CRITICAL: do NOT propagate the rollout-config seed into
                # SamplingParams. vLLM's `SamplingParams.seed`, when set,
                # resets the per-sequence sampling PRNG -> identical-prompt
                # replicas (the GRPO n-per-question pattern) all sample the
                # same trajectory -> within-group variance = 0 -> GRPO
                # advantage = 0 -> no gradient. The config's `seed` is meant
                # for the LLM engine init (line 119), not per-sample sampling.
                continue
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)
        print(f"FrozenLake multi-turn sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _get_multi_turn_mask(self, response_tokens):
        """
        Generate multi-turn conversation attention mask, masking all special tokens and prompt parts.
        Only keeps assistant response content.

        Args:
            response_tokens: Token sequence containing multi-turn conversation

        Returns:
            attention_mask: Mask of same size as response_tokens, only keeping assistant response content
        """

        # Get special token IDs
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        user_id = self.tokenizer.convert_tokens_to_ids("user")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        pad_id = self.tokenizer.pad_token_id
        newline_id = 198

        attention_mask = torch.zeros_like(response_tokens)  # Initialize all to 0
        current_pos = 0
        in_assistant_response = (
            True  # Initial state is True, starting from assistant response
        )
        while current_pos < len(response_tokens):
            if response_tokens[current_pos] == im_end_id:
                # Encounter im_end_id, switch state
                in_assistant_response = False
                current_pos += 1
                continue

            if (
                current_pos + 2 < len(response_tokens)
                and response_tokens[current_pos] == im_start_id
                and response_tokens[current_pos + 1] == assistant_id
                and response_tokens[current_pos + 2] == newline_id
            ):
                # Find new assistant response start (including newline)
                in_assistant_response = True
                current_pos += 3  # Skip im_start, assistant and newline
                continue

            if in_assistant_response and response_tokens[current_pos] != pad_id:
                # In assistant response content and not padding
                attention_mask[current_pos] = 1

            current_pos += 1

        return attention_mask

    def _assistant_turn_starts(self, response_tokens):
        """Return list of (turn_idx, token_pos) for each `<|im_start|>assistant\\n` start.

        Turn 0 is implicit (the first assistant turn begins at position 0 by the
        rollout convention that the prompt ends right before the first assistant
        response). Turn i (i >= 1) starts at the position right AFTER the matched
        im_start/assistant/newline triple.
        """
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        newline_id = 198
        starts = [(0, 0)]
        turn_idx = 1
        pos = 0
        while pos + 2 < len(response_tokens):
            if (
                response_tokens[pos] == im_start_id
                and response_tokens[pos + 1] == assistant_id
                and response_tokens[pos + 2] == newline_id
            ):
                starts.append((turn_idx, pos + 3))
                turn_idx += 1
                pos += 3
            else:
                pos += 1
        return starts

    def get_prefix_buffer_stats(self):
        """Return prefix-buffer stats dict for wandb logging (empty when disabled).

        Called per-step by ray_trainer.py:874. Keys become `prefix_buffer/{k}`.
        """
        if self.prefix_buffer is None:
            return {}
        return self.prefix_buffer.get_stats()

    def update_prefix_buffer_step(self, step):
        """Update buffer step counter for staleness filtering (ray_trainer.py:873)."""
        if self.prefix_buffer is not None:
            self.prefix_buffer.update_step(step)

    def _compute_immediate_terminate(self, sample_info):
        """For prefix-buffer right-type rollouts: did the model terminate as its
        very first NEW turn (i.e. learned to early-stop on a correct prefix)?

        Returns True/False for eligible samples, None otherwise. Powers the
        `prefix_buffer/right_prefix_immediate_terminate_ratio` wandb metric
        (ray_trainer.py:760-778).
        """
        if sample_info.get("mode") != "prefix_buffer":
            return None
        if sample_info.get("prefix_type") != "right":
            return None
        prefix_len = sample_info.get("prefix_len")
        if prefix_len is None:
            return None
        new_text = sample_info.get("sequence", "")[prefix_len:]
        # The first new turn ends at the first <|im_end|>; if no such marker,
        # treat the entire new text as the first turn.
        first_turn_end = new_text.find("<|im_end|>")
        first_turn = new_text if first_turn_end < 0 else new_text[:first_turn_end]
        return '"name": "terminate"' in first_turn or '"name":"terminate"' in first_turn

    def _collect_buffer_entries(self, samples_info):
        """Post-rollout: add eligible normal-mode rollouts to the prefix buffer.

        Walks `samples_info` and, for each normal-mode rollout that terminates
        cleanly with between 1 and `prefix_buffer_max_route_turns` route turns,
        builds a `PrefixBufferEntry` and inserts it into `self.prefix_buffer`.
        Both right (strict EM) and wrong rollouts are stored; sampling-time
        `wrong_ratio` rebalances the mix.

        Calls `self.prefix_buffer.update_step(self.batch_counter)` at the end so
        staleness filtering tracks training step.
        """
        if self.prefix_buffer is None:
            return

        # Lazy-import reward helpers via importlib so this module stays standalone.
        # The rollout file lives at EasyR1/verl/workers/rollout/<this>.py; four
        # dirname() walks land us at EasyR1/, from which the reward module sits
        # under training/reward_function/frozenlake_score.py.
        import importlib.util as _ilu
        import os as _os

        _here = _os.path.dirname(
            _os.path.dirname(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            )
        )
        _score_path = _os.path.join(
            _here, "training", "reward_function", "frozenlake_score.py"
        )
        if not _os.path.exists(_score_path):
            print(
                f"[frozenlake prefix-buffer] ERROR: reward module not found at "
                f"{_score_path}; skipping buffer collection."
            )
            return
        _spec = _ilu.spec_from_file_location("_frozenlake_score", _score_path)
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        parse_turns = _m.parse_turns
        is_em = _m.is_em
        progress_rate = _m.progress_rate
        bfs_dist = _m.bfs_dist

        from .frozenlake_prefix_buffer import PrefixBufferEntry

        # OLF-style gate at collection time. When (em_lower, em_upper) is
        # tighter than (0.0, 1.0), we first compute the per-question group
        # em-mean across all eligible non-buffer samples in this batch, then
        # only collect entries from questions whose group mean lies strictly
        # in (em_lower, em_upper).
        _enable_olf_gate = (
            self.prefix_buffer_collect_em_lower > 0.0
            or self.prefix_buffer_collect_em_upper < 1.0
        )
        _qid_mean: dict = {}
        if _enable_olf_gate:
            from collections import defaultdict as _dd

            _qid_to_ems = _dd(list)
            for _sinfo in samples_info:
                if _sinfo.get("mode") == "prefix_buffer":
                    continue
                if _sinfo.get("finish_reason") in ("format_error", "render_error"):
                    continue
                _turns = parse_turns(_sinfo["sequence"], fmt=self.fmt)
                if not _turns:
                    continue
                if _turns[-1]["kind"] != "terminate" or _turns[-1]["actions"] is None:
                    continue
                _ms = _sinfo.get("map_spec")
                if not _ms:
                    continue
                _target = _ms.get("target_pos")
                if _target is None:
                    continue
                _layout = _ms["layout"]
                _start = _ms["start_pos"]
                _level = _ms["level"]
                _n_opt = bfs_dist(_layout, _start, _target, _level)
                if _n_opt is None:
                    continue
                _em = float(
                    is_em(
                        _turns[-1]["actions"], _layout, _start, _target, _level, _n_opt
                    )
                )
                _q = "{}:{}:{}:{}".format(
                    _level, _start, _target, "".join("".join(row) for row in _layout)
                )
                _qid_to_ems[_q].append(_em)
            _qid_mean = {q: sum(v) / len(v) for q, v in _qid_to_ems.items()}

        collected_wrong = 0
        collected_right = 0
        skipped_by_olf_gate = 0
        for sinfo in samples_info:
            # Skip buffer-mode rollouts (matches pointing-task behavior at
            # multi_turn_rollout_tool_use.py:2635-2707: only is_buffer_rollout
            # is skipped; normal AND random_start AND any other non-buffer mode
            # all feed the buffer). Without this, runs with normal_mode_weight=0
            # would never fill the buffer.
            if sinfo.get("mode") == "prefix_buffer":
                continue
            if sinfo.get("finish_reason") in ("format_error", "render_error"):
                continue
            # fmt MUST match the rollout output format. The parse_turns default
            # is "tag"; on a reflection_tag transcript that silently mis-parses
            # every <ANSWER> as a terminate and every <FINAL> as unparseable ->
            # 0 route turns -> every rollout skipped -> the buffer never fills.
            turns = parse_turns(sinfo["sequence"], fmt=self.fmt)
            if not turns:
                continue
            route_turns = [
                t for t in turns if t["kind"] == "route" and t["actions"] is not None
            ]
            if len(route_turns) < 1:
                continue
            if len(route_turns) > self.prefix_buffer_max_route_turns:
                continue
            # Must have a clean terminate turn at the end so we can split prefix vs suffix.
            if turns[-1]["kind"] != "terminate" or turns[-1]["actions"] is None:
                continue

            map_spec = sinfo.get("map_spec")
            if not map_spec:
                continue
            target = map_spec.get("target_pos")
            if target is None:
                # Defensive: per_example_specs must include target_pos. Skip if missing.
                continue
            layout = map_spec["layout"]
            start = map_spec["start_pos"]
            level = map_spec["level"]

            n_opt = bfs_dist(layout, start, target, level)
            if n_opt is None:
                continue

            final_actions = turns[-1]["actions"]
            em = bool(is_em(final_actions, layout, start, target, level, n_opt))
            pr = progress_rate(final_actions, layout, start, target, level, n_opt)

            # Split the sequence at the start of the LAST `<|im_start|>assistant\n`
            # marker; everything before = prefix_conversation; everything after =
            # the terminate turn (stored as correct_suffix only if EM).
            last_assistant = sinfo["sequence"].rfind("<|im_start|>assistant\n")
            if last_assistant < 0:
                continue
            prefix_conversation = sinfo["sequence"][:last_assistant]
            correct_suffix = sinfo["sequence"][last_assistant:] if em else None

            # Skip the base map image (index 0); keep the rendered feedback PNGs.
            # Also capture imgs[0] as base_map_image so future buffer-mode rollouts
            # of OTHER data items can rebind the rollout's base image to THIS entry's
            # question (without this, replayed entries mix Q_B's prefix text/feedback
            # with Q_A's base image — a multi-modal context corruption).
            imgs = sinfo.get("multi_modal_data", {}).get("image", [])
            base_map_image = imgs[0] if imgs else None
            feedback_images = list(imgs[1:]) if len(imgs) > 1 else []

            # question_id must be GLOBALLY unique per map. sinfo["question_id"]
            # is only the within-batch position (0..N-1) and repeats every
            # step, so the buffer's per-question cap (max_per_question) froze
            # the buffer at max_per_question * questions_per_worker (= 12) and
            # it never reached min_size. Derive a stable id from the map itself
            # so the buffer accumulates distinct maps across steps.
            # layout may be a list of strings OR a list of lists of chars;
            # "".join(row) normalizes both to a flat string.
            q_global = "{}:{}:{}:{}".format(
                level, start, target, "".join("".join(row) for row in layout)
            )
            # OLF-style gate: skip this entry if its question's group em-mean
            # is outside the configured band. Default band is (0.0, 1.0) so
            # this is a no-op unless the user explicitly tightens the gate.
            if _enable_olf_gate:
                _m = _qid_mean.get(q_global)
                if (
                    _m is None
                    or _m <= self.prefix_buffer_collect_em_lower
                    or _m >= self.prefix_buffer_collect_em_upper
                ):
                    skipped_by_olf_gate += 1
                    continue
            entry = PrefixBufferEntry(
                entry_id="",
                question_id=q_global,
                map_spec=map_spec,
                ground_truth=map_spec,
                prefix_conversation=prefix_conversation,
                prefix_type="right" if em else "wrong",
                num_route_turns=len(route_turns),
                last_actions=list(route_turns[-1]["actions"]),
                final_pr=float(pr),
                base_map_image=base_map_image,
                feedback_images=feedback_images,
                collection_step=self.batch_counter,
                correct_suffix=correct_suffix,
            )
            if self.prefix_buffer.add(entry):
                if em:
                    collected_right += 1
                else:
                    collected_wrong += 1

        if collected_wrong + collected_right > 0 or skipped_by_olf_gate > 0:
            _gate_msg = (
                (
                    f"  skipped_by_olf_gate={skipped_by_olf_gate}"
                    f" (band=({self.prefix_buffer_collect_em_lower:.2f},"
                    f"{self.prefix_buffer_collect_em_upper:.2f}))"
                )
                if _enable_olf_gate
                else ""
            )
            print(
                f"[frozenlake prefix-buffer] collected: wrong={collected_wrong} "
                f"right={collected_right} buffer_size={len(self.prefix_buffer)}{_gate_msg}"
            )
        self.prefix_buffer.update_step(self.batch_counter)

    def _get_prompts_and_indices(self, samples_info):
        """Get prompts and indices for samples that haven't stopped."""
        prompts, multi_modal_data, indices = [], [], []
        for index, info in enumerate(samples_info):
            if not info["stop"]:
                prompts.append(info["sequence"])
                multi_modal_data.append(info["multi_modal_data"])
                indices.append(info["index"])
        return prompts, multi_modal_data, indices

    def _is_finished(self, response):
        """Stop after a clean `terminate` or any parse failure."""
        return is_finished(response, fmt=self.fmt)

    def _multi_turn_generate(
        self, vllm_inputs=None, sampling_params=None, use_tqdm=False, map_specs=None
    ):
        """Generate multi-turn conversations using batch processing."""

        sampling_params = copy.deepcopy(sampling_params)

        # Prepare initial samples
        new_vllm_inputs = []
        for single_vllm_input in vllm_inputs:
            prompt = self.tokenizer.decode(
                single_vllm_input["prompt_token_ids"], skip_special_tokens=False
            )
            new_vllm_inputs.extend(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": copy.deepcopy(
                            single_vllm_input["multi_modal_data"]
                        ),
                    }
                    for _ in range(sampling_params.n)
                ]
            )

        sampling_params.n = 1
        sampling_params.detokenize = True

        import random as _random

        # Per-question mode assignment. One mode per question (group of n rollouts),
        # chosen by seeded RNG so the assignment is reproducible across batches.
        # Modes: "normal" | "random_start" | "prefix_buffer".
        # NOTE: sampling_params.n was just reset to 1 above, so we recover the
        # per-question multiplier from self.sampling_params (the rollout config's n).
        n_per_question = max(1, self.sampling_params.n)
        num_questions = len(new_vllm_inputs) // n_per_question
        sample_modes = ["normal"] * len(new_vllm_inputs)
        if self.per_question_mode_selection:
            base_seed = getattr(self.config, "seed", 0)
            for q in range(num_questions):
                enabled = []
                if self.normal_mode_weight > 0:
                    enabled.append(("normal", self.normal_mode_weight))
                if self.random_start_mode_weight > 0:
                    enabled.append(("random_start", self.random_start_mode_weight))
                if (
                    self.prefix_buffer_mode_weight > 0
                    and self.prefix_buffer is not None
                    and self.prefix_buffer.can_sample()
                ):
                    enabled.append(("prefix_buffer", self.prefix_buffer_mode_weight))
                if not enabled:
                    enabled = [("normal", 1.0)]
                rng = _random.Random(q + base_seed + self.batch_counter * 10000)
                chosen = rng.choices(
                    [m for m, _ in enabled], weights=[w for _, w in enabled], k=1
                )[0]
                for s in range(n_per_question):
                    sample_modes[q * n_per_question + s] = chosen

        # Initialize sample info
        samples_info = []
        batch_id = f"{self.rank}_{self.batch_counter}_{self.current_batch_uuid}"
        for index, item in enumerate(new_vllm_inputs):
            origin_image = item["multi_modal_data"]["images"][0]
            # Load image as PIL Image; cache by path to avoid re-decoding the
            # same base map across GRPO replicas and across steps.
            processed_image = _load_base_image_cached(origin_image)
            sample_info = {
                "prompt": item["prompt"],
                "sequence": item["prompt"],
                "multi_modal_data": {
                    "image": [processed_image]
                },  # vLLM expects 'image' not 'images'
                "response": "",
                "stop": False,
                "finish_reason": None,
                "index": index,
                "batch_id": batch_id,
                "turn_count": 0,  # Initialize turn counter
                "map_spec": map_specs[
                    index
                ],  # Per-example FrozenLake map specification
            }
            sample_info["mode"] = sample_modes[index]
            sample_info["question_id"] = index // n_per_question
            samples_info.append(sample_info)

        # --- Prefix buffer injection ---
        # Sample one entry per question (in question_id order) for buffer-mode questions.
        # Replaces the initial conversation with a stored prefix and extends the image
        # list with the entry's feedback images so the model resumes from a known partial
        # trajectory.
        buffer_question_ids = []
        for q in range(num_questions):
            if sample_modes[q * n_per_question] == "prefix_buffer":
                buffer_question_ids.append(q)

        if buffer_question_ids and self.prefix_buffer is not None:
            entries = self.prefix_buffer.sample(n=len(buffer_question_ids))
            # If buffer ran short (can_sample() guards min_size, not exact n), fall back
            # to normal mode for the un-served questions.
            if len(entries) < len(buffer_question_ids):
                for q in buffer_question_ids[len(entries) :]:
                    for s in range(n_per_question):
                        samples_info[q * n_per_question + s]["mode"] = "normal"
                buffer_question_ids = buffer_question_ids[: len(entries)]
            for q, entry in zip(buffer_question_ids, entries):
                for s in range(n_per_question):
                    idx = q * n_per_question + s
                    sinfo = samples_info[idx]
                    # Replace conversation with stored prefix; turn_count tracks how many
                    # route turns are already "spent" so the rollout budget accounts for it.

                    sinfo["sequence"] = (
                        entry.prefix_conversation + "<|im_start|>assistant\n"
                    )

                    sinfo["map_spec"] = entry.map_spec

                    if entry.base_map_image is not None:
                        sinfo["multi_modal_data"]["image"] = [
                            entry.base_map_image
                        ] + list(entry.feedback_images)
                    else:
                        sinfo["multi_modal_data"]["image"] = sinfo["multi_modal_data"][
                            "image"
                        ] + list(entry.feedback_images)
                    sinfo["turn_count"] = entry.num_route_turns
                    sinfo["prefix_buffer_entry"] = (
                        entry  # for post-hoc recycle / replay
                    )
                    # Stash per-sample fields the trainer reads to build prefix_buffer/*
                    # wandb metrics. See ray_trainer.py:702-815 + metrics.py:126-156.
                    sinfo["prefix_type"] = entry.prefix_type  # "right" | "wrong"
                    sinfo["num_route_turns"] = entry.num_route_turns
                    # `prefix_len` is the byte offset where the model's NEW
                    # generation begins. With the assistant-marker fix above,
                    # the sequence now has `prefix_conversation + "<|im_start|>assistant\n"`
                    # before any new content, so we add the marker's length so
                    # `_compute_immediate_terminate`'s `sequence[prefix_len:]`
                    # slice gives only the model's generation (and, if force_pointing
                    # is on, the forced_stub plus the model's generation).
                    sinfo["prefix_len"] = len(entry.prefix_conversation) + len(
                        "<|im_start|>assistant\n"
                    )
                    # Forced-route stub: only for "wrong" entries when enabled.
                    if self.prefix_buffer_force_route and entry.prefix_type == "wrong":
                        forced_stub = (
                            '```json\n{\n  "function_call": {\n    "name": "route",'
                        )
                        # Format-aware: force the OPENING of a revision turn so
                        # the model cannot immediately re-commit the wrong
                        # answer. The json stub above is the legacy default;
                        # reflection_tag revision turns open with <THINK>, and
                        # legacy tag route turns open with <think>.
                        if self.fmt == "reflection_tag":
                            forced_stub = "<THINK>"
                        elif self.fmt == "tag":
                            forced_stub = "<think>"
                        sinfo["sequence"] = sinfo["sequence"] + forced_stub
                        sinfo["forced_route_stub_len"] = len(forced_stub)
                    # Correct-replay: designate sample s=0 to skip generation entirely
                    # and splice in the stored correct_suffix verbatim.
                    if (
                        self.prefix_buffer_correct_replay
                        and entry.prefix_type == "right"
                        and entry.correct_suffix is not None
                        and s == 0
                    ):
                        sinfo["sequence"] = (
                            entry.prefix_conversation + entry.correct_suffix
                        )
                        sinfo["response"] = entry.correct_suffix
                        sinfo["stop"] = True
                        sinfo["finish_reason"] = "prefix_buffer_correct_replay"

        # Multi-turn generation loop
        num_llm_calls_available = copy.deepcopy(self.config.num_llm_calls_available) - 1
        turn_number = 0  # Track current turn number

        while num_llm_calls_available >= 0:
            turn_number += 1  # Increment turn counter
            num_llm_calls_available -= 1

            # Get active prompts
            input_prompts, multi_modal_data, indices = self._get_prompts_and_indices(
                samples_info
            )

            # Print number of active conversations
            print(
                f"###### Turn {turn_number}: {len(input_prompts)} active conversations ######"
            )

            if not input_prompts:  # All samples finished
                break

            # Prepare vLLM inputs
            vllm_inputs = [
                {
                    "prompt_token_ids": self.tokenizer.encode(
                        prompt, add_special_tokens=False
                    )[: self.config.prompt_length + self.config.response_length],
                    "multi_modal_data": mm_data,
                }
                for prompt, mm_data in zip(input_prompts, multi_modal_data)
            ]

            # Generate responses
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=use_tqdm
            )

            sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
            responses = [x.outputs[0].text for x in sorted_outputs]
            finish_reason = [x.outputs[0].finish_reason for x in sorted_outputs]
            stop_reason = [x.outputs[0].stop_reason for x in sorted_outputs]

            # Check if this is the last call
            if num_llm_calls_available == -1:
                for i, index in enumerate(indices):
                    samples_info[index]["response"] += responses[i]
                    samples_info[index]["sequence"] += responses[i]
                    samples_info[index]["stop"] = True
                    samples_info[index]["finish_reason"] = finish_reason[i]
                break

            # Check for early stopping
            is_finished_list = [
                self._is_finished(responses[i]) for i in range(len(finish_reason))
            ]

            if all(is_finished_list):  # All samples finished
                for i, index in enumerate(indices):
                    samples_info[index]["response"] += responses[i]
                    samples_info[index]["sequence"] += responses[i]
                    samples_info[index]["stop"] = True
                    samples_info[index]["finish_reason"] = finish_reason[i]
                break

            # Process responses: parse route/terminate, render feedback PNG, append image-only user turn.
            for i, index in enumerate(indices):
                samples_info[index]["response"] += responses[i]
                samples_info[index]["sequence"] += responses[i]
                if is_finished_list[i]:
                    # Terminate (or parse-failure): stop this conversation.
                    samples_info[index]["stop"] = True
                    samples_info[index]["finish_reason"] = finish_reason[i]
                    continue
                # Parse the route; if it fails, stop with format_error.
                kind, actions, err = parse_route_terminate(responses[i], fmt=self.fmt)
                if err is not None or kind != "route" or actions is None:
                    samples_info[index]["stop"] = True
                    samples_info[index]["finish_reason"] = "format_error"
                    continue
                # Render the trajectory feedback image on the example's map.
                spec = samples_info[index]["map_spec"]
                feedback = render_feedback_pil(
                    spec["layout"], spec["start_pos"], actions, spec["level"]
                )
                samples_info[index]["turn_count"] = turn_number
                if feedback is not None:
                    samples_info[index]["multi_modal_data"]["image"].append(feedback)
                    # For the reflection format, the feedback-turn user message
                    # MUST include the prompt that primes the model to emit
                    # <FINAL> or <THINK>/<ANSWER>. Mirrors the eval-side wiring
                    # (api_inference_frozenlake.py:_feedback_user_text_for_fmt).
                    # The "tag" / "json" formats keep the image-only feedback.
                    if self.fmt == "reflection_tag":
                        user_feedback = (
                            "\n<|im_end|>\n<|im_start|>user\n"
                            "<|vision_start|><|image_pad|><|vision_end|>"
                            + FEEDBACK_USER_PROMPT_REFLECTION
                            + "<|im_end|>\n<|im_start|>assistant\n"
                        )
                    else:
                        user_feedback = (
                            "\n<|im_end|>\n<|im_start|>user\n"
                            "<|vision_start|><|image_pad|><|vision_end|>"
                            "<|im_end|>\n<|im_start|>assistant\n"
                        )
                else:
                    # render failed -- stop with format_error rather than continue blind
                    samples_info[index]["stop"] = True
                    samples_info[index]["finish_reason"] = "render_error"
                    continue
                samples_info[index]["sequence"] += user_feedback

        # Add EOS tokens
        for sample_info in samples_info:
            if sample_info["finish_reason"] != "length":
                sample_info["sequence"] += self.tokenizer.eos_token
                sample_info["response"] += self.tokenizer.eos_token

        # Extract results
        responses = [sample_info["response"] for sample_info in samples_info]
        sequences = [sample_info["sequence"] for sample_info in samples_info]
        image_inputs = [
            sample_info["multi_modal_data"]["image"] for sample_info in samples_info
        ]  # vLLM expects 'image' not 'images'

        # Optional: save feedback images + transcripts to disk for inspection.
        # Index 0 of each sample's image list is the base map; indices 1..k are
        # feedback images for routes 1..k. We only persist the feedback images.
        if self.temp_dir:
            try:
                self.batch_counter += 1
                batch_dir = os.path.join(
                    self.temp_dir, "rank%s_batch%04d" % (self.rank, self.batch_counter)
                )
                log_samples = []
                for s in samples_info:
                    imgs = s["multi_modal_data"].get("image", [])
                    feedback = imgs[1:] if len(imgs) > 1 else []
                    # Save BOTH the base map image (so the dump is fully self-
                    # contained; downstream viz can confirm Q_A vs Q_B match)
                    # AND the per-sample mode/prefix_type labels so viz tools
                    # don't have to guess via heuristics.
                    base_img = imgs[0] if imgs else None
                    log_samples.append(
                        {
                            "sample_idx": int(s["index"]),
                            "sequence": s["sequence"],
                            "map_spec": s.get("map_spec"),
                            "finish_reason": s.get("finish_reason"),
                            "feedback_images": feedback,
                            "base_map_image": base_img,
                            "mode": s.get("mode"),
                            "prefix_type": s.get("prefix_type"),  # None for non-buffer
                        }
                    )
                save_batch_logs(batch_dir, log_samples)
            except Exception as _e:
                print("[frozenlake rollout] failed to save batch logs:", _e)

        # Debug: Check if responses are empty
        empty_responses = sum(1 for r in responses if not r.strip())
        if empty_responses > 0:
            print(f"Warning: {empty_responses}/{len(responses)} responses are empty")
            # Print first few sequences to debug
            for i in range(min(3, len(sequences))):
                print(f"Sequence {i} length: {len(sequences[i])}")
                print(f"Response {i}: '{responses[i][:100]}...'")

        # Post-rollout: collect eligible normal-mode rollouts into the prefix buffer.
        # No-op when self.prefix_buffer is None (i.e. prefix_buffer_mode_weight == 0).
        self._collect_buffer_entries(samples_info)

        return responses, sequences, image_inputs, samples_info

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate sequences for FrozenLake multi-turn RL.

        Args:
            prompts: Input data containing prompts and metadata

        Returns:
            DataProto: Generated sequences with multi-turn data
        """
        # Extract input tensors
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not working properly.")

        # Prepare vLLM inputs
        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
            ):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": multi_modal_data,
                    }
                )
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)}
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # Decode per-example map spec from the dataset's map field.
        # The training path (MultiTurnRLHFDataset) names it `answer`; the
        # validation path (RLHFDataset) renames it to `ground_truth`. Accept
        # either to keep both code paths working.
        import json as _json

        per_example_specs = []
        answers = non_tensor_batch.get("answer", non_tensor_batch.get("ground_truth"))
        if answers is None:
            raise ValueError(
                "frozenlake_multi_turn rollout requires `answer` or `ground_truth` "
                "in non_tensor_batch (got keys: %s)" % sorted(non_tensor_batch.keys())
            )
        for a in answers:
            if isinstance(a, str):
                a = _json.loads(a)
            per_example_specs.append(
                {
                    "layout": a["layout"],
                    "start_pos": int(a["start_pos"]),
                    "target_pos": int(a["target_pos"]),
                    "level": int(a["level"]),
                }
            )

        # Generate multi-turn responses
        with self.update_sampling_params(**prompts.meta_info):
            # repeat per-sample specs to match the `n` sampling expansion
            per_sample_specs = []
            for s in per_example_specs:
                per_sample_specs.extend([s] * self.sampling_params.n)
            responses, sequences, image_inputs, samples_info = (
                self._multi_turn_generate(
                    vllm_inputs=vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                    map_specs=per_sample_specs,
                )
            )

            # Handle sampling parameter n > 1
            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(
                    attention_mask, self.sampling_params.n
                )
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                # Expand any non_tensor_batch fields that came in from the dataset
                # (e.g. `answer`, `task`, `map_id`) so they line up with the n-fold
                # sample expansion. `raw_prompt_ids` and `multi_modal_data` were
                # popped earlier and will be re-added at the expanded size below.
                for _k, _v in list(non_tensor_batch.items()):
                    non_tensor_batch[_k] = _repeat_interleave(
                        _v, self.sampling_params.n
                    )

        # FOUR-FIELD ALIGNMENT FIX (field 4): for any sample that was rolled out
        # in prefix_buffer mode, the rollout already overrode `sinfo["map_spec"]`
        # to the buffer entry's spec (see _multi_turn_generate). We now propagate
        # the same override into the non_tensor_batch ground-truth field so the
        # reward function scores against the buffer entry's question, not the
        # data item's. Without this, buffer-mode rollouts are scored against the
        # wrong question and right-prefix replays become structurally em=0.
        gt_key = (
            "answer"
            if "answer" in non_tensor_batch
            else ("ground_truth" if "ground_truth" in non_tensor_batch else None)
        )
        if gt_key is not None and len(samples_info) == len(non_tensor_batch[gt_key]):
            gt_arr = list(non_tensor_batch[gt_key])
            n_overridden = 0
            for i, sinfo in enumerate(samples_info):
                if sinfo.get("mode") == "prefix_buffer":
                    gt_arr[i] = _json.dumps(sinfo["map_spec"])
                    n_overridden += 1
            non_tensor_batch[gt_key] = np.array(gt_arr, dtype=object)
            if n_overridden > 0:
                print(
                    f"[frozenlake rollout] overrode {gt_key} for {n_overridden} "
                    f"buffer-mode samples to align with their stored map_spec"
                )

        # Update raw prompt IDs with complete sequences
        non_tensor_batch["raw_prompt_ids"] = [
            self.tokenizer.encode(sequence, add_special_tokens=False)[
                : self.config.prompt_length + self.config.response_length
            ]
            for sequence in sequences
        ]

        # Process sequences for tensor outputs
        valid_prompt_len = torch.sum(attention_mask, dim=-1)
        response_ids = []
        response_mask = []
        response_position_ids = []
        model_inputs = []
        multi_turn_mask = []

        for idx, prompt_len in enumerate(valid_prompt_len):
            # Process sequence with processor
            inputs = self.processor(
                text=sequences[idx],
                images=image_inputs[idx],
                add_special_tokens=False,
                return_tensors="pt",
            )

            # Get position IDs (use correct get_rope_index based on model type).
            # NOTE: Qwen3-VL reuses Qwen2VLImageProcessorFast, so inspect the TOP-LEVEL
            # processor class (mirrors multi_turn_rollout.py and vllm_rollout_spmd.py).
            processor_class_name = (
                self.processor.__class__.__name__ if self.processor else ""
            )
            is_qwen3 = "Qwen3" in processor_class_name
            try:
                if is_qwen3:
                    from ...models.transformers.qwen3_vl import get_rope_index
                else:
                    from ...models.transformers.qwen2_vl import get_rope_index
                new_position_ids = get_rope_index(
                    self.processor,
                    input_ids=inputs["input_ids"][0],
                    image_grid_thw=inputs["image_grid_thw"],
                    attention_mask=inputs["attention_mask"][0],
                )
            except ImportError:
                # Fallback if get_rope_index is not available
                seq_len = inputs["input_ids"][0].size(0)
                repeat_dim = 4 if is_qwen3 else 3
                new_position_ids = (
                    torch.arange(seq_len, device=inputs["input_ids"][0].device)
                    .unsqueeze(0)
                    .repeat(repeat_dim, 1)
                )

            # Validate input consistency
            try:
                assert (
                    torch.sum(
                        input_ids[idx][-prompt_len:].cpu()
                        == inputs["input_ids"][0][:prompt_len].cpu()
                    )
                    == prompt_len
                ), f"Input IDs mismatch at batch index {idx}"

                assert (
                    torch.sum(
                        attention_mask[idx][-prompt_len:].cpu()
                        == inputs["attention_mask"][0][:prompt_len].cpu()
                    )
                    == prompt_len
                ), f"Attention mask mismatch at batch index {idx}"
            except Exception as _val_err:
                print(
                    f"WARNING: Input ID/mask validation failed at batch index {idx}: {_val_err}"
                )
                print(
                    self.processor.tokenizer.decode(
                        input_ids[idx][-prompt_len:].cpu(), skip_special_tokens=False
                    )
                )
                print(
                    self.processor.tokenizer.decode(
                        inputs["input_ids"][0][:prompt_len].cpu(),
                        skip_special_tokens=False,
                    )
                )

            # Extract response parts (slice RELATIVE to prompt_len)
            resp_end = prompt_len + self.config.response_length
            response_ids.append(inputs["input_ids"][0][prompt_len:resp_end])
            response_mask.append(inputs["attention_mask"][0][prompt_len:resp_end])

            # Pad position IDs for response
            pad_position_ids = VF.pad_sequence_to_length(
                new_position_ids[:, prompt_len:resp_end],
                max_seq_len=self.config.response_length,
                pad_token_id=0,
                left_pad=False,
            ).to(input_ids.device)
            response_position_ids.append(pad_position_ids)

            # Generate multi-turn mask
            tmp_multi_turn_mask = self._get_multi_turn_mask(
                inputs["input_ids"][0][prompt_len:resp_end]
            )

            # Apply random-start loss masking for samples in random_start mode.
            # Walks the response tokens to find the first `k`th `<|im_start|>assistant\n`
            # boundary and zeros out all mask positions BEFORE turn `k` so the loss only
            # flows through turns `k..final`. See plan Task 4 for the full rationale.
            import random as _random

            sinfo = samples_info[idx]
            if sinfo.get("mode") == "random_start":
                starts = self._assistant_turn_starts(
                    inputs["input_ids"][0][prompt_len:resp_end]
                )
                num_turns = len(starts)
                if num_turns >= 2:
                    if self.random_start_mask_all_until_final:
                        keep_from_turn = num_turns - 1  # final assistant turn only
                    else:
                        # `random_start_turn_min=0` is supported and means "K=0
                        # is in the sampling support; when sampled, no masking
                        # is applied (the entire trajectory is trained, exactly
                        # as in normal_mode)." This lets random_start mode put
                        # gradient on turn 1 for ~1/N of rollouts while still
                        # masking-toward-revision for the rest, keeping turn-1
                        # and revision skills from becoming homogeneous.
                        min_k = max(0, self.random_start_turn_min)
                        max_k = max(min_k, num_turns - 1)
                        keep_from_turn = _random.randint(min_k, max_k)
                    # Zero out mask for tokens before turn `keep_from_turn`.
                    # When keep_from_turn == 0, no masking is applied — the
                    # full multi-turn loss mask is preserved so turn 1 receives
                    # gradient just like normal_mode.
                    if keep_from_turn > 0:
                        cutoff_pos = starts[keep_from_turn][1]
                        tmp_multi_turn_mask[:cutoff_pos] = 0
                    # Record the sampled start turn for the random_restart/* metrics
                    # built by compute_random_restart_metrics (metrics.py:126-156).
                    sinfo["start_turn"] = keep_from_turn
            elif sinfo.get("mode") == "prefix_buffer" and self.mask_buffer_prefix:
                # Zero the loss on the buffered prefix's assistant turns so the
                # gradient flows only through the model's NEW generation. The
                # prefix has `num_route_turns` assistant turns (route turns of
                # the original on-policy rollout, excluding the dropped final
                # terminate turn). `_assistant_turn_starts` reports the start
                # offset of each assistant turn in the response slice; the
                # (num_route_turns)-th entry corresponds to the first turn of
                # the new generation, so we zero the mask up to that position.
                # Defensive: if the model failed to begin a new turn at all
                # (degenerate; `starts` shorter than num_prefix_turns+1) we
                # leave the mask untouched rather than zero everything —
                # preserves SOME training signal on the degenerate rollout.
                starts = self._assistant_turn_starts(
                    inputs["input_ids"][0][prompt_len:resp_end]
                )
                num_prefix_turns = int(sinfo.get("num_route_turns", 0))
                if num_prefix_turns > 0 and len(starts) > num_prefix_turns:
                    cutoff_pos = starts[num_prefix_turns][1]
                    tmp_multi_turn_mask[:cutoff_pos] = 0
                # Record where the new generation began for any future metric.
                sinfo["start_turn"] = num_prefix_turns

            multi_turn_mask.append(tmp_multi_turn_mask)

            # Prepare model inputs
            inputs.pop("input_ids")
            inputs.pop("attention_mask")
            model_inputs.append(dict(inputs))

        # Pad response IDs
        response_ids = VF.pad_2d_list_to_length(
            response_ids, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)

        non_tensor_batch["multi_modal_inputs"] = model_inputs

        # Create final tensors
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_position_ids = torch.stack(response_position_ids, dim=0).to(
            input_ids.device
        )
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_mask = VF.pad_2d_list_to_length(
            response_mask, 0, max_length=self.config.response_length
        ).to(input_ids.device)

        multi_turn_mask = VF.pad_2d_list_to_length(
            multi_turn_mask, 0, max_length=self.config.response_length
        ).to(input_ids.device)

        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # Print statistics
        valid_lengths = torch.sum(attention_mask, dim=1)
        max_valid_length = torch.max(valid_lengths).cpu()
        min_valid_length = torch.min(valid_lengths).cpu()
        avg_valid_length = torch.mean(valid_lengths.float()).cpu()

        print(f"Size of prompt_ids: {input_ids.size()}")
        print(f"Size of response_ids: {response_ids.size()}")
        print(f"Size of sequence_ids: {sequence_ids.size()}")
        print(
            f"Valid Length - Max: {max_valid_length}, Min: {min_valid_length}, Avg: {avg_valid_length:.2f}"
        )

        # Create final batch
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
                "multi_turn_mask": multi_turn_mask,
            },
            batch_size=batch_size,
        )

        # --- Per-sample metadata for wandb metrics ---
        # ray_trainer.py:702-815 reads these to build `prefix_buffer/acc_*`,
        # `prefix_buffer/reflection_*`, and `prefix_buffer/right_prefix_*` keys.
        # metrics.py:126 reads `start_turn` to build `random_restart/*` keys.
        is_pb_arr = [bool(s.get("mode") == "prefix_buffer") for s in samples_info]
        pb_turns_arr = [
            int(s.get("num_route_turns", 0)) if s.get("mode") == "prefix_buffer" else 0
            for s in samples_info
        ]
        pb_types_arr = [
            s.get("prefix_type") if s.get("mode") == "prefix_buffer" else None
            for s in samples_info
        ]
        pb_imm_arr = [self._compute_immediate_terminate(s) for s in samples_info]
        start_turn_arr = [int(s.get("start_turn", 0)) for s in samples_info]
        non_tensor_batch["is_prefix_buffer_rollout"] = np.array(is_pb_arr, dtype=object)
        non_tensor_batch["prefix_buffer_num_turns"] = np.array(
            pb_turns_arr, dtype=object
        )
        non_tensor_batch["prefix_buffer_type"] = np.array(pb_types_arr, dtype=object)
        non_tensor_batch["prefix_buffer_immediate_terminate"] = np.array(
            pb_imm_arr, dtype=object
        )
        non_tensor_batch["start_turn"] = np.array(start_turn_arr, dtype=object)

        # Convert non-tensor batch to numpy arrays
        for key, value in non_tensor_batch.items():
            if not isinstance(value, np.ndarray):
                non_tensor_batch[key] = np.array(value, dtype=object)

        print(
            repr(
                self.tokenizer.decode(
                    batch["responses"][0][batch["response_mask"][0] == 1]
                )
            )
        )

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
