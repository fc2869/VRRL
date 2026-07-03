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
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    # Override vLLM's torch.compile cudagraph mode. Options: None (vLLM default),
    # "NONE", "PIECEWISE", "FULL". Setting "PIECEWISE" is the Qwen3-Next recipe
    # workaround for the "illegal memory access" vLLM V1 CUDA graph fault on
    # multi-modal models — see https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html
    cudagraph_mode: Optional[str] = None
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    # vLLM multi-modal processor cache size in GiB; None lets vLLM decide its default.
    mm_processor_cache_gb: Optional[float] = None
    val_override_config: dict[str, Any] = field(default_factory=dict)
    # Multi-turn specific parameters
    max_turns: int = 8
    num_llm_calls_available: int = 8
    single_turn_response_length: int = 500
    crop_size: int = 200
    # max_pixels for image resizing during crop/overlay generation.
    # Must match the training data pipeline (data.max_pixels) so that
    # pixel coordinates used for OOB checks and cropping are consistent.
    max_pixels: int = 1600 * 28 * 28
    # Tool-use cropping behavior
    multi_point_enclosing_crop: bool = False
    enclosing_padding: int = 50
    temp_dir: str = "temp_crops"
    # Out-of-bounds early stopping behavior
    oob_early_stop: bool = False
    # Format error early stopping: stop rollout immediately when format error occurs
    # (invalid JSON or coordinates cannot be extracted). The error turn is still included in trace.
    format_error_early_stop: bool = False
    # Stop early if a pointing coordinate repeats across assistant turns
    repeat_pointing_early_stop: bool = False
    # Provide corrective feedback text between turns
    provide_feedback: bool = False
    feedback_distance_threshold: int = -1
    # Suppress axis direction if |dx| or |dy| <= this epsilon
    feedback_axis_epsilon: int = 5
    # Probability of masking out the first assistant turn to force learning reflection
    # 0.0 = never mask (always learn from first turn), 1.0 = always mask (never learn from first turn)
    mask_first_turn_prob: float = 0.0
    # Random restart training: sample a start turn k uniformly, mask turns 1..k-1, optimize from turn k onward
    # This teaches one-shot solving (k=1) and self-correction from arbitrary states (k>1)
    # Set to True to enable, False to disable. Supersedes mask_first_turn_prob if enabled.
    random_start_turn: bool = False
    # Minimum start turn (1 = can start from turn 1, 2 = always mask at least turn 1, etc.)
    random_start_turn_min: int = 1
    # Number of completions per question to apply random start turn sampling (subset of n)
    random_start_rollout_num: int = 0
    # If True, when using random_start_turn, mask all assistant turns except the final two (pointing + terminate)
    random_start_mask_all_until_final: bool = False
    # Reflection completions (random-init rollouts)
    enable_random_init_rollout: bool = False
    random_init_rollout_num: int = 4
    random_init_min_dist: int = 100
    random_init_max_dist: int = 180
    random_init_max_step_size: int = 10000
    min_step_size: int = 0
    # Shared first turn: all rollouts share the same templated first turn prefix
    # Unlike random_init where each rollout has its own first turn, shared_first_turn
    # creates a single first turn from template and all rollouts continue from there
    enable_shared_first_turn: bool = False
    shared_first_turn_rollout_num: int = 8  # Number of rollouts that share the first turn
    shared_first_turn_min_dist: int = 100  # Min distance from GT for the shared starting point
    shared_first_turn_max_dist: int = 180  # Max distance from GT for the shared starting point
    shared_first_turn_random_anywhere: bool = False  # If True, ignore min/max_dist and sample anywhere on image
    # Per-question mode selection: instead of splitting n rollouts among modes,
    # select ONE mode for all n rollouts per question. Mode is chosen randomly based on weights.
    per_question_mode_selection: bool = False
    # Weights for each mode when per_question_mode_selection=True
    # weight > 0 means the mode is enabled, weight = 0 means disabled
    # Modes are selected randomly based on these weights (no need to set enable flags separately)
    normal_mode_weight: float = 1.0
    random_start_mode_weight: float = 0.0  # Set > 0 to enable random_start mode
    random_init_mode_weight: float = 0.0   # Set > 0 to enable random_init mode
    shared_first_turn_mode_weight: float = 0.0  # Set > 0 to enable shared_first_turn mode
    prefix_buffer_mode_weight: float = 0.0  # Set > 0 to enable prefix_buffer mode

    # Prefix Buffer Configuration
    # The prefix buffer stores on-policy prefixes from previous rollouts for training
    # correction (wrong prefixes) and early stopping (right prefixes)
    prefix_buffer_size: int = 1000  # Maximum number of entries in the buffer
    prefix_buffer_wrong_ratio: float = 0.8  # Ratio of wrong prefixes to sample (vs right prefixes)
    prefix_buffer_max_staleness_steps: int = 100  # Max steps before an entry is considered stale
    prefix_buffer_min_size: int = 100  # Minimum buffer size before sampling is enabled
    prefix_buffer_max_per_question: int = 3  # Max entries per question (ensures question diversity)
    # Force the model to start with a pointing call after a prefix buffer prefix.
    # When True, the assistant's next turn after the buffered prefix is seeded with the
    # beginning of a "pointing" function call (```json\n{\n  "function_call": {\n    "name": "pointing",)
    # so the model cannot immediately terminate. The forced prefix IS included in the loss.
    prefix_buffer_force_pointing: bool = False
    # Recycle failed prefix buffer rollouts: if a prefix buffer rollout's completion
    # does not achieve coordinate reward=1, (a) refresh the original prefix in the
    # buffer so it stays fresh, and (b) add the extended wrong completion as a new
    # wrong prefix (with more pointing turns than the original).
    prefix_buffer_recycle: bool = False
    # Maximum number of pointing turns a prefix buffer entry can have.
    # Entries with more pointing turns than this are not added to the buffer.
    # Defaults to 6 (= max_turns - 2), ensuring the model always has room for
    # at least one more pointing turn + terminate when replaying the prefix.
    prefix_buffer_max_pointing_turns: int = 6
    # Correct replay: for "right" prefix buffer entries, include the original
    # correct terminate call as 1 of the N rollouts and sample N-1 new ones.
    # GRPO over all N creates a contrastive signal encouraging early termination.
    prefix_buffer_correct_replay: bool = False
    # OLF-style gate at buffer collection time. Buffer entries are only stored
    # from questions whose group em-mean across the batch lies strictly in
    # (em_lower, em_upper). Defaults (0.0, 1.0) apply no gating. Tighten to
    # e.g. (0.1, 1.0) to drop "hopeless" wrong rollouts (group em=0).
    prefix_buffer_collect_em_lower: float = 0.0
    prefix_buffer_collect_em_upper: float = 1.0
    # Mask the buffered prefix's assistant turns from the loss when sampling
    # a prefix_buffer rollout. When True, only the model's NEW generation
    # (turns AFTER the prefix) contributes gradient — the wrong prefix's
    # <think>/<ANSWER> are treated as a fixed condition, not as training
    # targets, matching the semantics of multi_turn_rollout_tool_use.py.
    # When False (default), every assistant token in the prefix receives the
    # rollout's group-relative advantage, which can interfere with turn-1
    # training in regimes where prefix_buffer is the dominant turn-1 signal
    # (normal_mode_weight=0 with random_start_turn_min=0).
    mask_buffer_prefix: bool = False
    # Distance threshold for reward calculation (used to determine correctness)
    # Will be populated from worker.reward.reward_function_kwargs.distance_threshold
    reward_distance_threshold: int = -1
    # Use vLLM structured outputs (guided JSON decoding) to enforce valid JSON responses.
    # When enabled, the model is constrained to produce well-formed JSON matching the
    # expected {think, function_call: {name, arguments}} schema on every generation turn,
    # eliminating malformed/truncated JSON at the cost of slightly slower inference.
    use_guided_json: bool = False
    # Use Qwen3 <tool_call> format for forced pointing prefix and shared first-turn templates.
    # When True, forced prefixes and injected responses use the computer_use/action=point
    # schema with coordinates in [0, 1000] normalized space instead of the old JSON format.
    qwen3_tool_use_format: bool = False
    # Whether model outputs coordinates in Qwen3 [0, 1000] normalized range.
    # When True, predicted coordinates are converted from [0,1000] to pixel space for
    # cropping/OOB checks, and injected coordinates are converted from pixel to [0,1000].
    # When False, coordinates are assumed to be in inference pixel space already.
    # This is independent of qwen3_tool_use_format — a model can use <tool_call> XML
    # format but output pixel coordinates (not normalized to 0-1000).
    use_qwen3_normalization_range: bool = True
    # Whether ground truth is a bounding box [[x1, y1, x2, y2]] instead of a point [x, y].
    # Will be populated from worker.reward.reward_function_kwargs.use_bbox
    use_bbox: bool = False
    # Output format the rolled-out policy speaks. Threaded into the FrozenLake
    # multi-turn rollout (`MultiTurnRolloutFrozenLake.fmt`) which forwards it
    # to `parse_route_terminate` / `is_finished` so the parser matches the
    # model. One of:
    #   "tag"            -- lowercase <route>/<answer> format
    #   "reflection_tag" -- uppercase <ANSWER>/<FINAL> with <THINK> on
    #                       revision turns
    #   "json"           -- JSON function_call schema
    fmt: str = "tag"
    # below are auto keys
    # Propagated from data.train_files so the rollout worker can inspect the dataset path
    train_files: str = field(default="", init=False)
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
