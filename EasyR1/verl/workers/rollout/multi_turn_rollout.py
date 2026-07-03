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
import tempfile
import shutil
import copy
import uuid
from contextlib import contextmanager
from typing import Any, Optional, Union, List, Dict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from PIL import Image
from qwen_vl_utils import fetch_image

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig
from .vllm_utils import add_engine_arg_if_supported


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any], min_pixels: int, max_pixels: int, video_fps: float
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(process_video(video, min_pixels, max_pixels, video_fps))

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


def crop_square_around_point(image_input, point: tuple, crop_size: int = 200, 
                           rank: int = 0, batch_id: str = "", sample_index: int = 0, turn_count: int = 0) -> str:
    """
    Crop a square region centered at the given point from the image and save to temp file.
    The cropped image will have a red point added at the extracted coordinate.
    
    Args:
        image_input: Either a path to the original image (str) or a PIL Image object
        point: (x, y) coordinates of the center
        crop_size: Size of the square crop
        rank: Rank of the process
        batch_id: Unique batch identifier
        sample_index: Index of the sample in the batch
        turn_count: Current turn number in the conversation
        
    Returns:
        Path to the saved crop image with red point
    """
    # First add the red point to the original image
    from tools import add_sign_to_image, plot_points, get_image_size_during_inference
    
    # Create the points XML format for the extracted coordinate
    x, y = point
    points_xml = f'<points x1="{x}" y1="{y}">extracted_point</points>'
    
    # Handle both image paths and PIL Image objects
    if isinstance(image_input, str):
        # It's a file path
        image_path = image_input
        # Add red point to the original image
        image_with_point = add_sign_to_image(
            image_path=image_path,
            response=points_xml,
            plot_func=plot_points,
            max_dim=max(get_image_size_during_inference(image_path)),
            draw_text=False,
            radius=5,
            alpha=128
        )
    else:
        # It's a PIL Image object - save it temporarily first
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image_input.save(tmp_file.name)
            temp_img_path = tmp_file.name
        
        # Add red point to the original image
        image_with_point = add_sign_to_image(
            image_path=temp_img_path,
            response=points_xml,
            plot_func=plot_points,
            max_dim=max(get_image_size_during_inference(temp_img_path)),
            draw_text=False,
            radius=5,
            alpha=128
        )
        
        # Clean up temporary file
        os.unlink(temp_img_path)
    
    # Now crop the image with the red point
    x, y = point
    # If point is out of image boundary, do not crop and signal upstream via empty path
    if x < 0 or y < 0 or x >= image_with_point.width or y >= image_with_point.height:
        return ""
    half = crop_size // 2
    left = max(x - half, 0)
    upper = max(y - half, 0)
    right = min(x + half, image_with_point.width)
    lower = min(y + half, image_with_point.height)
    
    # Ensure valid crop coordinates
    if left >= right or upper >= lower:
        # If coordinates are invalid, return empty path
        return ""
    
    crop = image_with_point.crop((left, upper, right, lower))
    
    # Save to temp file
    temp_dir = os.environ.get("TEMP_CROP_DIR", "temp_crops")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate unique filename with the specified format
    filename = f"rank{rank}_batch{batch_id}_sample_{sample_index}_turn_{turn_count}.png"
    temp_path = os.path.join(temp_dir, filename)
    crop.save(temp_path)
    return temp_path


def parse_points_tag(points_tag: str) -> tuple:
    """
    Parse a <points ...> tag and return (x, y) coordinates.
    """
    import re
    x_match = re.search(r'x1="([-\d.]+)"', points_tag)
    y_match = re.search(r'y1="([-\d.]+)"', points_tag)
    if x_match and y_match:
        return (int(float(x_match.group(1))), int(float(y_match.group(1))))
    return None


def process_single_response(response: str, original_image_path: str, crop_size: int = 200,
                          rank: int = 0, batch_id: str = "", sample_index: int = 0, turn_count: int = 0):
    """Process a single response to extract points and generate crop images."""
    import re
    
    # Extract points from response
    points = []
    points_matches = re.findall(r'<points[^>]*>.*?</points>', response, re.DOTALL)
    for match in points_matches:
        point = parse_points_tag(match)
        if point:
            points.append(point)
    
    # Use only the last point if multiple points found
    if len(points) > 1:
        points = [points[-1]]  # Keep only the last point
    elif len(points) == 0:
        return None, []
    
    # Generate crop image for the single point with unique naming
    crop_path = crop_square_around_point(
        original_image_path, points[0], crop_size, 
        rank=rank, batch_id=batch_id, sample_index=sample_index, turn_count=turn_count
    )
    
    # If crop_path is empty, it means coordinates were invalid
    if not crop_path:
        return points, []
    
    return points, [crop_path]


class MultiTurnRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A multi-turn vLLM rollout for GRPO training with conversation history.

        Args:
            model_path: Path to the model
            config: RolloutConfig
            tokenizer: The task/model tokenizer
            processor: Optional processor for multi-modal data
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        self.tokenizer = tokenizer
        self.processor = processor
        
        # Multi-turn specific parameters
        self.max_turns = getattr(config, 'max_turns', 8)
        self.num_llm_calls_available = getattr(config, 'num_llm_calls_available', 8)
        self.single_turn_response_length = getattr(config, 'single_turn_response_length', 500)
        self.crop_size = getattr(config, 'crop_size', 200)
        self.temp_dir = getattr(config, 'temp_dir', "temp_crops")
        
        # Add batch counter and UUID for unique image naming
        self.batch_counter = 0
        self.current_batch_uuid = str(uuid.uuid4())
        
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            mm_cache_gb = getattr(config, "mm_processor_cache_gb", None)
            if mm_cache_gb is not None:
                add_engine_arg_if_supported(engine_kwargs, "mm_processor_cache_gb", mm_cache_gb)
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or (config.prompt_length + config.response_length) * 2,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=20000, #config.max_num_batched_tokens,
            max_num_seqs=10,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            # model_impl='transformers',  # or 'vllm' (default)
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": self.single_turn_response_length,  # Use single turn response length
            "detokenize": True,
            "stop": ["</tool_call>"],  # Stop generation when </tool_call> token (ID: 151658) is generated
            "include_stop_str_in_output": True,
            "logprobs": 5,
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if key == "seed":
                # See vllm_rollout_spmd.py / multi_turn_rollout_frozenlake.py
                # for the rationale: SamplingParams.seed pins per-sequence
                # PRNG -> identical-prompt replicas collapse to identical
                # outputs -> GRPO group variance = 0. The engine seed is set
                # in the LLM() constructor only.
                continue
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Multi-turn sampling params: {sampling_kwargs}.")
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
        in_assistant_response = True  # Initial state is True, starting from assistant response
        while current_pos < len(response_tokens):
            if response_tokens[current_pos] == im_end_id:
                # Encounter im_end_id, switch state
                in_assistant_response = False
                current_pos += 1
                continue
                
            if (current_pos + 2 < len(response_tokens) and 
                response_tokens[current_pos] == im_start_id and 
                response_tokens[current_pos + 1] == assistant_id and
                response_tokens[current_pos + 2] == newline_id):
                # Find new assistant response start (including newline)
                in_assistant_response = True
                current_pos += 3  # Skip im_start, assistant and newline
                continue
                
            if in_assistant_response and response_tokens[current_pos] != pad_id:
                # In assistant response content and not padding
                attention_mask[current_pos] = 1
                
            current_pos += 1

        return attention_mask

    def _get_prompts_and_indices(self, samples_info):
        """Get prompts and indices for samples that haven't stopped."""
        prompts, multi_modal_data, indices = [], [], []
        for index, info in enumerate(samples_info):
            if not info['stop']:
                prompts.append(info['sequence'])
                multi_modal_data.append(info['multi_modal_data'])
                indices.append(info['index'])
        return prompts, multi_modal_data, indices

    def _is_finished(self, response):
        """Check if the response indicates the conversation should stop."""
        import re

        if not response.startswith("<|im_start|>assistant\n"):
            response = "<|im_start|>assistant\n" + response
        
        # Check if response contains <answer> tag (original logic)
        if "<answer>" in response:
            return True
        
        # Check for repeated coordinates across the entire response (excluding <answer> tag)
        # First, remove the <answer> tag content to exclude it from repetition check
        response_without_answer = re.sub(r'<answer>.*?</answer>', '', response, flags=re.DOTALL)
        
        # Extract all coordinate points from the response (excluding answer tag)
        points_pattern = r'<points\s+x1="([^"]+)"\s+y1="([^"]+)">'
        points_matches = re.findall(points_pattern, response_without_answer)
        
        # Convert to set of coordinate tuples to detect duplicates
        seen_coordinates = set()
        for x, y in points_matches:
            try:
                coord = (int(float(x)), int(float(y)))
                if coord in seen_coordinates:
                    return True  # Stop if same coordinate appears again (excluding answer tag)
                seen_coordinates.add(coord)
            except ValueError:
                # Skip malformed coordinates
                continue
        
        # Check format requirements for early stopping
        # The response should have a <think>...</think> block followed by either 
        # <tool_call>...</tool_call> or <answer>...</answer> block
        
        # Find the last assistant turn content
        assistant_pattern = r'<\|im_start\|>assistant\s*\n(.*?)(?=<\|im_end\|>|<\|im_start\|>user|$)'
        assistant_turns = re.findall(assistant_pattern, response, re.DOTALL)
        
        if not assistant_turns:
            return False
        
        # Check the last (current) assistant turn
        current_turn = assistant_turns[-1].strip()
        if not current_turn:
            return False
        
        # Check if there's a <think> block
        think_match = re.search(r'<think>(.*?)</think>', current_turn, re.DOTALL)
        if not think_match:
            return True  # Stop if no <think> block found
        
        # Find the position after </think>
        think_end_pos = current_turn.find('</think>')
        if think_end_pos == -1:
            return True  # Stop if </think> is missing
        
        # Check what comes after </think>
        remaining_text = current_turn[think_end_pos + 7:].strip()
        
        # Check if there's either <tool_call> or <answer> after </think>
        has_tool_call = bool(re.search(r'<tool_call>(.*?)</tool_call>', remaining_text, re.DOTALL))
        has_answer = bool(re.search(r'<answer>(.*?)</answer>', remaining_text, re.DOTALL))
        
        # If neither <tool_call> nor <answer> is found, stop the trace
        if not has_tool_call and not has_answer:
            return True
        
        return False

    def _multi_turn_generate(self, vllm_inputs=None, sampling_params=None, use_tqdm=False):
        """Generate multi-turn conversations using batch processing."""
        
        sampling_params = copy.deepcopy(sampling_params)
        
        # Prepare initial samples
        new_vllm_inputs = []
        for single_vllm_input in vllm_inputs:
            prompt = self.tokenizer.decode(single_vllm_input['prompt_token_ids'], skip_special_tokens=False)
            new_vllm_inputs.extend([{
                "prompt": prompt,
                "multi_modal_data": copy.deepcopy(single_vllm_input['multi_modal_data']),
            } for _ in range(sampling_params.n)])
        
        sampling_params.n = 1
        sampling_params.detokenize = True
        
        # Initialize sample info
        samples_info = []
        batch_id = f"{self.rank}_{self.batch_counter}_{self.current_batch_uuid}"
        for index, item in enumerate(new_vllm_inputs):
            origin_image = item['multi_modal_data']['images'][0]
            # Store original image path
            original_image_path = origin_image if isinstance(origin_image, str) else None
            # Load image as PIL Image
            if isinstance(origin_image, str):
                processed_image = Image.open(origin_image).convert('RGB')
            else:
                processed_image = origin_image
            sample_info = {
                "prompt": item["prompt"],
                "sequence": item["prompt"],
                "multi_modal_data": {"image": [processed_image]},  # vLLM expects 'image' not 'images'
                "original_image_path": original_image_path,  # Store original image path
                "response": "",
                "stop": False,
                "finish_reason": None,
                "index": index,
                "batch_id": batch_id,
                "turn_count": 0,  # Initialize turn counter
                "crop_paths": [],  # Track cropped image paths for each turn
            }
            samples_info.append(sample_info)
        
        # Multi-turn generation loop
        num_llm_calls_available = copy.deepcopy(self.config.num_llm_calls_available) - 1
        turn_number = 0  # Track current turn number
        
        while num_llm_calls_available >= 0:
            turn_number += 1  # Increment turn counter
            num_llm_calls_available -= 1
            
            # Get active prompts
            input_prompts, multi_modal_data, indices = self._get_prompts_and_indices(samples_info)
            
            # Print number of active conversations
            print(f"###### Turn {turn_number}: {len(input_prompts)} active conversations ######")
            
            if not input_prompts:  # All samples finished
                break
            
            # Prepare vLLM inputs
            vllm_inputs = [{
                'prompt_token_ids': self.tokenizer.encode(prompt, add_special_tokens=False)[:self.config.prompt_length + self.config.response_length],
                'multi_modal_data': mm_data
            } for prompt, mm_data in zip(input_prompts, multi_modal_data)]
            
            # Generate responses
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs, 
                sampling_params=sampling_params, 
                use_tqdm=use_tqdm
            )
            
            sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
            responses = [x.outputs[0].text for x in sorted_outputs]
            finish_reason = [x.outputs[0].finish_reason for x in sorted_outputs]
            stop_reason = [x.outputs[0].stop_reason for x in sorted_outputs]
            
            # Check if this is the last call
            if num_llm_calls_available == -1:
                for i, index in enumerate(indices):
                    samples_info[index]['response'] += responses[i]
                    samples_info[index]['sequence'] += responses[i]
                    samples_info[index]['stop'] = True
                    samples_info[index]['finish_reason'] = finish_reason[i]
                break
            
            # Check for early stopping
            is_finished = [self._is_finished(responses[i]) for i in range(len(finish_reason))]
            
            if all(is_finished):  # All samples finished
                for i, index in enumerate(indices):
                    samples_info[index]['response'] += responses[i]
                    samples_info[index]['sequence'] += responses[i]
                    samples_info[index]['stop'] = True
                    samples_info[index]['finish_reason'] = finish_reason[i]
                break
            
            # Process responses and generate crop images
            with ThreadPoolExecutor(max_workers=max(min(len(indices), os.cpu_count(), 64), 1)) as executor:
                process_tasks = []
                for i, index in enumerate(indices):
                    if not is_finished[i]:
                        # Update turn counter for this sample
                        samples_info[index]['turn_count'] = turn_number
                        
                        task = executor.submit(
                            process_single_response, 
                            responses[i], 
                            samples_info[index]['multi_modal_data']['image'][0],  # vLLM expects 'image' not 'images'
                            self.crop_size,
                            self.rank,  # Pass rank
                            samples_info[index]['batch_id'],  # Pass batch_id
                            samples_info[index]['index'],  # Pass sample_index
                            turn_number  # Pass current turn number
                        )
                        process_tasks.append((i, index, task))
                
                # Update samples with processed results
                for i, index, task in process_tasks:
                    points, crop_paths = task.result()
                    
                    # Add response to sequence
                    samples_info[index]['response'] += responses[i]
                    samples_info[index]['sequence'] += responses[i]
                    
                    # Store crop paths for this turn
                    if crop_paths:
                        samples_info[index]['crop_paths'].append({
                            'turn': turn_number,
                            'paths': crop_paths,
                            'points': points
                        })
                    
                    if points and crop_paths:
                        # Add user feedback with crop image
                        user_feedback = f"\n<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
                        samples_info[index]['sequence'] += user_feedback
                        # Load crop images as PIL Images
                        processed_crop_images = [Image.open(crop_paths[0]).convert('RGB')]  # Load crop images as PIL Images
                        samples_info[index]['multi_modal_data']['image'].extend(processed_crop_images)  # vLLM expects 'image' not 'images'
                    else:
                        # Add user feedback without image
                        # If a point was extracted but crop failed, likely out-of-bounds
                        if points and len(points) >= 1:
                            px, py = points[-1]
                            img_w = samples_info[index]['multi_modal_data']['image'][0].width
                            img_h = samples_info[index]['multi_modal_data']['image'][0].height
                            msgs = []
                            if px < 0 or px >= img_w:
                                msgs.append("x is out of image boundary.")
                            if py < 0 or py >= img_h:
                                msgs.append("y is out of image boundary.")
                            if not msgs:
                                msgs.append("No valid pointing extracted from assistant's response.")
                            user_feedback_no_image = f"\n<|im_end|>\n<|im_start|>user\n{ ' '.join(msgs) }<|im_end|>\n<|im_start|>assistant\n"
                        else:
                            user_feedback_no_image = f"\n<|im_end|>\n<|im_start|>user\nNo valid pointing extracted from assistant's response.<|im_end|>\n<|im_start|>assistant\n"
                        samples_info[index]['sequence'] += user_feedback_no_image
        
            # Mark finished samples
            for i, index in enumerate(indices):
                if is_finished[i]:
                    samples_info[index]['response'] += responses[i]
                    samples_info[index]['sequence'] += responses[i]
                    samples_info[index]['stop'] = True
                    samples_info[index]['finish_reason'] = finish_reason[i]
        
        # Add EOS tokens
        for sample_info in samples_info:
            if sample_info['finish_reason'] != 'length':
                sample_info['sequence'] += self.tokenizer.eos_token
                sample_info['response'] += self.tokenizer.eos_token
        
        # Extract results
        responses = [sample_info['response'] for sample_info in samples_info]
        sequences = [sample_info['sequence'] for sample_info in samples_info]
        image_inputs = [sample_info['multi_modal_data']['image'] for sample_info in samples_info]  # vLLM expects 'image' not 'images'
        crop_paths_data = [sample_info['crop_paths'] for sample_info in samples_info]  # Extract crop paths data

        # Debug: Check if responses are empty
        empty_responses = sum(1 for r in responses if not r.strip())
        if empty_responses > 0:
            print(f"Warning: {empty_responses}/{len(responses)} responses are empty")
            # Print first few sequences to debug
            for i in range(min(3, len(sequences))):
                print(f"Sequence {i} length: {len(sequences[i])}")
                print(f"Response {i}: '{responses[i][:100]}...'")
        
        return responses, sequences, image_inputs, crop_paths_data

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate sequences for table reading multi-turn RL.
        
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
                non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({
                    "prompt_token_ids": list(raw_prompt_ids), 
                    "multi_modal_data": multi_modal_data
                })
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} 
                for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # Generate multi-turn responses
        with self.update_sampling_params(**prompts.meta_info):
            responses, sequences, image_inputs, crop_paths_data = self._multi_turn_generate(
                vllm_inputs=vllm_inputs, 
                sampling_params=self.sampling_params, 
                use_tqdm=False
            )

            # Handle sampling parameter n > 1
            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)

                    
        # Update raw prompt IDs with complete sequences
        non_tensor_batch["raw_prompt_ids"] = [
            self.tokenizer.encode(sequence, add_special_tokens=False)[:self.config.prompt_length + self.config.response_length] 
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
                return_tensors="pt"
            )
            
            # Get position IDs (use correct get_rope_index based on model type)
            try:
                # Detect model type from processor class name.
                # NOTE: Qwen3-VL reuses Qwen2VLImageProcessorFast, so inspect the TOP-LEVEL
                # processor class. See verl#4483.
                processor_class_name = self.processor.__class__.__name__ if self.processor else ""
                is_qwen3 = "Qwen3" in processor_class_name
                if is_qwen3:
                    from ...models.transformers.qwen3_vl import get_rope_index
                else:
                    from ...models.transformers.qwen2_vl import get_rope_index
                new_position_ids = get_rope_index(
                    self.processor,
                    input_ids=inputs['input_ids'][0],
                    image_grid_thw=inputs["image_grid_thw"],
                    attention_mask=inputs['attention_mask'][0],
                )
            except ImportError:
                # Fallback if get_rope_index is not available
                seq_len = inputs['input_ids'][0].size(0)
                repeat_dim = 4 if is_qwen3 else 3
                new_position_ids = (
                    torch.arange(seq_len, device=inputs['input_ids'][0].device).unsqueeze(0).repeat(repeat_dim, 1)
                )
            
            # Validate input consistency
            try:
                assert torch.sum(input_ids[idx][-prompt_len:].cpu() == inputs['input_ids'][0][:prompt_len].cpu()) == prompt_len, \
                    f"Input IDs mismatch at batch index {idx}"
                
                assert torch.sum(attention_mask[idx][-prompt_len:].cpu() == inputs['attention_mask'][0][:prompt_len].cpu()) == prompt_len, \
                    f"Attention mask mismatch at batch index {idx}"
            except:
                import pdb; pdb.set_trace()
                print(self.processor.tokenizer.decode(input_ids[idx][-prompt_len:].cpu(), skip_special_tokens=False))
                print(self.processor.tokenizer.decode(inputs['input_ids'][0][:prompt_len].cpu(), skip_special_tokens=False))

                
            # Extract response parts
            # response_ids.append(inputs['input_ids'][0][prompt_len:self.config.response_length])
            # response_mask.append(inputs['attention_mask'][0][prompt_len:self.config.response_length])

            # Extract response parts (slice RELATIVE to prompt_len)
            resp_end = prompt_len + self.config.response_length
            response_ids.append(inputs['input_ids'][0][prompt_len:resp_end])
            response_mask.append(inputs['attention_mask'][0][prompt_len:resp_end])
            
            # Pad position IDs for response
            pad_position_ids = VF.pad_sequence_to_length(
                new_position_ids[:, prompt_len:resp_end], 
                max_seq_len=self.config.response_length, 
                pad_token_id=0, 
                left_pad=False
            ).to(input_ids.device)
            response_position_ids.append(pad_position_ids)
            
            # Generate multi-turn mask
            tmp_multi_turn_mask = self._get_multi_turn_mask(inputs['input_ids'][0][prompt_len:resp_end])
            multi_turn_mask.append(tmp_multi_turn_mask)
            
            # Prepare model inputs
            inputs.pop('input_ids')
            inputs.pop('attention_mask')
            model_inputs.append(dict(inputs))

        # Pad response IDs
        response_ids = VF.pad_2d_list_to_length(
            response_ids, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)
        
        non_tensor_batch["multi_modal_inputs"] = model_inputs
        non_tensor_batch["crop_paths_data"] = crop_paths_data  # Add crop paths data to non_tensor_batch

        # Create final tensors
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_position_ids = torch.stack(response_position_ids, dim=0).to(input_ids.device)
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
        print(f"Valid Length - Max: {max_valid_length}, Min: {min_valid_length}, Avg: {avg_valid_length:.2f}")

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
        
        # Convert non-tensor batch to numpy arrays
        for key, value in non_tensor_batch.items():
            if not isinstance(value, np.ndarray):
                non_tensor_batch[key] = np.array(value, dtype=object)

        # import pdb; pdb.set_trace()

        print(repr(self.tokenizer.decode(batch['responses'][0][batch['response_mask'][1] == 1])))
                
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
 