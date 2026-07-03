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
Implement Actor
"""

import os
import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange as fa_rearrange, unpad_input  # type: ignore
except ImportError:
    index_first_axis = None  # type: ignore
    pad_input = None  # type: ignore
    fa_rearrange = None  # type: ignore
    unpad_input = None  # type: ignore

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__name__)


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

        # Training step counter for logging
        self._train_step = 0

    def _log_per_sample_vision_mismatch(self, input_ids: torch.Tensor, mm_lists: dict, context: str) -> None:
        """
        Log potential per-sample mismatches between visual placeholders in text and provided vision inputs.
        This ONLY logs (does not modify) and tries to be informative while cheap.

        It checks:
        - number of <|image_pad|> and <|video_pad|> tokens per sample
        - number of images/videos provided per sample via image_grid_thw/video_grid_thw
        - rough grid "cells" per sample (sum of t*h*w) to give a feel of visual token volume
        """
        try:
            B = input_ids.size(0)
            cfg = getattr(self.actor_module, "config", None)
            if cfg is None:
                return

            image_token_id = getattr(cfg, "image_token_id", None)
            video_token_id = getattr(cfg, "video_token_id", None)

            # Per-sample token counts
            img_tok_counts = (
                (input_ids == image_token_id).sum(dim=1).tolist() if image_token_id is not None else [0] * B
            )
            vid_tok_counts = (
                (input_ids == video_token_id).sum(dim=1).tolist() if video_token_id is not None else [0] * B
            )

            # Per-sample provided media counts and rough grid info
            img_grids = mm_lists.get("image_grid_thw", None)
            vid_grids = mm_lists.get("video_grid_thw", None)

            def grid_info_per_sample(grids_list, bidx):
                if grids_list is None or bidx >= len(grids_list) or grids_list[bidx] is None:
                    return 0, 0  # count, rough_cells
                g = grids_list[bidx]
                # g is expected [Ni, 3] (t, h, w) per image/video
                if not isinstance(g, torch.Tensor):
                    try:
                        g = torch.as_tensor(g)
                    except Exception:
                        return 0, 0
                if g.numel() == 0:
                    return 0, 0
                if g.dim() == 1 and g.numel() == 3:
                    g = g.view(1, 3)
                cnt = g.size(0)
                rough_cells = int((g[:, 0] * g[:, 1] * g[:, 2]).sum().item())
                return cnt, rough_cells

            any_warn = False
            for i in range(B):
                img_cnt, img_cells = grid_info_per_sample(img_grids, i)
                vid_cnt, vid_cells = grid_info_per_sample(vid_grids, i)

                # Potential mismatches to flag:
                cond_img = (img_tok_counts[i] == 0 and img_cnt > 0) or (img_tok_counts[i] > 0 and img_cnt == 0)
                cond_vid = (vid_tok_counts[i] == 0 and vid_cnt > 0) or (vid_tok_counts[i] > 0 and vid_cnt == 0)

                # Also flag if tokens exist but there are zero rough "cells" (very unusual)
                cond_img |= (img_tok_counts[i] > 0 and img_cells == 0 and img_cnt > 0)
                cond_vid |= (vid_tok_counts[i] > 0 and vid_cells == 0 and vid_cnt > 0)

                if cond_img or cond_vid:
                    any_warn = True
                    logger.warning(
                        "[Rank %s][%s] Sample %d potential vision mismatch: "
                        "image_pad_tokens=%d, images=%d, image_grid_cells=%d | "
                        "video_pad_tokens=%d, videos=%d, video_grid_cells=%d",
                        self.rank,
                        context,
                        i,
                        img_tok_counts[i],
                        img_cnt,
                        img_cells,
                        vid_tok_counts[i],
                        vid_cnt,
                        vid_cells,
                    )

            # Additionally, summarize when anything was flagged.
            if any_warn:
                logger.warning(
                    "[Rank %s][%s] Summary per-sample tokens: image_pad=%s | video_pad=%s",
                    self.rank,
                    context,
                    img_tok_counts,
                    vid_tok_counts,
                )
        except Exception as e:
            # Never break training due to logging
            logger.debug("Per-sample vision mismatch logging failed: %s", str(e))

    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        """
        Try to decode token ids to text using a tokenizer if available.
        Falls back to printing token ids if tokenizer is not attached.
        """
        tok = getattr(self.actor_module, "tokenizer", None)
        if tok is None:
            tok = getattr(self, "tokenizer", None)

        ids = token_ids.detach().cpu().tolist()
        if tok is None:
            return f"[no tokenizer] ids={ids}"

        # Try common decode signatures
        try:
            return tok.decode(ids, skip_special_tokens=True)
        except Exception:
            try:
                out = tok.batch_decode([ids], skip_special_tokens=True)
                return out[0] if isinstance(out, list) and len(out) > 0 else str(ids)
            except Exception:
                return str(ids)

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            # Get per-sample lists (do not concat yet)
            multi_modal_inputs_lists = batch_collate(micro_batch["multi_modal_inputs"])

            # Per-sample diagnostics (before we concatenate across batch)
            if self.rank == 0:
                self._log_per_sample_vision_mismatch(
                    input_ids=input_ids, mm_lists=multi_modal_inputs_lists, context="actor._forward_micro_batch"
                )

            # Now concatenate for model call
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs_lists.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            if unpad_input is None or index_first_axis is None or pad_input is None:
                raise RuntimeError("padding_free path requires flash_attn.bert_padding utilities installed.")
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    def _maybe_extract_first_response_text(self, model_inputs: dict) -> Optional[str]:
        """
        Helper to extract and decode the first response (sample 0) from a model_inputs dict.
        Uses response_mask if present to strip padding.
        """
        try:
            responses = model_inputs.get("responses", None)
            if responses is None or responses.size(0) == 0:
                return None
            resp_tokens = responses[0]  # (response_len,)
            # Prefer response_mask to trim padding
            resp_mask = model_inputs.get("response_mask", None)
            if resp_mask is not None and resp_mask.size(0) > 0:
                mask0 = resp_mask[0].to(dtype=torch.bool)
                # Align shapes if needed
                if mask0.numel() == resp_tokens.numel():
                    resp_tokens = resp_tokens[mask0]
            return self._decode_tokens(resp_tokens)
        except Exception as e:
            logger.debug("Failed to decode first response for logging: %s", str(e))
            return None

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        # Add multi_turn_mask if available for multi-turn training
        if "multi_turn_mask" in data.batch:
            select_keys.append("multi_turn_mask")
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs", "overlong_traces", "void_traces"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            # Capture first response text for logging (rank 0 only)
            first_response_text: Optional[str] = None

            for mini_batch in mini_batches:
                # Use multi_turn_mask if available for token counting
                if "multi_turn_mask" in mini_batch.batch:
                    total_response_tokens = torch.sum(mini_batch.batch["multi_turn_mask"])
                else:
                    total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    # Store the first response text (only once per step)
                    if self.rank == 0 and first_response_text is None:
                        first_response_text = self._maybe_extract_first_response_text(model_inputs)

                    # Use multi_turn_mask if available, otherwise fall back to response_mask
                    if "multi_turn_mask" in model_inputs:
                        policy_mask = model_inputs["multi_turn_mask"]
                    else:
                        policy_mask = model_inputs["response_mask"]
                    
                    # Apply overlong filtering if enabled
                    if self.config.overlong_filtering and "overlong_traces" in model_inputs:
                        overlong_traces = model_inputs["overlong_traces"]
                        # Assume ndarray input; coerce object/bool to int array
                        try:
                            arr = np.asarray(overlong_traces, dtype=bool).astype(np.int64)
                        except Exception:
                            raise "Warn: failed to coerce overlong_traces ndarray"

                        overlong_tensor_1d = torch.from_numpy(arr).to(device=policy_mask.device, dtype=policy_mask.dtype)

                        # Validate batch-size alignment
                        if overlong_tensor_1d.numel() != policy_mask.size(0):
                            print(
                                f"Warn: overlong_traces batch mismatch: {overlong_tensor_1d.numel()} vs {policy_mask.size(0)}; skip filtering"
                            )
                        else:
                            # Expand to [batch, response_len]
                            overlong_tensor = overlong_tensor_1d.unsqueeze(-1).expand_as(policy_mask)
                            # Mask out overlong traces (set to 0 where overlong_traces is True)
                            policy_mask = policy_mask * (1 - overlong_tensor)

                            # Log number of overlong traces (not tokens)
                            num_overlong_traces = int(overlong_tensor_1d.sum().item())
                            if num_overlong_traces > 0:
                                print(
                                    f"Overlong filtering: masked {num_overlong_traces} overlong traces out of {overlong_tensor_1d.numel()} total traces (mini-batch)"
                                )

                    # Apply void trace filtering (format reward == 0)
                    if self.config.void_trace_filtering and "void_traces" in model_inputs:
                        void_traces = model_inputs["void_traces"]
                        try:
                            arr_void = np.asarray(void_traces, dtype=bool).astype(np.int64)
                        except Exception:
                            raise "Warn: failed to coerce void_traces ndarray"

                        void_tensor_1d = torch.from_numpy(arr_void).to(device=policy_mask.device, dtype=policy_mask.dtype)

                        if void_tensor_1d.numel() != policy_mask.size(0):
                            print(
                                f"Warn: void_traces batch mismatch: {void_tensor_1d.numel()} vs {policy_mask.size(0)}; skip filtering"
                            )
                        else:
                            void_tensor = void_tensor_1d.unsqueeze(-1).expand_as(policy_mask)
                            policy_mask = policy_mask * (1 - void_tensor)

                            num_void_traces = int(void_tensor_1d.sum().item())
                            if num_void_traces > 0:
                                print(
                                    f"Void trace filtering: masked {num_void_traces} traces (format==0) out of {void_tensor_1d.numel()} total traces (mini-batch)"
                                )
                    
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)

                    # === DIAGNOSTIC: check log_probs vs old_log_probs/ref_log_probs at first micro-batch of step 1 ===
                    if self._train_step <= 1 and not hasattr(self, '_diag_logged_step1'):
                        self._diag_logged_step1 = True
                        _d_old = old_log_probs
                        _d_new = log_probs
                        _d_ref = model_inputs.get("ref_log_probs", None)
                        # Use response_mask for comparison (same as ray_trainer diagnostic)
                        _d_resp_mask = model_inputs.get("response_mask", policy_mask)
                        _d_mt_mask = policy_mask  # multi_turn_mask
                        print(f"\n{'='*60}")
                        print(f"[DIAG actor step={self._train_step}, first micro-batch]")
                        # Compare with response_mask (same as ray_trainer diag)
                        _rn = _d_resp_mask.sum().item()
                        _mn = _d_mt_mask.sum().item()
                        print(f"  response_mask tokens: {_rn}, multi_turn_mask tokens: {_mn}")
                        # Raw tensor comparison (no masking - element-wise)
                        _raw_diff = (_d_new - _d_old)
                        _nonzero_mask = (_d_old != 0)  # only compare non-padding positions
                        _nz = _nonzero_mask.sum().item()
                        print(f"  RAW (no mask) old!=0 positions: {_nz}")
                        if _nz > 0:
                            _raw_nz_diff = _raw_diff[_nonzero_mask]
                            print(f"  RAW diff mean: {_raw_nz_diff.mean().item():.6f}")
                            print(f"  RAW diff abs_mean: {_raw_nz_diff.abs().mean().item():.6f}")
                            print(f"  RAW diff max_abs: {_raw_nz_diff.abs().max().item():.6f}")
                        # With response_mask
                        _resp_diff = _raw_diff * _d_resp_mask
                        print(f"  RESP_MASK new mean: {(_d_new * _d_resp_mask).sum().item() / max(_rn, 1):.6f}")
                        print(f"  RESP_MASK old mean: {(_d_old * _d_resp_mask).sum().item() / max(_rn, 1):.6f}")
                        print(f"  RESP_MASK diff mean: {_resp_diff.sum().item() / max(_rn, 1):.6f}")
                        print(f"  RESP_MASK diff abs_mean: {_resp_diff.abs().sum().item() / max(_rn, 1):.6f}")
                        # With multi_turn_mask
                        _mt_diff = _raw_diff * _d_mt_mask
                        print(f"  MT_MASK new mean: {(_d_new * _d_mt_mask).sum().item() / max(_mn, 1):.6f}")
                        print(f"  MT_MASK old mean: {(_d_old * _d_mt_mask).sum().item() / max(_mn, 1):.6f}")
                        print(f"  MT_MASK diff mean: {_mt_diff.sum().item() / max(_mn, 1):.6f}")
                        print(f"  MT_MASK diff abs_mean: {_mt_diff.abs().sum().item() / max(_mn, 1):.6f}")
                        # Per-sample first 2 samples
                        for _si in range(min(2, _d_new.shape[0])):
                            _s_rm = _d_resp_mask[_si]
                            _s_rn = _s_rm.sum().item()
                            _s_new = (_d_new[_si] * _s_rm).sum().item() / max(_s_rn, 1)
                            _s_old = (_d_old[_si] * _s_rm).sum().item() / max(_s_rn, 1)
                            _s_diff_abs = ((_d_new[_si] - _d_old[_si]) * _s_rm).abs().sum().item() / max(_s_rn, 1)
                            print(f"  sample[{_si}] resp_mask: new={_s_new:.4f}, old={_s_old:.4f}, abs_diff={_s_diff_abs:.6f}, ntok={_s_rn}")
                        if _d_ref is not None:
                            _kr = ((_d_new - _d_ref) * _d_resp_mask)
                            print(f"  RESP_MASK kl(new-ref) mean: {_kr.sum().item() / max(_rn, 1):.6f}")
                            print(f"  RESP_MASK kl(new-ref) abs_mean: {_kr.abs().sum().item() / max(_rn, 1):.6f}")
                        print(f"  shapes: new={_d_new.shape}, old={_d_old.shape}, resp_mask={_d_resp_mask.shape}, mt_mask={_d_mt_mask.shape}")
                        print(f"  dtypes: new={_d_new.dtype}, old={_d_old.dtype}")
                        print(f"{'='*60}\n")
                    # === END DIAGNOSTIC ===

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=policy_mask,  # Use policy_mask instead of response_mask
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, policy_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    loss = loss * torch.sum(policy_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_metrics["pg_clipfrac_higher"],
                        "actor/pg_clipfrac_lower": pg_metrics["pg_clipfrac_lower"],
                        "actor/entropy_loss": pg_metrics["entropy_loss"],
                        "actor/ppo_kl": pg_metrics["ppo_kl"],
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": float(grad_norm.detach().item())})

                # Log the first response after the optimizer step (rank 0 only)
                if self.rank == 0 and first_response_text is not None:
                    logger.info("[TrainStep %d] First response: %s", self._train_step, first_response_text)

                # Increment training step counter after each optimizer step
                self._train_step += 1

        return metrics