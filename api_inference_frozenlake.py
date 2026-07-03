#!/usr/bin/env python
"""
api_inference_frozenlake.py — multi-turn evaluation.

The model proposes a `route` (comma-separated up/down/left/right). The script
simulates that route on the real map, renders the resulting trajectory, and
feeds the rendered image back image-only (no text), matching training. The
loop ends on `terminate` or --max_turns.
"""

import argparse
import base64
import io
import json
import logging
import os
import random
import re
from collections import Counter, deque
from datetime import datetime

import pandas as pd
from datasets import Dataset
from frozenlake_scoring import (
    _parse_action_csv,
    _render_frozenlake_base_rgb,
    evaluate_frozenlake_trajectory,
)
from inferencer_class import NextTurnGenerator
from PIL import Image

# Reflection / VPRL-Direct system prompt, sourced from prompts.py so train and
# eval stay byte-equal. This is the only prompt the released recipe selects
# (prompt_variant "reflection"/"vprl_direct").
from prompts import (
    SYSTEM_PROMPT_TAG_VPRL_DIRECT as FROZENLAKE_SYSTEM_PROMPT_VPRL_DIRECT,
)

# VPRL Direct user-message text -- minimal, since the system prompt already
# contains the full task description. Just nudges the model to act.
FIRST_USER_TEXT_TAG_VPRL_DIRECT = "Generate the shortest action sequence."

# Reflection feedback prompt (appended after the feedback <image> on every
# feedback turn during eval).
FEEDBACK_USER_PROMPT_REFLECTION = (
    "The image shows your proposed action sequence executed on the grid.\n"
    "Verify the path against the task requirements. If the path reaches the\n"
    "gift without stepping on any ice holes, confirm with\n"
    "<FINAL>actions</FINAL>. Otherwise, provide your reasoning in\n"
    "<THINK>...</THINK> and a revised sequence in <ANSWER>actions</ANSWER>."
)


def _system_prompt_for_fmt(fmt, variant="reflection"):
    """System prompt for the released tag/reflection formats.

    The released recipe (scripts/frozenlake_model_family.py) only ever selects
    prompt_variant "reflection" or "vprl_direct", both of which use the
    VPRL-Direct system prompt.
    """
    return FROZENLAKE_SYSTEM_PROMPT_VPRL_DIRECT


def _first_user_text_for_fmt(fmt, variant="reflection"):
    return FIRST_USER_TEXT_TAG_VPRL_DIRECT


def _feedback_user_text_for_fmt(fmt, variant="reflection"):
    """Text appended after each feedback <image> turn. For reflection_tag,
    returns the reflection prompt that primes <FINAL> or <THINK>/<ANSWER>."""
    if fmt == "reflection_tag":
        return FEEDBACK_USER_PROMPT_REFLECTION
    return ""


_DIR_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


# ----------------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------------


def _extract_json_blob(text):
    """Return the first JSON object found in `text` parsed to a dict, or None.
    Tolerates ```json fences and bare JSON; falls back to json_repair."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = m.group(0) if m else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except Exception:
        try:
            import json_repair

            return json_repair.loads(candidate)
        except Exception:
            return None


# Tag-format regexes. Body is captured non-greedy; DOTALL so a <think> can span
# newlines. The body of <route>/<answer> is re-split on commas+whitespace, so
# both legacy comma-separated and new space-separated action lists work.
_TAG_ROUTE_RE = re.compile(r"<route\b[^>]*>(.*?)</route>", re.DOTALL | re.IGNORECASE)
_TAG_ANSWER_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_TAG_FINAL_RE = re.compile(r"<final\b[^>]*>(.*?)</final>", re.DOTALL | re.IGNORECASE)


def _parse_action_csv_or_space(body):
    """Tokenize `body` accepting commas, whitespace, or both as separators.

    Returns (tokens_list_or_None, error_str_or_None). Wraps `_parse_action_csv`
    by first normalizing whitespace runs to single commas. Lowercases tokens.
    """
    if not isinstance(body, str):
        return None, "tag body is not a string"
    s = body.strip()
    if not s:
        return None, "tag body is empty"
    # Replace any whitespace run with a comma; collapse multiple commas. This
    # keeps `_parse_action_csv`'s single-source-of-truth validation while
    # accepting space-separated input.
    normalized = re.sub(r"[\s,]+", ",", s).strip(",")
    tokens, _, err = _parse_action_csv(normalized, {})
    return tokens, err


def parse_route_terminate(text, fmt="tag"):
    """Parse a route/terminate response.

    Returns (kind, actions, error) where kind is 'route' | 'terminate' | None,
    actions is a validated list of direction tokens or None, error is a short
    diagnostic string or None.

    Args:
        text: the raw model output.
        fmt: which output format to expect:
            "tag"  -- parse ``<route>``/``<answer>`` tags only (default).
                      ``<answer>`` maps to ``"terminate"`` so loop control is
                      unchanged.
            "json" -- parse the JSON ``function_call`` schema only.

    Action lists in both formats tolerate comma- or space-separated tokens.
    """
    if fmt == "tag":
        return _parse_tag_format(text)
    if fmt == "reflection_tag":
        return _parse_reflection_tag_format(text)
    if fmt == "json":
        return _parse_json_format(text)
    return (
        None,
        None,
        (f"unknown fmt={fmt!r}; expected 'tag', 'reflection_tag', or 'json'"),
    )


def _parse_reflection_tag_format(text):
    """Parse the reflection-corpus tag format: <FINAL> = commit (terminate),
    <ANSWER> = proposal that needs feedback (treated as 'route' so the loop
    renders a feedback image and continues). <THINK> blocks are ignored —
    only the action sequence inside <ANSWER> or <FINAL> matters."""
    if not isinstance(text, str):
        return None, None, "no <FINAL>/<ANSWER> tag in response"
    m_final = _TAG_FINAL_RE.search(text)
    if m_final is not None:
        tokens, err = _parse_action_csv_or_space(m_final.group(1))
        return "terminate", tokens, err
    m_ans = _TAG_ANSWER_RE.search(text)
    if m_ans is not None:
        tokens, err = _parse_action_csv_or_space(m_ans.group(1))
        # Proposal — caller will render feedback and continue.
        return "route", tokens, err
    return None, None, "no <FINAL>/<ANSWER> tag in response"


def _parse_tag_format(text):
    """Parse ``<route>``/``<answer>`` tag format. Returns (kind, actions, err)."""
    if not isinstance(text, str):
        return None, None, "no <route>/<answer> tag in response"
    # Prefer <answer> over <route> if both happen to appear -- the model has
    # committed.
    m_ans = _TAG_ANSWER_RE.search(text)
    if m_ans is not None:
        tokens, err = _parse_action_csv_or_space(m_ans.group(1))
        return "terminate", tokens, err
    m_route = _TAG_ROUTE_RE.search(text)
    if m_route is not None:
        tokens, err = _parse_action_csv_or_space(m_route.group(1))
        return "route", tokens, err
    return None, None, "no <route>/<answer> tag in response"


def _parse_json_format(text):
    """Parse the legacy JSON ``function_call`` schema. (kind, actions, err)."""
    data = _extract_json_blob(text)
    if not isinstance(data, dict):
        return None, None, "no JSON object in response"
    fc = data.get("function_call")
    if not isinstance(fc, dict):
        return None, None, "function_call missing or not an object"
    name = fc.get("name")
    args = fc.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}
    if name == "terminate":
        ans = args.get("answer", "")
        tokens, _, err = _parse_action_csv(
            ans if isinstance(ans, str) else str(ans), {}
        )
        return "terminate", tokens, err
    if name == "route":
        act = args.get("actions", "")
        tokens, _, err = _parse_action_csv(
            act if isinstance(act, str) else str(act), {}
        )
        return "route", tokens, err
    return None, None, f"unknown function name: {name!r}"


# ----------------------------------------------------------------------------
# Map helpers
# ----------------------------------------------------------------------------


def bfs_shortest_path_len(layout, start_pos, target_pos, level):
    """BFS over walkable cells (any non-'H'). Returns number of moves on the
    shortest path, or None if unreachable."""
    L = int(level)
    start = int(start_pos)
    target = int(target_pos)
    if start == target:
        return 0
    visited = {start}
    q = deque([(start, 0)])
    while q:
        pos, d = q.popleft()
        r, c = divmod(pos, L)
        for dr, dc in _DIR_DELTA.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < L and 0 <= nc < L):
                continue
            npos = nr * L + nc
            if npos in visited:
                continue
            if layout[nr][nc] == "H":
                continue
            if npos == target:
                return d + 1
            visited.add(npos)
            q.append((npos, d + 1))
    return None


def resize_image_if_needed(src_path, dst_path, max_pixels):
    """If src_path exceeds max_pixels, downscale (preserve aspect) and save to
    dst_path. Returns the path to use. max_pixels<=0 disables resizing."""
    if max_pixels <= 0:
        return src_path
    try:
        with Image.open(src_path) as img:
            w, h = img.size
            if w * h <= max_pixels:
                return src_path
            scale = (max_pixels / (w * h)) ** 0.5
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            img.resize((nw, nh), Image.LANCZOS).save(dst_path)
        return dst_path
    except Exception as e:
        logging.warning(f"resize_image_if_needed: {src_path}: {e}; using original")
        return src_path


def save_data_uri_png(data_uri, dst_path, max_pixels=0):
    """Decode a base64 PNG data URI, optionally downscale to <= max_pixels,
    save to dst_path. Returns dst_path on success, None on failure."""
    if not data_uri or not data_uri.startswith("data:image/png;base64,"):
        return None
    try:
        b = base64.b64decode(data_uri.split(",", 1)[1])
        img = Image.open(io.BytesIO(b)).convert("RGB")
        if max_pixels > 0 and img.size[0] * img.size[1] > max_pixels:
            scale = (max_pixels / (img.size[0] * img.size[1])) ** 0.5
            nw, nh = max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))
            img = img.resize((nw, nh), Image.LANCZOS)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img.save(dst_path, "PNG")
        return dst_path
    except Exception as e:
        logging.warning(f"save_data_uri_png failed for {dst_path}: {e}")
        return None


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------


def score_actions(actions, layout, start_pos, level, optimal_len):
    """Score a final action sequence. Returns dict with strict_correct,
    lenient_correct, optimal, length, sim."""
    if actions is None:
        return {
            "strict_correct": False,
            "lenient_correct": False,
            "optimal": False,
            "length": None,
            "sim": None,
        }
    ev = evaluate_frozenlake_trajectory(
        actions, layout, int(start_pos), int(level), None
    )
    sim = ev["simulation"] or {}
    strict = bool(ev["is_correct"])
    lenient = bool(sim.get("reached_goal", False))
    optimal = bool(strict and optimal_len is not None and len(actions) == optimal_len)
    return {
        "strict_correct": strict,
        "lenient_correct": lenient,
        "optimal": optimal,
        "length": len(actions),
        "sim": sim,
    }


# ----------------------------------------------------------------------------
# Dataset loader
# ----------------------------------------------------------------------------


def load_frozenlake_eval_examples(
    eval_dataset_path, levels, sample_size, eval_first_k, seed=42
):
    """Load FrozenLake eval examples from eval_data.json.

    Filtering order: level filter -> (optional random sample_size total) ->
    (optional eval_first_k per-level).
    """
    raw = json.load(open(eval_dataset_path))
    base_dir = os.path.dirname(os.path.abspath(eval_dataset_path))
    levels_set = set(int(x) for x in levels)
    examples = []
    for ex in raw:
        md = ex["metadata"]
        if int(md["level"]) not in levels_set:
            continue
        layout = md["layout"]
        start_pos = int(md["start_pos"])
        target_pos = int(md["target_pos"])
        level = int(md["level"])
        opt = bfs_shortest_path_len(layout, start_pos, target_pos, level)
        examples.append(
            {
                "id": ex["id"],
                "level": level,
                "layout": layout,
                "start_pos": start_pos,
                "target_pos": target_pos,
                "optimal_path_len": opt,
                "image_path": os.path.join(base_dir, ex["input_map"]),
            }
        )
    if sample_size is not None and sample_size > 0:
        rng = random.Random(seed)
        examples = rng.sample(examples, k=min(sample_size, len(examples)))
    if eval_first_k > 0:
        by_level = {}
        for ex in examples:
            by_level.setdefault(ex["level"], []).append(ex)
        examples = []
        for lvl in sorted(by_level):
            examples.extend(by_level[lvl][:eval_first_k])
    return examples


# ----------------------------------------------------------------------------
# Trajectory rendering (wall-aware)
# ----------------------------------------------------------------------------
#
# When the model emits an action that would step off the grid, gym's FrozenLake
# treats it as a no-op (the agent stays put). This renderer extends the polyline
# half a cell beyond the grid boundary in the attempted direction for each
# wall-hit step so it is visually evident. The agent's logical (r, c) stays
# unchanged (matching gym semantics), so subsequent legal actions are unaffected.


def _compute_trajectory_vertices_with_walls(
    actions, layout, start_pos, level, cell_px=64
):
    """Compute pixel-space (x, y) vertices for the trajectory polyline,
    placing wall-hit attempts at the phantom-cell center just outside the grid
    so they are visually evident. Length = len(actions) + 1.
    """
    L = int(level)
    cell_half = cell_px / 2

    def center(r_, c_):
        return (c_ * cell_px + cell_half, r_ * cell_px + cell_half)

    r, c = divmod(int(start_pos), L)
    vertices = [center(r, c)]
    for act in actions or []:
        if act not in _DIR_DELTA:
            break
        dr, dc = _DIR_DELTA[act]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < L and 0 <= nc < L):
            vertices.append(center(nr, nc))  # off-grid phantom-cell center
            continue
        r, c = nr, nc
        vertices.append(center(r, c))
    return vertices


def render_frozenlake_trajectory_with_walls_data_uri(
    layout,
    start_pos,
    actions,
    level,
    line_color="#e10600",
    end_color="#dc2626",
    cell_px=64,
    render_dpi=160,
):
    """Render the trajectory, handling wall-hits by extending the polyline
    off-grid. Returns a base64 PNG data URI, or None on failure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
    except Exception:
        return None
    try:
        img = _render_frozenlake_base_rgb(layout, start_pos, level)
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(level * 0.9 + 0.2, level * 0.9 + 0.2))
    try:
        ax.imshow(img)
        vertices = _compute_trajectory_vertices_with_walls(
            actions, layout, start_pos, level, cell_px=cell_px
        )
        if vertices:
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            if len(xs) >= 2:
                ax.plot(
                    xs,
                    ys,
                    color=line_color,
                    linewidth=5,
                    alpha=0.5,
                    zorder=3,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    path_effects=[
                        pe.Stroke(linewidth=9, foreground="black"),
                        pe.Normal(),
                    ],
                )
            ax.scatter(
                [xs[-1]],
                [ys[-1]],
                s=400,
                marker="*",
                c=end_color,
                edgecolors="black",
                linewidths=1.5,
                alpha=1.0,
                zorder=6,
            )
            # Extend axes only if any vertex actually went off-grid, so
            # wall-free trajectories are unaffected.
            img_h, img_w = img.shape[:2]
            pad = cell_px * 0.25
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            if min_x < 0 or max_x > img_w or min_y < 0 or max_y > img_h:
                ax.set_xlim(min(min_x - pad, 0), max(max_x + pad, img_w))
                ax.set_ylim(max(max_y + pad, img_h), min(min_y - pad, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        buf = io.BytesIO()
        fig.savefig(
            buf, format="PNG", dpi=render_dpi, bbox_inches="tight", pad_inches=0.05
        )
    finally:
        plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


# ----------------------------------------------------------------------------
# Multi-turn rollout
# ----------------------------------------------------------------------------


class FrozenLakeMultiTurnEval:
    def __init__(
        self,
        inferencer,
        max_turns,
        feedback_dir,
        max_image_pixels,
        fmt="tag",
        prompt_variant="think",
        feedback_max_pixels=-1,
        feedback_render_dpi=160,
    ):
        self.inferencer = inferencer
        self.max_turns = int(max_turns)
        self.feedback_dir = feedback_dir
        self.max_image_pixels = int(max_image_pixels)
        # Feedback-image-only pixel cap; >=0 overrides max_image_pixels for the
        # rendered feedback PNG only (the initial puzzle image still uses
        # max_image_pixels). -1 = inherit max_image_pixels.
        self.feedback_max_pixels = int(feedback_max_pixels)
        # Matplotlib savefig dpi for the feedback renderer. 160 = default
        # (~118 px/cell). Lower => natively-rendered smaller feedback image
        # (crisp lines drawn at the target size, no PIL resampling).
        self.feedback_render_dpi = int(feedback_render_dpi)
        # "tag" (default, active SFT format with <route>/<answer>) or "json"
        # (legacy function_call schema). Passed through to parse_route_terminate.
        self.fmt = fmt
        # Only relevant for fmt=tag: "think" (with-think system prompt) or
        # "nothink" (Direct-style no-think system prompt). Must match the
        # SFT system prompt the model was trained on.
        self.prompt_variant = prompt_variant

    def run(self, examples):
        convs = []
        for i, ex in enumerate(examples):
            img_path = ex["image_path"]
            if self.max_image_pixels > 0:
                resized_path = os.path.join(
                    self.feedback_dir,
                    "..",
                    "_resized_inputs",
                    f"{ex['id']}_initial.png",
                )
                resized_path = os.path.normpath(resized_path)
                img_path = resize_image_if_needed(
                    img_path, resized_path, self.max_image_pixels
                )
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": _system_prompt_for_fmt(
                                self.fmt, self.prompt_variant
                            ),
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": _first_user_text_for_fmt(
                                self.fmt, self.prompt_variant
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": os.path.abspath(img_path)},
                        },
                    ],
                },
            ]
            convs.append(
                {
                    "idx": i,
                    "ex": ex,
                    "messages": messages,
                    "turn_responses": [],
                    "finished": False,
                    "final_kind": None,
                    "final_actions": None,
                    "first_actions": None,
                    "format_error_msg": None,
                    "n_route_calls": 0,
                }
            )

        for turn_idx in range(self.max_turns):
            active = [c for c in convs if not c["finished"]]
            if not active:
                break
            logging.info(f"### Turn {turn_idx + 1}: {len(active)} active ###")
            ds = Dataset.from_dict(
                {
                    "messages": [c["messages"] for c in active],
                    "__conv_idx": [c["idx"] for c in active],
                }
            )
            result = self.inferencer.next_turn_generator(ds).dataset
            # Match responses back to conversations by a STABLE key, not by position.
            # When require_all_responses=False, curator may drop or null-out failed
            # requests (finish_reason=length, transient disconnects); a positional
            # zip would then misalign and corrupt OTHER rollouts. The map handles
            # dropped rows (key absent) and null responses (value None) identically.
            resp_by_idx = {}
            for row in result:
                try:
                    resp_by_idx[int(row["__conv_idx"])] = row.get("response")
                except Exception:
                    pass
            for conv in active:
                text = resp_by_idx.get(conv["idx"])
                if not text:
                    # Generation failed for THIS rollout only. Judge it EM=0:
                    # final_actions stays None -> score_actions() returns all-False,
                    # and the record still counts toward the n=250 denominator, so
                    # the remaining rollouts are completely unaffected.
                    conv["finished"] = True
                    conv["final_kind"] = "generation_error"
                    conv["final_actions"] = (
                        None  # force EM=0 even if a prior route was cached
                    )
                    conv["format_error_msg"] = (
                        "generation failed (finish_reason=length or request error)"
                    )
                    continue
                conv["messages"].append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": text}],
                    }
                )
                conv["turn_responses"].append(text)
                kind, actions, err = parse_route_terminate(text, fmt=self.fmt)
                if turn_idx == 0:
                    conv["first_actions"] = actions
                if kind == "terminate":
                    conv["finished"] = True
                    conv["final_kind"] = "terminate"
                    conv["final_actions"] = actions
                    if err:
                        conv["format_error_msg"] = err
                elif kind == "route" and actions is not None:
                    conv["n_route_calls"] += 1
                    conv["final_actions"] = actions  # fallback if max_turns hit
                    uri = render_frozenlake_trajectory_with_walls_data_uri(
                        conv["ex"]["layout"],
                        conv["ex"]["start_pos"],
                        actions,
                        conv["ex"]["level"],
                        render_dpi=self.feedback_render_dpi,
                    )
                    fb_path = None
                    if uri:
                        dst = os.path.join(
                            self.feedback_dir,
                            f"{conv['ex']['id']}_turn{turn_idx + 1}.png",
                        )
                        fb_cap = (
                            self.feedback_max_pixels
                            if self.feedback_max_pixels >= 0
                            else self.max_image_pixels
                        )
                        fb_path = save_data_uri_png(uri, dst, fb_cap)
                    if fb_path:
                        fb_user_content = [
                            {
                                "type": "image_url",
                                "image_url": {"url": os.path.abspath(fb_path)},
                            },
                        ]
                        fb_text = _feedback_user_text_for_fmt(
                            self.fmt, self.prompt_variant
                        )
                        if fb_text:
                            fb_user_content.append({"type": "text", "text": fb_text})
                        conv["messages"].append(
                            {
                                "role": "user",
                                "content": fb_user_content,
                            }
                        )
                    else:
                        conv["finished"] = True
                        conv["final_kind"] = "format_error"
                        conv["format_error_msg"] = "feedback render failed"
                else:
                    conv["finished"] = True
                    conv["final_kind"] = "format_error"
                    conv["format_error_msg"] = err or "unparseable response"

        for conv in convs:
            if not conv["finished"]:
                conv["finished"] = True
                conv["final_kind"] = "max_turns"
        return convs


# ----------------------------------------------------------------------------
# API client
# ----------------------------------------------------------------------------


class APICallToolUse:
    def __init__(
        self,
        model_name,
        model_temperature,
        vllm_model_port,
        max_tokens=512,
        require_all_responses=True,
    ):
        self.model_name = model_name
        self.model_temperature = model_temperature
        self.vllm_model_port = vllm_model_port
        self.max_tokens = max_tokens
        # When False, requests that fail after retries (e.g. finish_reason=length,
        # transient disconnects) are tolerated instead of aborting the whole level;
        # the affected rollout is scored EM=0 downstream (see run()).
        self.require_all_responses = require_all_responses
        self._init_generator()

    def _init_generator(self):
        if self.vllm_model_port:
            logging.info(
                f"vLLM backend: model={self.model_name} port={self.vllm_model_port}"
            )
            cfg = {
                "model_name": self.model_name,
                "generation_params": {
                    "max_tokens": self.max_tokens,
                    "include_stop_str_in_output": True,
                    "skip_special_tokens": False,
                    "temperature": self.model_temperature,
                },
                "backend": "openai",
                "backend_params": {
                    "base_url": f"http://localhost:{self.vllm_model_port}/v1",
                    "api_key": "dummy-key",
                    "max_requests_per_minute": 50000,
                    "max_tokens_per_minute": 4000000,
                    "max_concurrent_requests": 50000,
                    "require_all_responses": self.require_all_responses,
                    "max_retries": 5,
                },
            }
        else:
            cfg = {
                "model_name": self.model_name,
                "generation_params": {
                    "max_tokens": 30000,
                    "temperature": self.model_temperature,
                },
                "backend": "litellm",
                "backend_params": {
                    "max_requests_per_minute": 300,
                    "max_tokens_per_minute": 400000,
                    "seconds_to_pause_on_rate_limit": 15.0,
                    "require_all_responses": self.require_all_responses,
                    "max_retries": 5,
                },
            }
        self.next_turn_generator = NextTurnGenerator(**cfg)


# ----------------------------------------------------------------------------
# Metrics aggregation
# ----------------------------------------------------------------------------


def _safe_pct(num, denom):
    return float(num) / float(denom) if denom else 0.0


def aggregate_metrics(records):
    def group_metrics(group):
        n = len(group)
        if n == 0:
            return None
        strict = sum(1 for r in group if r["strict_correct"])
        lenient = sum(1 for r in group if r["lenient_correct"])
        optimal = sum(1 for r in group if r["optimal"])
        turn1 = sum(1 for r in group if r["turn1_strict_correct"])
        fmt_err = sum(1 for r in group if r["final_kind"] == "format_error")
        max_t = sum(1 for r in group if r["final_kind"] == "max_turns")
        clean = sum(1 for r in group if r["final_kind"] == "terminate")
        avg_turns = sum(r["n_turns"] for r in group) / n
        avg_route = sum(r["n_route_calls"] for r in group) / n
        return {
            "n": n,
            "strict_success_rate": _safe_pct(strict, n),
            "lenient_success_rate": _safe_pct(lenient, n),
            "optimal_rate": _safe_pct(optimal, n),
            "turn1_strict_rate": _safe_pct(turn1, n),
            "final_strict_rate": _safe_pct(strict, n),
            "turn1_to_final_delta": _safe_pct(strict - turn1, n),
            "avg_turns": avg_turns,
            "avg_route_calls": avg_route,
            "format_error_rate": _safe_pct(fmt_err, n),
            "max_turns_rate": _safe_pct(max_t, n),
            "clean_terminate_rate": _safe_pct(clean, n),
        }

    by_level = {}
    for lvl in sorted({r["level"] for r in records}):
        by_level[str(lvl)] = group_metrics([r for r in records if r["level"] == lvl])
    in_dist = group_metrics([r for r in records if r["level"] in (3, 4)])
    overall = group_metrics(records)
    return {"by_level": by_level, "in_distribution_3_4": in_dist, "overall": overall}


def build_records(convs):
    records = []
    for conv in convs:
        ex = conv["ex"]
        scored_final = score_actions(
            conv["final_actions"],
            ex["layout"],
            ex["start_pos"],
            ex["level"],
            ex["optimal_path_len"],
        )
        scored_turn1 = score_actions(
            conv["first_actions"],
            ex["layout"],
            ex["start_pos"],
            ex["level"],
            ex["optimal_path_len"],
        )
        records.append(
            {
                "id": ex["id"],
                "level": ex["level"],
                "start_pos": ex["start_pos"],
                "target_pos": ex["target_pos"],
                "optimal_path_len": ex["optimal_path_len"],
                "n_turns": len(conv["turn_responses"]),
                "n_route_calls": conv["n_route_calls"],
                "final_kind": conv["final_kind"],
                "format_error_msg": conv["format_error_msg"],
                "final_actions": conv["final_actions"],
                "first_actions": conv["first_actions"],
                "strict_correct": scored_final["strict_correct"],
                "lenient_correct": scored_final["lenient_correct"],
                "optimal": scored_final["optimal"],
                "final_path_length": scored_final["length"],
                "turn1_strict_correct": scored_turn1["strict_correct"],
                "turn1_lenient_correct": scored_turn1["lenient_correct"],
                "turn1_optimal": scored_turn1["optimal"],
                "turn_responses": conv["turn_responses"],
                "messages": conv["messages"],
                "is_correct": int(bool(scored_final["strict_correct"])),
                "final_response": (
                    conv["turn_responses"][-1] if conv["turn_responses"] else None
                ),
            }
        )
    return records


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def _print_group(name, g):
    if g is None:
        print(f"\n# {name}: (empty)")
        return
    print(f"\n# {name} (n={g['n']}):")
    for k in [
        "strict_success_rate",
        "lenient_success_rate",
        "optimal_rate",
        "turn1_strict_rate",
        "final_strict_rate",
        "turn1_to_final_delta",
        "avg_turns",
        "avg_route_calls",
        "format_error_rate",
        "max_turns_rate",
        "clean_terminate_rate",
    ]:
        v = g[k]
        print(f"  {k:24s}: {v:.4f}" if isinstance(v, float) else f"  {k:24s}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="FrozenLake multi-turn eval (route/terminate self-correction)."
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name/path served by vLLM."
    )
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default 0.0 = greedy).",
    )
    parser.add_argument(
        "--vllm_model_port",
        type=int,
        default=None,
        help="vLLM OpenAI-compatible port (e.g. 8900).",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Max tokens per turn (default 512)."
    )
    parser.add_argument(
        "--require_all_responses",
        type=str,
        default="true",
        choices=["true", "false"],
        help="If 'false', tolerate requests that fail after retries "
        "(e.g. finish_reason=length, disconnects): the affected "
        "rollout is scored EM=0 instead of aborting the whole "
        "level. Default 'true' preserves legacy behavior.",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="data/FrozenLake/eval_data/eval_data.json"
    )
    parser.add_argument(
        "--eval_first_k",
        type=int,
        default=0,
        help="If > 0, take first K examples PER LEVEL (balanced).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="If set, random subset of this size total (seed=42).",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="3,4,5,6",
        help="Comma-separated grid levels (default 3,4,5,6).",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=5,
        help="Max route/terminate turns per example (default 5).",
    )
    parser.add_argument(
        "--fmt",
        type=str,
        default="reflection_tag",
        choices=["tag", "reflection_tag"],
        help="Expected model output format: 'reflection_tag' "
        "(active reflection SFT format, default) or 'tag'.",
    )
    parser.add_argument(
        "--prompt_variant",
        type=str,
        default="reflection",
        choices=["reflection", "vprl_direct"],
        help="System prompt variant. Both map to the VPRL-Direct "
        "system prompt used by the released checkpoints.",
    )
    parser.add_argument(
        "--max_image_pixels",
        type=int,
        default=1254400,
        help="Client-side image pixel cap (default 1254400 = 1600*28*28).",
    )
    parser.add_argument(
        "--feedback_max_pixels",
        type=int,
        default=-1,
        help="Feedback-image-only pixel cap. >=0 overrides "
        "--max_image_pixels for rendered feedback PNGs "
        "(initial puzzle image unaffected). -1 = inherit.",
    )
    parser.add_argument(
        "--feedback_render_dpi",
        type=int,
        default=160,
        help="Matplotlib dpi for the feedback renderer. 160 = "
        "default (~118 px/cell). Lower => natively-"
        "rendered smaller feedback image, no resampling.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default eval_outputs/frozenlake_<timestamp>/).",
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="0-indexed shard ID for parallel runs (default 0).",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of shards (default 1 = no sharding).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logging.info(f"args: {args}")

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        args.output_dir = os.path.join("eval_outputs", f"frozenlake_{ts}")
    feedback_dir = os.path.join(args.output_dir, "feedback_images")
    os.makedirs(feedback_dir, exist_ok=True)
    logging.info(f"output_dir = {args.output_dir}")

    examples = load_frozenlake_eval_examples(
        args.eval_dataset,
        levels=levels,
        sample_size=args.sample_size,
        eval_first_k=args.eval_first_k,
    )
    if args.num_shards > 1:
        examples = [
            ex for i, ex in enumerate(examples) if i % args.num_shards == args.shard_idx
        ]
        logging.info(
            f"Shard {args.shard_idx}/{args.num_shards}: "
            f"this shard has {len(examples)} examples."
        )
    logging.info(
        f"Loaded {len(examples)} examples "
        f"(by level: {dict(Counter(e['level'] for e in examples))})"
    )
    if not examples:
        logging.error("No examples to evaluate. Exiting.")
        return

    inferencer = APICallToolUse(
        model_name=args.model_name,
        model_temperature=args.model_temperature,
        vllm_model_port=args.vllm_model_port,
        max_tokens=args.max_tokens,
        require_all_responses=(args.require_all_responses.lower() == "true"),
    )

    evaluator = FrozenLakeMultiTurnEval(
        inferencer=inferencer,
        max_turns=args.max_turns,
        feedback_dir=feedback_dir,
        max_image_pixels=args.max_image_pixels,
        fmt=args.fmt,
        prompt_variant=args.prompt_variant,
        feedback_max_pixels=args.feedback_max_pixels,
        feedback_render_dpi=args.feedback_render_dpi,
    )
    convs = evaluator.run(examples)

    records = build_records(convs)
    metrics = aggregate_metrics(records)

    outputs_path = os.path.join(args.output_dir, "outputs.json")
    result_df = pd.DataFrame(records)
    with open(outputs_path, "w") as f:
        json.dump(result_df.to_dict(orient="records"), f, default=str)
    logging.info(f"Outputs saved to {outputs_path}")

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {metrics_path}")

    print("\n=== FrozenLake Eval Metrics ===")
    _print_group("Overall", metrics["overall"])
    _print_group("In-distribution (levels 3+4)", metrics["in_distribution_3_4"])
    for lvl in sorted(metrics["by_level"]):
        _print_group(f"Level {lvl}x{lvl}", metrics["by_level"][lvl])


if __name__ == "__main__":
    main()
