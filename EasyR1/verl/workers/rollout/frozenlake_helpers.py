"""Pure-Python helpers for the FrozenLake multi-turn rollout.

Imports the parser from `frozenlake_score` (the reward module, which is stdlib-only)
and the gym-FrozenLake renderer from the repo-root `api_inference_frozenlake.py`.
Adding sys.path entries keeps this module importable from any cwd.
"""

import base64
import io
import os
import re
import sys

# repo root + reward module dir on sys.path for the parser, renderer, and
# tag-format system-prompt imports (from repo-root prompts.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
_REWARD = os.path.join(_REPO, "EasyR1", "training", "reward_function")
for _p in (_REPO, _REWARD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# parser: single source of truth lives in the reward module
import frozenlake_score as _fz


# Reflection-format system prompt, re-exported from repo-root prompts.py. The
# Jinja template at EasyR1/training/format_prompt/frozenlake_reflection.jinja
# MUST render to this string (verified by test_frozenlake_helpers.py).
import prompts as _sp_tag  # noqa: E402
FROZENLAKE_SYSTEM_PROMPT_REFLECTION = _sp_tag.SYSTEM_PROMPT_TAG_VPRL_DIRECT

# Reflection feedback-turn user prompt. Appended to every feedback-turn user
# message during RL rollout when self.fmt == "reflection_tag", mirroring the
# eval-side wiring in api_inference_frozenlake.py:FEEDBACK_USER_PROMPT_REFLECTION.
FEEDBACK_USER_PROMPT_REFLECTION = (
    "The image shows your proposed action sequence executed on the grid.\n"
    "Verify the path against the task requirements. If the path reaches the\n"
    "gift without stepping on any ice holes, confirm with\n"
    "<FINAL>actions</FINAL>. Otherwise, provide your reasoning in\n"
    "<THINK>...</THINK> and a revised sequence in <ANSWER>actions</ANSWER>."
)


def parse_route_terminate(text, fmt="tag"):
    """Parse one assistant turn into (kind, actions, error).

    Args:
        text: raw assistant turn text.
        fmt: which output format to expect: "tag" (default) or "json"
            (function_call schema).

    Delegates to ``frozenlake_score._parse_one_turn`` and reshapes the
    return tuple to (kind, actions, error_str_or_None).
    """
    kind, actions, ok = _fz._parse_one_turn(text, fmt=fmt)
    if ok and kind is not None:
        return kind, actions, None
    return None, None, f"could not parse a valid route/terminate {fmt} response"


def is_finished(turn_text, fmt="tag"):
    """Return True if the loop should stop after this assistant turn.

    Stops on: (a) a parseable ``terminate`` (or ``<answer>`` in tag fmt),
    (b) any parse failure (the model never saw recovery turns in SFT, so
    injecting one would be off-distribution). A parseable ``route`` returns
    False -- continue.

    ``fmt`` selects "tag" or "json"; see ``parse_route_terminate``.
    """
    kind, _actions, err = parse_route_terminate(turn_text, fmt=fmt)
    if err is not None:
        return True
    return kind == "terminate"


def save_batch_logs(batch_dir, samples):
    """Write feedback PNGs + transcripts.jsonl for one rollout batch.

    `samples` is a list of dicts, one per rollout sample:
        {
          "sample_idx": int,
          "sequence":   str,                # full conversation transcript
          "map_spec":   {"layout","start_pos","level"},
          "finish_reason": str or None,
          "feedback_images": [PIL.Image],   # in turn order (route 1 feedback first)
          "base_map_image": PIL.Image | None,  # OPTIONAL: base map at rollout time
          "mode":       "normal"|"random_start"|"prefix_buffer" | None,
          "prefix_type": "wrong" | "right" | None,
        }

    Saves feedback_images[k] as `sample_{idx:03d}_turn_{k+1:02d}.png`, optionally
    saves base_map_image as `sample_{idx:03d}_base.png`, and appends one JSONL
    line per sample to `{batch_dir}/transcripts.jsonl`. Returns the log path.
    """
    import os as _os
    import json as _json
    _os.makedirs(batch_dir, exist_ok=True)
    log_path = _os.path.join(batch_dir, "transcripts.jsonl")
    with open(log_path, "w", encoding="utf-8") as logf:
        for s in samples:
            idx = int(s["sample_idx"])
            image_filenames = []
            for k, img in enumerate(s.get("feedback_images") or []):
                if img is None:
                    continue
                fname = "sample_%03d_turn_%02d.png" % (idx, k + 1)
                img.save(_os.path.join(batch_dir, fname))
                image_filenames.append(fname)
            # Optionally persist the base map too so downstream viz can confirm
            # Q_A vs Q_B alignment without re-loading from the dataset.
            base_fname = None
            base_img = s.get("base_map_image")
            if base_img is not None:
                base_fname = "sample_%03d_base.png" % idx
                try:
                    base_img.save(_os.path.join(batch_dir, base_fname))
                except Exception:
                    base_fname = None
            rec = {
                "sample_idx": idx,
                "map_spec": s.get("map_spec"),
                "finish_reason": s.get("finish_reason"),
                "sequence": s.get("sequence", ""),
                "feedback_image_filenames": image_filenames,
                "base_map_image_filename": base_fname,
                "mode": s.get("mode"),
                "prefix_type": s.get("prefix_type"),
            }
            logf.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    return log_path


def render_feedback_pil(layout, start_pos, actions, level):
    """Return the rendered FrozenLake trajectory image as a PIL.Image.

    Reuses `render_frozenlake_trajectory_with_walls_data_uri` and decodes the
    base64 data URI back into a PIL image. Returns None on render failure.
    """
    # imported lazily so unit tests don't pay for matplotlib at module-load
    from PIL import Image
    from api_inference_frozenlake import render_frozenlake_trajectory_with_walls_data_uri
    uri = render_frozenlake_trajectory_with_walls_data_uri(
        layout, start_pos, actions, level)
    if uri is None:
        return None
    m = re.match(r"^data:image/png;base64,(.+)$", uri, re.DOTALL)
    if not m:
        return None
    return Image.open(io.BytesIO(base64.b64decode(m.group(1)))).convert("RGB")
