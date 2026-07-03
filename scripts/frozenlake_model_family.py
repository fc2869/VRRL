"""Detect VLM family and resolve the eval recipe for a FrozenLake checkpoint.

Given a checkpoint dir, resolves what fmt/prompt_variant/max_turns the eval
client should use and which VLM family it is, so the eval prompt always matches
the checkpoint.
"""
import argparse
import json
import os
import sys


def _load_config(ckpt_dir):
    """Find config.json for a merged dir or an FSDP actor dir."""
    for cand in (
        os.path.join(ckpt_dir, "config.json"),
        os.path.join(ckpt_dir, "huggingface", "config.json"),
        os.path.join(ckpt_dir, "actor", "huggingface", "config.json"),
    ):
        if os.path.isfile(cand):
            return json.load(open(cand))
    raise FileNotFoundError(f"no config.json under {ckpt_dir}")


def detect_family(ckpt_dir):
    """Return 'qwen2.5-vl' or 'qwen3-vl' from the checkpoint's config."""
    cfg = _load_config(ckpt_dir)
    blob = (" ".join(cfg.get("architectures", []) or []) + " " + str(cfg.get("model_type", ""))).lower()
    if "qwen3" in blob:
        return "qwen3-vl"
    if "qwen2_5" in blob or "qwen2.5" in blob or "qwen2_vl" in blob:
        return "qwen2.5-vl"
    raise ValueError(f"unrecognized VLM family from config: {blob!r}")


def recipe_for(ckpt_dir):
    """Return {'fmt', 'prompt_variant', 'max_turns'} for the eval client.

    Project default = multi-turn reflection. Single-turn VPRL-Direct / single-turn-SFT
    bases override to the single-turn tag recipe (max_turns=2).
    """
    path = ckpt_dir.lower()
    if "vprl_direct" in path or "single_turn" in path or "singleturn" in path:
        return {"fmt": "tag", "prompt_variant": "vprl_direct", "max_turns": 2}
    return {"fmt": "reflection_tag", "prompt_variant": "reflection", "max_turns": 10}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", help="print the VLM family for this checkpoint dir")
    ap.add_argument("--recipe", help="print 'fmt prompt_variant max_turns' for this checkpoint dir")
    a = ap.parse_args()
    if a.family:
        print(detect_family(a.family))
    elif a.recipe:
        r = recipe_for(a.recipe)
        print(f"{r['fmt']} {r['prompt_variant']} {r['max_turns']}")
    else:
        ap.error("pass --family DIR or --recipe DIR")


if __name__ == "__main__":
    sys.exit(main())
