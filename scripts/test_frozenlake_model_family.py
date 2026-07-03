"""Tests for frozenlake_model_family. Runs under bare python (no pytest needed):
    python test_frozenlake_model_family.py
(also pytest-compatible if pytest is available)."""
import json, os, tempfile, subprocess, sys
import frozenlake_model_family as fmf


def _mk(cfg):
    d = tempfile.mkdtemp()
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    return d


def test_detect_qwen25():
    d = _mk({"architectures": ["Qwen2_5_VLForConditionalGeneration"], "model_type": "qwen2_5_vl"})
    assert fmf.detect_family(d) == "qwen2.5-vl"


def test_detect_qwen3():
    d = _mk({"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl"})
    assert fmf.detect_family(d) == "qwen3-vl"


def test_detect_actor_fallback():
    d = tempfile.mkdtemp()
    hf = os.path.join(d, "huggingface")
    os.makedirs(hf)
    json.dump({"architectures": ["Qwen3VLForConditionalGeneration"]}, open(os.path.join(hf, "config.json"), "w"))
    assert fmf.detect_family(d) == "qwen3-vl"


def test_recipe_reflection_default():
    d = _mk({"architectures": ["Qwen3VLForConditionalGeneration"]})
    r = fmf.recipe_for(d)  # reflection multi-turn is the project default
    assert r == {"fmt": "reflection_tag", "prompt_variant": "reflection", "max_turns": 10}


def test_recipe_vprl_direct_override():
    d = tempfile.mkdtemp()
    sub = os.path.join(d, "qwen3_vl_4b_frozenlake_sft_vprl_direct_l3_l5", "merged")
    os.makedirs(sub)
    json.dump({"architectures": ["Qwen3VLForConditionalGeneration"]}, open(os.path.join(sub, "config.json"), "w"))
    assert fmf.recipe_for(sub) == {"fmt": "tag", "prompt_variant": "vprl_direct", "max_turns": 2}


def test_cli_recipe():
    d = _mk({"architectures": ["Qwen2_5_VLForConditionalGeneration"]})
    out = subprocess.check_output(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozenlake_model_family.py"), "--recipe", d]
    ).decode().strip()
    assert out == "reflection_tag reflection 10"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
