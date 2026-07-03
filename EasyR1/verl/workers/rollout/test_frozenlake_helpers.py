import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frozenlake_helpers as h
from PIL import Image as _PILImage

MAP_OPEN = {"layout": ["SFF", "FFF", "FFG"], "start_pos": 0, "level": 3}
TURN_ROUTE = '{"think": "x", "function_call": {"name": "route", "arguments": {"actions": "down, right"}}}'
TURN_TERM = '{"think": "x", "function_call": {"name": "terminate", "arguments": {"answer": "down, right"}}}'


def test_parse_route_terminate_route():
    # JSON inputs require fmt="json" explicitly under the new fmt flag.
    kind, actions, err = h.parse_route_terminate(TURN_ROUTE, fmt="json")
    assert kind == "route"
    assert actions == ["down", "right"]
    assert err is None


def test_parse_route_terminate_terminate():
    kind, actions, err = h.parse_route_terminate(TURN_TERM, fmt="json")
    assert kind == "terminate"
    assert actions == ["down", "right"]
    assert err is None


def test_parse_route_terminate_bad():
    # garbage fails in both fmts
    for fmt in ("tag", "json"):
        kind, actions, err = h.parse_route_terminate("not json at all", fmt=fmt)
        assert kind is None
        assert actions is None
        assert err  # non-empty diagnostic


def test_parse_route_terminate_default_fmt_rejects_json():
    # Default fmt="tag" must NOT silently accept JSON inputs.
    kind, _, err = h.parse_route_terminate(TURN_ROUTE)
    assert kind is None
    assert err


def test_is_finished_on_terminate():
    # a clean terminate stops the loop
    assert h.is_finished(TURN_TERM, fmt="json") is True


def test_is_finished_on_route():
    # a route does not stop the loop
    assert h.is_finished(TURN_ROUTE, fmt="json") is False


def test_is_finished_on_garbage():
    # unparseable -> stop with format_error (don't keep rolling)
    assert h.is_finished("not json") is True


def test_render_feedback_pil_returns_image():
    img = h.render_feedback_pil(MAP_OPEN["layout"], MAP_OPEN["start_pos"],
                                ["down", "right"], MAP_OPEN["level"])
    assert isinstance(img, _PILImage.Image)
    assert img.size[0] > 50 and img.size[1] > 50  # non-trivial render


# ---------------------------------------------------------------------------
# Tag-format parser
# ---------------------------------------------------------------------------

TAG_ROUTE = "<think>plan</think><route>down right</route>"
TAG_ANSWER = "<think>final</think><answer>right right down down</answer>"


def test_parse_route_terminate_tag_route():
    kind, actions, err = h.parse_route_terminate(TAG_ROUTE)
    assert kind == "route"
    assert actions == ["down", "right"]
    assert err is None


def test_parse_route_terminate_tag_answer_is_terminate():
    # <answer> must surface as kind "terminate" so loop control is unchanged.
    kind, actions, err = h.parse_route_terminate(TAG_ANSWER)
    assert kind == "terminate"
    assert actions == ["right", "right", "down", "down"]
    assert err is None


def test_is_finished_on_tag_answer():
    assert h.is_finished(TAG_ANSWER) is True
    assert h.is_finished(TAG_ROUTE) is False


# ---------------------------------------------------------------------------
# Reflection-format system prompt + Jinja template + feedback prompt
# ---------------------------------------------------------------------------

REFLECTION_ANSWER = "<ANSWER>down right</ANSWER>"
REFLECTION_THINK_ANSWER = "<THINK>need to revise</THINK><ANSWER>right down</ANSWER>"
REFLECTION_FINAL = "<FINAL>right right down down</FINAL>"


def test_system_prompt_reflection_alias_matches_canonical():
    # The rollout-side alias must be byte-equal to the canonical VPRL Direct
    # prompt in repo-root prompts.py (the SFT-side source).
    import prompts as _sp  # repo root is on sys.path via helpers
    assert isinstance(h.FROZENLAKE_SYSTEM_PROMPT_REFLECTION, str)
    assert h.FROZENLAKE_SYSTEM_PROMPT_REFLECTION == _sp.SYSTEM_PROMPT_TAG_VPRL_DIRECT, \
        "frozenlake_helpers.FROZENLAKE_SYSTEM_PROMPT_REFLECTION has drifted " \
        "from prompts.py:SYSTEM_PROMPT_TAG_VPRL_DIRECT"
    # Sanity: the prompt mentions the uppercase ANSWER tag so a casual reader
    # can tell which protocol it describes.
    assert "<ANSWER>" in h.FROZENLAKE_SYSTEM_PROMPT_REFLECTION


def test_feedback_user_prompt_reflection_matches_eval_source():
    # The rollout-side feedback prompt must stay byte-equal to the eval-side
    # constant in api_inference_frozenlake.py (repo root is on sys.path via
    # helpers).
    from api_inference_frozenlake import FEEDBACK_USER_PROMPT_REFLECTION as _EVAL
    assert h.FEEDBACK_USER_PROMPT_REFLECTION == _EVAL, \
        "frozenlake_helpers.FEEDBACK_USER_PROMPT_REFLECTION drifted from " \
        "api_inference_frozenlake.FEEDBACK_USER_PROMPT_REFLECTION"


def test_jinja_template_reflection_matches_helper_prompt():
    """The reflection jinja file renders to the same system prompt the helpers
    expose -- byte-equal modulo leading/trailing whitespace from the Jinja
    control structures."""
    from jinja2 import Template
    here = os.path.dirname(os.path.abspath(__file__))
    jinja_path = os.path.join(here, "..", "..", "..", "training",
                              "format_prompt", "frozenlake_reflection.jinja")
    jinja_path = os.path.normpath(jinja_path)
    rendered = Template(open(jinja_path, encoding="utf-8").read()).render(content="").strip()
    assert rendered == h.FROZENLAKE_SYSTEM_PROMPT_REFLECTION.strip()


def test_parse_route_terminate_reflection_answer():
    # <ANSWER> in reflection_tag fmt = proposal (kind="route").
    kind, actions, err = h.parse_route_terminate(REFLECTION_ANSWER,
                                                 fmt="reflection_tag")
    assert kind == "route"
    assert actions == ["down", "right"]
    assert err is None


def test_parse_route_terminate_reflection_think_answer():
    kind, actions, err = h.parse_route_terminate(REFLECTION_THINK_ANSWER,
                                                 fmt="reflection_tag")
    assert kind == "route"
    assert actions == ["right", "down"]
    assert err is None


def test_parse_route_terminate_reflection_final():
    # <FINAL> in reflection_tag fmt = commit (kind="terminate").
    kind, actions, err = h.parse_route_terminate(REFLECTION_FINAL,
                                                 fmt="reflection_tag")
    assert kind == "terminate"
    assert actions == ["right", "right", "down", "down"]
    assert err is None


def test_is_finished_on_reflection_final():
    # A clean <FINAL> stops the loop; a bare <ANSWER> does not.
    assert h.is_finished(REFLECTION_FINAL, fmt="reflection_tag") is True
    assert h.is_finished(REFLECTION_ANSWER, fmt="reflection_tag") is False
    assert h.is_finished(REFLECTION_THINK_ANSWER, fmt="reflection_tag") is False
