import pytest
import frozenlake_score as fz

# 3x3, no holes. layout rows; flat indices row-major. start=0 (top-left), goal=8.
MAP_OPEN = {"layout": ["SFF", "FFF", "FFG"], "start_pos": 0, "target_pos": 8, "level": 3}
# 3x3 with a hole at (1,1) = flat index 4.
MAP_HOLE = {"layout": ["SFF", "FHF", "FFG"], "start_pos": 0, "target_pos": 8, "level": 3}


def test_bfs_dist_open_and_hole():
    assert fz.bfs_dist(MAP_OPEN["layout"], 0, 8, 3) == 4
    assert fz.bfs_dist(MAP_HOLE["layout"], 0, 8, 3) == 4  # detours around hole, still 4


def test_is_em_true_on_optimal():
    assert fz.is_em(["right", "right", "down", "down"], MAP_OPEN["layout"], 0, 8, 3, 4) is True


def test_is_em_false_when_short_or_hole():
    assert fz.is_em(["down", "right"], MAP_OPEN["layout"], 0, 8, 3, 4) is False
    # steps onto the hole at (1,1)
    assert fz.is_em(["right", "down", "down"], MAP_HOLE["layout"], 0, 8, 3, 4) is False


def test_is_em_false_on_passthrough():
    # Strict EM: a passthrough that reaches the goal but keeps going is NOT EM.
    # Example: 4x4 map where goal is at index 5 (n_opt=2). The model emits
    # 3 actions; even though step 2 hits the goal, the extra step disqualifies it.
    layout = ["SHHH", "FGHF", "FFFH", "FFHH"]  # G at (1,1) = index 5
    assert fz.is_em(["down", "right", "right"], layout, 0, 5, 4, 2) is False
    # And a clean exactly-n_opt trajectory IS EM.
    assert fz.is_em(["down", "right"], layout, 0, 5, 4, 2) is True
    # Goal must be hit on the LAST action; reaching goal early then doing more is rejected.
    assert fz.is_em(["right", "right", "down", "down", "down"], MAP_OPEN["layout"], 0, 8, 3, 4) is False


def test_progress_rate_values():
    assert fz.progress_rate(["right", "right", "down", "down"], MAP_OPEN["layout"], 0, 8, 3, 4) == 1.0
    assert fz.progress_rate(["down", "right"], MAP_OPEN["layout"], 0, 8, 3, 4) == 0.5
    assert fz.progress_rate(["up", "right"], MAP_OPEN["layout"], 0, 8, 3, 4) == 0.0  # off-grid first step
    assert fz.progress_rate(["right", "up"], MAP_OPEN["layout"], 0, 8, 3, 4) == 0.25


# ---------------------------------------------------------------------------
# Task 2: Transcript parser (route/terminate turns)
# ---------------------------------------------------------------------------

TURN_ROUTE = '{"think": "go down then right", "function_call": {"name": "route", "arguments": {"actions": "down, right"}}}'
TURN_TERM = '{"think": "confirmed", "function_call": {"name": "terminate", "arguments": {"answer": "right, right, down, down"}}}'
USER_FEEDBACK = "\n<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n"
TWO_TURN_RESPONSE = TURN_ROUTE + USER_FEEDBACK + TURN_TERM + "<|im_end|>"


def test_parse_actions_csv():
    assert fz._parse_actions_csv("right, right, down") == (["right", "right", "down"], True)
    assert fz._parse_actions_csv("") == ([], True)
    assert fz._parse_actions_csv("right, sideways") == (None, False)
    assert fz._parse_actions_csv(None) == (None, False)


def test_parse_one_turn():
    # fmt="json" is required to parse the legacy function-call format
    assert fz._parse_one_turn(TURN_ROUTE, fmt="json") == ("route", ["down", "right"], True)
    assert fz._parse_one_turn(TURN_TERM, fmt="json") == ("terminate", ["right", "right", "down", "down"], True)
    assert fz._parse_one_turn("not json at all", fmt="json") == (None, None, False)
    # default fmt="tag" should NOT parse JSON inputs
    assert fz._parse_one_turn(TURN_ROUTE) == (None, None, False)


def test_parse_turns_two_turn_transcript():
    turns = fz.parse_turns(TWO_TURN_RESPONSE, fmt="json")
    assert [t["kind"] for t in turns] == ["route", "terminate"]
    assert turns[0]["actions"] == ["down", "right"]
    assert turns[1]["actions"] == ["right", "right", "down", "down"]
    assert all(t["ok"] for t in turns)


def test_parse_turns_empty_response():
    # empty input parses to [] for either fmt
    assert fz.parse_turns("") == []
    assert fz.parse_turns("   \n  ") == []
    assert fz.parse_turns("", fmt="json") == []


def test_parse_turns_strips_leaked_vision_prefix():
    # a decoded response can occasionally start with leaked vision/image tokens;
    # parsing must still recover the same turns
    leaked = "<|image_pad|><|vision_end|>" + TWO_TURN_RESPONSE
    turns = fz.parse_turns(leaked, fmt="json")
    assert [t["kind"] for t in turns] == ["route", "terminate"]
    assert turns[0]["actions"] == ["down", "right"]
    assert turns[1]["actions"] == ["right", "right", "down", "down"]


def test_parse_one_turn_unknown_fmt_raises():
    import pytest as _pt
    with _pt.raises(ValueError, match="unknown fmt"):
        fz._parse_one_turn(TURN_ROUTE, fmt="xml")


# ---------------------------------------------------------------------------
# Task 3: Illegal-move detection and ground-truth decoding
# ---------------------------------------------------------------------------

import json as _json


def test_simulate_has_illegal():
    # clean optimal path on the open map: no illegal step
    assert fz.simulate_has_illegal(["right", "right", "down", "down"], MAP_OPEN["layout"], 0, 3) is False
    # steps onto the hole at (1,1) on the hole map
    assert fz.simulate_has_illegal(["right", "down"], MAP_HOLE["layout"], 0, 3) is True
    # first move leaves the grid
    assert fz.simulate_has_illegal(["up"], MAP_OPEN["layout"], 0, 3) is True
    # empty action list is vacuously legal
    assert fz.simulate_has_illegal([], MAP_OPEN["layout"], 0, 3) is False


def test_trajectory_leaves_grid():
    # on-grid optimal path
    assert fz.trajectory_leaves_grid(["right", "right", "down", "down"], MAP_OPEN["layout"], 0, 3) is False
    # first move leaves the grid
    assert fz.trajectory_leaves_grid(["up"], MAP_OPEN["layout"], 0, 3) is True
    # a later move leaves the grid
    assert fz.trajectory_leaves_grid(["right", "up"], MAP_OPEN["layout"], 0, 3) is True
    # stepping onto a hole is NOT leaving the grid (off-grid only)
    assert fz.trajectory_leaves_grid(["right", "down"], MAP_HOLE["layout"], 0, 3) is False
    # empty action list is vacuously on-grid
    assert fz.trajectory_leaves_grid([], MAP_OPEN["layout"], 0, 3) is False


def test_decode_ground_truth_accepts_dict_and_json_string():
    assert fz._decode_ground_truth(MAP_OPEN) == (["SFF", "FFF", "FFG"], 0, 8, 3)
    assert fz._decode_ground_truth(_json.dumps(MAP_OPEN)) == (["SFF", "FFF", "FFG"], 0, 8, 3)


# ---------------------------------------------------------------------------
# Task 4: Delta-form reflection score
# ---------------------------------------------------------------------------


def test_reflection_score_zero_without_two_routes():
    # reflection requires >= 2 route proposals; otherwise no revision happened
    assert fz.compute_reflection_score([], lambda_deg=1.0) == 0.0
    assert fz.compute_reflection_score([0.7], lambda_deg=1.0) == 0.0
    assert fz.compute_reflection_score([1.0], lambda_deg=1.0) == 0.0


def test_reflection_score_sums_signed_deltas():
    # R_reflect = clip(sum_t w(delta_t)) -- PR_1 is NOT included.
    assert fz.compute_reflection_score([0.25, 1.0], lambda_deg=1.0) == 0.75  # one +0.75 delta
    # [0.5, 0.25, 0.75] -> w(-0.25) + w(0.5) = 0.25 at lambda=1.0
    assert fz.compute_reflection_score([0.5, 0.25, 0.75], lambda_deg=1.0) == 0.25


def test_reflection_score_penalizes_backslide_when_lambda_gt_one():
    # progress only: lambda has no effect
    assert fz.compute_reflection_score([0.0, 0.5, 1.0], lambda_deg=2.0) == 1.0
    # pure backslide: 0.5 + 2.0*(-0.25) = 0.0
    assert fz.compute_reflection_score([0.5, 0.25], lambda_deg=2.0) == 0.0
    # dip then recover: the running sum is NOT clipped between steps, so
    # 0.5 -> 0.5 + 2.0*(-0.5) = -0.5 -> -0.5 + 0.5 = 0.0, final clip -> 0.0
    assert fz.compute_reflection_score([0.5, 0.0, 0.5], lambda_deg=2.0) == 0.0


# ---- lower_clip: surfaces breaks in the reward signal -------------------

def test_reflection_score_default_lower_clip_is_zero():
    # Backward-compat: no kwarg => clip floor stays at 0.0.
    assert fz.compute_reflection_score([1.0, 0.5], lambda_deg=1.0) == 0.0


def test_reflection_score_lower_clip_allows_negative_for_break():
    # T1 correct (PR=1.0) -> break (PR=0.5). delta=-0.5, lambda=1.0 => sum=-0.5.
    # New floor lets it through to -0.05 (or whatever was specified).
    assert fz.compute_reflection_score([1.0, 0.5], lambda_deg=1.0,
                                       lower_clip=-0.05) == -0.05
    assert fz.compute_reflection_score([1.0, 0.5], lambda_deg=1.0,
                                       lower_clip=-0.1) == -0.1
    # And smaller breaks still saturate to the floor when sum < floor:
    assert fz.compute_reflection_score([1.0, 0.4], lambda_deg=1.0,
                                       lower_clip=-0.05) == -0.05


def test_reflection_score_lower_clip_does_not_affect_positive_sums():
    # Recovery: PR_T1=0.0 -> PR_T2=0.5. sum=+0.5, floor irrelevant.
    assert fz.compute_reflection_score([0.0, 0.5], lambda_deg=1.0,
                                       lower_clip=-0.5) == 0.5
    # Capped at upper bound 1.0 regardless of floor:
    assert fz.compute_reflection_score([0.0, 1.0, 1.0], lambda_deg=1.0,
                                       lower_clip=-0.5) == 1.0


def test_reflection_score_lower_clip_does_not_apply_when_too_few_routes():
    # Single route turn: no delta computed -> 0.0 regardless of floor.
    assert fz.compute_reflection_score([0.7], lambda_deg=1.0,
                                       lower_clip=-0.5) == 0.0


def test_reflection_score_negative_lower_clip_passes_through_small_break():
    # delta=-0.1, lambda=1.0 => sum=-0.1. With floor=-0.5 the raw sum survives.
    assert fz.compute_reflection_score([1.0, 0.9], lambda_deg=1.0,
                                       lower_clip=-0.5) == pytest.approx(-0.1)


# ---------------------------------------------------------------------------
# Task 5: Format/gate score
# ---------------------------------------------------------------------------


def _turns(*kinds_actions_ok):
    return [{"kind": k, "actions": a, "ok": o} for (k, a, o) in kinds_actions_ok]


def test_format_score_valid_structure():
    opt = ["right", "right", "down", "down"]
    # route -> revised route -> terminate that commits the last route unchanged
    turns = _turns(("route", ["down", "right"], True),
                   ("route", opt, True),
                   ("terminate", opt, True))
    assert fz.compute_format_score(turns, opt, MAP_OPEN["layout"], 0, 3) == 0.1


def test_format_score_zero_on_parse_failure():
    turns = _turns(("route", ["down"], True), (None, None, False))
    assert fz.compute_format_score(turns, ["down"], MAP_OPEN["layout"], 0, 3) == 0.0


def test_format_score_zero_on_bad_structure():
    # terminate with no preceding route
    turns = _turns(("terminate", ["right"], True))
    assert fz.compute_format_score(turns, ["right"], MAP_OPEN["layout"], 0, 3) == 0.0
    # no terminate at all (hit max turns)
    turns = _turns(("route", ["down"], True), ("route", ["down", "right"], True))
    assert fz.compute_format_score(turns, ["down", "right"], MAP_OPEN["layout"], 0, 3) == 0.0


def test_format_score_terminate_must_match_last_route():
    # constraint 1: terminate edits the answer instead of committing the last route
    opt = ["right", "right", "down", "down"]
    turns = _turns(("route", ["down", "right"], True),
                   ("route", ["down", "down"], True),
                   ("terminate", opt, True))
    assert fz.compute_format_score(turns, opt, MAP_OPEN["layout"], 0, 3) == 0.0


def test_format_score_rejects_repeated_routes():
    # constraint 3: two route turns with an identical trajectory
    dup = ["down", "right"]
    turns = _turns(("route", dup, True),
                   ("route", dup, True),
                   ("terminate", dup, True))
    assert fz.compute_format_score(turns, dup, MAP_OPEN["layout"], 0, 3) == 0.0


def test_format_score_rejects_empty_trajectory():
    turns = _turns(("route", [], True), ("terminate", [], True))
    assert fz.compute_format_score(turns, [], MAP_OPEN["layout"], 0, 3) == 0.0


def test_format_score_min_turns_gate():
    opt = ["right", "right", "down", "down"]
    # 2 turns (one route + terminate): passes min_turns=1, fails min_turns=3
    turns2 = _turns(("route", opt, True), ("terminate", opt, True))
    assert fz.compute_format_score(turns2, opt, MAP_OPEN["layout"], 0, 3, min_turns=1) == 0.1
    assert fz.compute_format_score(turns2, opt, MAP_OPEN["layout"], 0, 3, min_turns=3) == 0.0
    # 3 turns (a revised route): passes min_turns=3
    turns3 = _turns(("route", ["down", "right"], True), ("route", opt, True), ("terminate", opt, True))
    assert fz.compute_format_score(turns3, opt, MAP_OPEN["layout"], 0, 3, min_turns=3) == 0.1


def test_format_score_allows_hole_in_committed_answer():
    # A committed answer that steps on a hole is NOT a format failure. The
    # strict EM check (not this gate) is what blocks such trajectories from
    # earning the correct-branch reward.
    turns = _turns(("route", ["right", "down"], True),
                   ("terminate", ["right", "down"], True))
    final = ["right", "down"]  # steps onto the hole on MAP_HOLE
    assert fz.compute_format_score(turns, final, MAP_HOLE["layout"], 0, 3) == 0.1


def test_format_score_allows_off_grid_at_any_turn():
    # An off-grid intermediate route is NOT a format failure. The model is
    # expected to recover via reflection; strict EM independently rejects
    # illegal committed trajectories so they cannot earn R = 1.0.
    opt = ["right", "right", "down", "down"]
    turns = _turns(("route", ["up"], True),          # 'up' from (0,0) leaves the grid
                   ("route", opt, True),
                   ("terminate", opt, True))
    assert fz.compute_format_score(turns, opt, MAP_OPEN["layout"], 0, 3) == 0.1


# ---------------------------------------------------------------------------
# Task 6: compute_score integration
# ---------------------------------------------------------------------------

# well-formed transcript: route "down, right" -> revised route optimal -> terminate
# commits the last route unchanged (EM) on MAP_OPEN
_TURN_ROUTE_OPT = '{"think": "go around", "function_call": {"name": "route", "arguments": {"actions": "right, right, down, down"}}}'
_RESP_FIX_TO_EM = TURN_ROUTE + USER_FEEDBACK + _TURN_ROUTE_OPT + USER_FEEDBACK + TURN_TERM + "<|im_end|>"
# transcript: turn-1 route "down, right" (PR 0.5), terminate "down, right" (PR 0.5, not EM)
_TERM_HALF = '{"think": "good enough", "function_call": {"name": "terminate", "arguments": {"answer": "down, right"}}}'
_RESP_PARTIAL = TURN_ROUTE + USER_FEEDBACK + _TERM_HALF + "<|im_end|>"
# unparseable transcript
_RESP_GARBAGE = "this is not json" + "<|im_end|>"


def test_compute_score_correct_gets_one():
    out = fz.compute_score([{"response": _RESP_FIX_TO_EM, "ground_truth": MAP_OPEN}], lambda_deg=1.0, fmt="json")
    assert out[0]["overall"] == 1.0
    assert out[0]["em"] == 1.0


def test_compute_score_single_route_gets_format_only():
    # _RESP_PARTIAL has a single route turn -> no reflection happened ->
    # R_reflect = 0.0, so R = R_format(0.1) + 0.9 * 0.0 = 0.1
    out = fz.compute_score([{"response": _RESP_PARTIAL, "ground_truth": MAP_OPEN}], lambda_deg=1.0, fmt="json")
    assert out[0]["em"] == 0.0
    assert out[0]["reflection"] == 0.0
    assert abs(out[0]["overall"] - 0.1) < 1e-9


def _break_transcript():
    # T1 = optimal (PR=1.0 on MAP_OPEN). Revision steers off-grid after 2 steps
    # (PR=0.5). Terminate commits the broken revision.
    r1 = '{"think": "x", "function_call": {"name": "route", "arguments": {"actions": "right, right, down, down"}}}'
    r2 = '{"think": "x", "function_call": {"name": "route", "arguments": {"actions": "right, right, up, up"}}}'
    tm = '{"think": "x", "function_call": {"name": "terminate", "arguments": {"answer": "right, right, up, up"}}}'
    return r1 + USER_FEEDBACK + r2 + USER_FEEDBACK + tm + "<|im_end|>"


def test_compute_score_em_turn1_and_em_final_fields_present():
    # Backward compat: legacy `em` stays, AND `em_turn1` / `em_final` are added.
    # _RESP_FIX_TO_EM commits a paper-EM final after a wrong T1 -> em_final=1, em_turn1=0.
    out = fz.compute_score(
        [{"response": _RESP_FIX_TO_EM, "ground_truth": MAP_OPEN}],
        lambda_deg=1.0, fmt="json")[0]
    assert "em" in out
    assert "em_turn1" in out
    assert "em_final" in out
    assert out["em"] == out["em_final"]      # backward compat: em mirrors em_final
    assert out["em_final"] == 1.0
    assert out["em_turn1"] == 0.0            # T1 was not optimal


def test_compute_score_em_turn1_one_when_t1_optimal():
    # Single optimal route turn -> em_turn1 = 1.0, em_final = 1.0.
    resp = (
        '{"think": "x", "function_call": {"name": "route", "arguments": {"actions": "right, right, down, down"}}}'
        + USER_FEEDBACK +
        '{"think": "x", "function_call": {"name": "terminate", "arguments": {"answer": "right, right, down, down"}}}'
        + "<|im_end|>"
    )
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           lambda_deg=1.0, fmt="json")[0]
    assert out["em_turn1"] == 1.0
    assert out["em_final"] == 1.0


def test_compute_score_break_flag_zero_with_no_revision():
    # Single route + terminate (no revision) -> break = 0 regardless of EM.
    out = fz.compute_score(
        [{"response": _RESP_FIX_TO_EM, "ground_truth": MAP_OPEN}],
        lambda_deg=1.0, fmt="json")[0]
    assert out["break"] == 0.0


def test_compute_score_break_flag_one_on_t1_optimal_revised_wrong():
    # Break: PR T1=1.0 -> revision PR=0.5. Raw sum < 0 => break=1.
    out = fz.compute_score(
        [{"response": _break_transcript(), "ground_truth": MAP_OPEN}],
        lambda_deg=1.0, fmt="json")[0]
    assert out["break"] == 1.0
    assert out["em_turn1"] == 1.0    # T1 was optimal
    assert out["em_final"] == 0.0    # but the broken revision was committed


def test_compute_score_break_flag_zero_on_recovery():
    # Recovery: T1 wrong (PR=0.25) -> revised to better (PR=0.75). sum > 0 => break=0.
    r1 = '{"think": "x", "function_call": {"name": "route", "arguments": {"actions": "down"}}}'
    r2 = '{"think": "x", "function_call": {"name": "route", "arguments": {"actions": "down, right, down"}}}'
    tm = '{"think": "x", "function_call": {"name": "terminate", "arguments": {"answer": "down, right, down"}}}'
    resp = r1 + USER_FEEDBACK + r2 + USER_FEEDBACK + tm + "<|im_end|>"
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           lambda_deg=1.0, fmt="json")[0]
    assert out["break"] == 0.0
    assert out["em_turn1"] == 0.0    # T1 wasn't optimal
    assert out["em_final"] == 0.0    # neither was the revision (PR=0.75 < 1)


def test_compute_score_break_flag_independent_of_lower_clip():
    # The break flag tracks the RAW (pre-clip) sum so it's comparable across
    # runs with different reflect_lower_clip values.
    for clip in (0.0, -0.05, -0.5):
        out = fz.compute_score(
            [{"response": _break_transcript(), "ground_truth": MAP_OPEN}],
            lambda_deg=1.0, fmt="json", reflect_lower_clip=clip)[0]
        assert out["break"] == 1.0, f"break should be 1 at clip={clip}"


def test_floor_result_includes_new_metrics():
    # _floor_result is what bad ground truth / unparsable rollouts get;
    # it must include the new keys so wandb aggregation doesn't KeyError.
    out = fz.compute_score(
        [{"response": "garbage", "ground_truth": "not json"}],
        lambda_deg=1.0, fmt="json")[0]
    assert "em_turn1" in out and out["em_turn1"] == 0.0
    assert "em_final" in out and out["em_final"] == 0.0
    assert "break" in out and out["break"] == 0.0


def test_compute_score_reflect_lower_clip_default_floors_break_to_zero():
    # Legacy behavior: break is invisible (R_reflect = 0).
    out = fz.compute_score(
        [{"response": _break_transcript(), "ground_truth": MAP_OPEN}],
        lambda_deg=1.0, fmt="json")[0]
    assert out["pr_turn1"] == pytest.approx(1.0)
    assert out["pr_final"] == pytest.approx(0.5)
    assert out["reflection"] == 0.0
    assert out["overall"] == pytest.approx(0.1)


def test_compute_score_reflect_lower_clip_surfaces_break_when_negative():
    # reflect_lower_clip=-0.05 lets the break show up as a negative R_reflect.
    out = fz.compute_score(
        [{"response": _break_transcript(), "ground_truth": MAP_OPEN}],
        lambda_deg=1.0, fmt="json", reflect_lower_clip=-0.05)[0]
    assert out["reflection"] == pytest.approx(-0.05)
    assert out["overall"] == pytest.approx(0.1 + 0.9 * (-0.05))


def test_compute_score_multi_route_partial_uses_reflection():
    # two distinct routes, PR 0.25 -> 0.75, neither EM: R_reflect = sum of deltas
    # only -> w(+0.5) = 0.5 (PR_1 is NOT included).
    r1 = '{"think": "x", "function_call": {"name": "route", "arguments": {"actions": "down"}}}'
    r2 = '{"think": "x", "function_call": {"name": "route", "arguments": {"actions": "down, right, down"}}}'
    tm = '{"think": "x", "function_call": {"name": "terminate", "arguments": {"answer": "down, right, down"}}}'
    resp = r1 + USER_FEEDBACK + r2 + USER_FEEDBACK + tm + "<|im_end|>"
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}], lambda_deg=1.0, fmt="json")
    assert out[0]["em"] == 0.0
    assert abs(out[0]["reflection"] - 0.5) < 1e-9
    assert abs(out[0]["overall"] - (0.1 + 0.9 * 0.5)) < 1e-9


def test_compute_score_garbage_floors_to_zero():
    out = fz.compute_score([{"response": _RESP_GARBAGE, "ground_truth": MAP_OPEN}], lambda_deg=1.0, fmt="json")
    assert out[0]["overall"] == 0.0
    assert out[0]["format"] == 0.0


def test_compute_score_bad_ground_truth_returns_floor():
    out = fz.compute_score([{"response": _RESP_FIX_TO_EM, "ground_truth": "not json"}], lambda_deg=1.0, fmt="json")
    assert out[0]["overall"] == 0.0


def test_compute_score_keys_are_consistent_and_kwargs_tolerant():
    batch = [{"response": _RESP_FIX_TO_EM, "ground_truth": MAP_OPEN},
             {"response": _RESP_GARBAGE, "ground_truth": MAP_OPEN}]
    # extra kwargs from the reward manager (e.g. is_random_init_rollout) must be ignored
    out = fz.compute_score(batch, lambda_deg=1.5, is_random_init_rollout=[False, False], fmt="json")
    assert len(out) == 2
    assert set(out[0].keys()) == set(out[1].keys())
    assert "overall" in out[0]


def test_compute_score_unreachable_goal_returns_floor():
    # layout ["SFH","FHH","HHG"]: goal at index 8 is surrounded by holes (indices 5 and 7),
    # so bfs_dist returns None and _score_one must short-circuit to the floor result.
    unreachable = {"layout": ["SFH", "FHH", "HHG"], "start_pos": 0, "target_pos": 8, "level": 3}
    out = fz.compute_score([{"response": _RESP_FIX_TO_EM, "ground_truth": unreachable}], lambda_deg=1.0, fmt="json")
    assert out[0]["overall"] == 0.0
    assert out[0]["em"] == 0.0


def test_compute_score_correct_value_kwarg_is_used():
    # When the committed answer is EM, overall must equal correct_value, not a hardcoded 1.0.
    out = fz.compute_score([{"response": _RESP_FIX_TO_EM, "ground_truth": MAP_OPEN}],
                           lambda_deg=1.0, correct_value=2.0, fmt="json")
    assert out[0]["em"] == 1.0
    assert out[0]["overall"] == 2.0


def test_compute_score_reflection_excludes_terminate_turn():
    # route "down, right" then terminate "down" -- terminate edits the answer
    # (constraint 1 violation). R_reflect scores route turns ONLY: the route-only
    # pr_list is [0.5] (one route) -> R_reflect == 0.0 (single-route guard).
    # If the terminate turn were wrongly counted, pr_list would be [0.5, 0.25]
    # -> w(-0.25) = -0.25 -> clip = 0.0 also. So we check format to disambiguate:
    # the constraint-1 violation fires, format = 0.0 (regardless of reflection).
    term_down = '{"think": "x", "function_call": {"name": "terminate", "arguments": {"answer": "down"}}}'
    resp = TURN_ROUTE + USER_FEEDBACK + term_down + "<|im_end|>"
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}], lambda_deg=1.0, fmt="json")
    assert out[0]["em"] == 0.0
    assert out[0]["reflection"] == 0.0
    assert out[0]["format"] == 0.0  # constraint 1 violated -> format gate 0


def test_compute_score_off_grid_recovery_earns_em():
    # route 1 leaves the grid; route 2 is optimal and committed (EM). Under
    # the relaxed format gate, an off-grid intermediate route does NOT fail
    # format -- the model is expected to recover via reflection, and here it
    # did. Format = 0.1, em = 1, overall = correct_value = 1.0.
    r_off = '{"think": "x", "function_call": {"name": "route", "arguments": {"actions": "up"}}}'
    r_opt = '{"think": "x", "function_call": {"name": "route", "arguments": {"actions": "right, right, down, down"}}}'
    tm = '{"think": "x", "function_call": {"name": "terminate", "arguments": {"answer": "right, right, down, down"}}}'
    resp = r_off + USER_FEEDBACK + r_opt + USER_FEEDBACK + tm + "<|im_end|>"
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}], lambda_deg=1.0, fmt="json")
    assert out[0]["format"] == 0.1
    assert out[0]["em"] == 1.0
    assert out[0]["overall"] == 1.0


# ---------------------------------------------------------------------------
# Tag-format parser (new SFT format): <think>...</think><route>...</route>
# and <think>...</think><answer>...</answer>. <answer> maps to kind 'terminate'
# so downstream reward logic is unchanged.
# ---------------------------------------------------------------------------

TAG_ROUTE = "<think>go down then right</think><route>down right</route>"
TAG_ANSWER = "<think>confirmed</think><answer>right right down down</answer>"


def test_parse_actions_csv_accepts_space_separator():
    # New SFT data uses space-separated tokens
    assert fz._parse_actions_csv("right right down") == (["right", "right", "down"], True)
    # Mixed comma + whitespace inside a single chunk still works
    assert fz._parse_actions_csv("up , right  down") == (["up", "right", "down"], True)
    # An invalid token in the space-separated form must still flag failure
    assert fz._parse_actions_csv("right sideways") == (None, False)


def test_parse_one_turn_tag_route():
    assert fz._parse_one_turn(TAG_ROUTE) == ("route", ["down", "right"], True)


def test_parse_one_turn_tag_answer_maps_to_terminate():
    # <answer> kind must map to "terminate" so the reward path is unchanged.
    assert fz._parse_one_turn(TAG_ANSWER) == ("terminate", ["right", "right", "down", "down"], True)


def test_parse_one_turn_tag_with_messy_whitespace():
    msg = "<think>\n  multi-line\n  reasoning\n</think>\n<route>  down   right  </route>"
    assert fz._parse_one_turn(msg) == ("route", ["down", "right"], True)


def test_parse_turns_tag_format_two_turn_transcript():
    resp = TAG_ROUTE + USER_FEEDBACK + TAG_ANSWER + "<|im_end|>"
    turns = fz.parse_turns(resp)
    assert [t["kind"] for t in turns] == ["route", "terminate"]
    assert turns[0]["actions"] == ["down", "right"]
    assert turns[1]["actions"] == ["right", "right", "down", "down"]
    assert all(t["ok"] for t in turns)


def test_parse_turns_fmt_does_not_mix_formats():
    # With fmt="tag", JSON turns must NOT be silently accepted, and vice
    # versa. Mixing formats in one transcript is no longer supported -- the
    # explicit fmt flag picks one path and sticks to it.
    resp = TURN_ROUTE + USER_FEEDBACK + TAG_ANSWER + "<|im_end|>"
    # In tag mode: turn 1 (JSON) fails to parse; turn 2 (tag) succeeds.
    turns_tag = fz.parse_turns(resp, fmt="tag")
    assert turns_tag[0]["kind"] is None
    assert turns_tag[1]["kind"] == "terminate"
    # In json mode: turn 1 (JSON) succeeds; turn 2 (tag) fails to parse.
    turns_json = fz.parse_turns(resp, fmt="json")
    assert turns_json[0]["kind"] == "route"
    assert turns_json[1]["kind"] is None


# ---------------------------------------------------------------------------
# reflection-format parser + format reward + end-to-end compute_score
# ---------------------------------------------------------------------------

# Synthetic GT layouts. The 3x3 open map (MAP_OPEN) has optimal n=4; one of the
# optimal sequences is ["right", "right", "down", "down"].
REFL_ANSWER_OPT = "<ANSWER>right right down down</ANSWER>"
REFL_ANSWER_BAD = "<ANSWER>down right</ANSWER>"
REFL_ANSWER_BAD2 = "<ANSWER>down down right right</ANSWER>"
REFL_THINK_ANSWER_OPT = "<THINK>need to go right first</THINK><ANSWER>right right down down</ANSWER>"
REFL_THINK_ANSWER_BAD2 = "<THINK>try the other way</THINK><ANSWER>down down right right</ANSWER>"
REFL_FINAL_OPT = "<FINAL>right right down down</FINAL>"
REFL_FINAL_BAD = "<FINAL>down right</FINAL>"


def _join_turns(*assistant_turns):
    """Build a multi-turn assistant transcript from a sequence of raw turns,
    inserting the user-feedback boundary between successive turns. Matches
    the structure that the rollout produces."""
    if not assistant_turns:
        return ""
    out = assistant_turns[0]
    for t in assistant_turns[1:]:
        out += USER_FEEDBACK + t
    return out + "<|im_end|>"


def test_parse_one_turn_reflection_answer():
    # Bare <ANSWER> = proposal (kind="route").
    assert fz._parse_one_turn(REFL_ANSWER_OPT, fmt="reflection_tag") == (
        "route", ["right", "right", "down", "down"], True)


def test_parse_one_turn_reflection_think_answer():
    # <THINK>...</THINK><ANSWER>X</ANSWER> = proposal; THINK is metadata.
    assert fz._parse_one_turn(REFL_THINK_ANSWER_OPT, fmt="reflection_tag") == (
        "route", ["right", "right", "down", "down"], True)


def test_parse_one_turn_reflection_final():
    # <FINAL> = commit (kind="terminate").
    assert fz._parse_one_turn(REFL_FINAL_OPT, fmt="reflection_tag") == (
        "terminate", ["right", "right", "down", "down"], True)


def test_parse_one_turn_reflection_both_prefers_final():
    # If both FINAL and ANSWER appear in the same turn (malformed), prefer
    # FINAL -- the model has committed.
    msg = "<ANSWER>down right</ANSWER><FINAL>right right down down</FINAL>"
    kind, actions, ok = fz._parse_one_turn(msg, fmt="reflection_tag")
    assert kind == "terminate"
    assert actions == ["right", "right", "down", "down"]
    assert ok


def test_parse_one_turn_reflection_no_tag():
    assert fz._parse_one_turn("just plain text", fmt="reflection_tag") == (
        None, None, False)


def test_parse_one_turn_reflection_unknown_fmt_still_raises():
    import pytest as _pt
    with _pt.raises(ValueError, match="unknown fmt"):
        fz._parse_one_turn(REFL_ANSWER_OPT, fmt="xml")


def test_format_score_reflection_valid_two_turn():
    # (1) Valid 2-turn no-revision: <ANSWER>X</ANSWER> -> <FINAL>X</FINAL>.
    resp = _join_turns(REFL_ANSWER_OPT, REFL_FINAL_OPT)
    turns = fz.parse_turns(resp, fmt="reflection_tag")
    assert fz.compute_format_score(turns, ["right", "right", "down", "down"],
                                   MAP_OPEN["layout"], 0, 3,
                                   fmt="reflection_tag") == 0.1


def test_format_score_reflection_valid_three_turn():
    # (2) Valid 3-turn 1-revision: <ANSWER> -> <THINK><ANSWER> -> <FINAL>.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    turns = fz.parse_turns(resp, fmt="reflection_tag")
    assert fz.compute_format_score(turns, ["right", "right", "down", "down"],
                                   MAP_OPEN["layout"], 0, 3,
                                   fmt="reflection_tag") == 0.1


def test_format_score_reflection_valid_four_turn():
    # (3) Valid 4-turn 2-revision: <ANSWER> -> <THINK><ANSWER> -> <THINK><ANSWER> -> <FINAL>.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_BAD2,
                       REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    turns = fz.parse_turns(resp, fmt="reflection_tag")
    assert fz.compute_format_score(turns, ["right", "right", "down", "down"],
                                   MAP_OPEN["layout"], 0, 3,
                                   fmt="reflection_tag") == 0.1


def test_format_score_reflection_invalid_revision_missing_think():
    # (4) Revision turn (turn-2 of 3) is a bare <ANSWER> -- missing <THINK>.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_ANSWER_OPT, REFL_FINAL_OPT)
    turns = fz.parse_turns(resp, fmt="reflection_tag")
    assert fz.compute_format_score(turns, ["right", "right", "down", "down"],
                                   MAP_OPEN["layout"], 0, 3,
                                   fmt="reflection_tag") == 0.0


def test_format_score_reflection_invalid_final_has_think():
    # (5) <FINAL> preceded by <THINK> in the same turn -> 0.
    final_with_think = "<THINK>checked it</THINK><FINAL>right right down down</FINAL>"
    resp = _join_turns(REFL_ANSWER_OPT, final_with_think)
    turns = fz.parse_turns(resp, fmt="reflection_tag")
    assert fz.compute_format_score(turns, ["right", "right", "down", "down"],
                                   MAP_OPEN["layout"], 0, 3,
                                   fmt="reflection_tag") == 0.0


def test_format_score_reflection_invalid_turn1_final():
    # (6) Turn-1 is <FINAL> (commit before any proposal) -> 0.
    resp = _join_turns(REFL_FINAL_OPT)
    turns = fz.parse_turns(resp, fmt="reflection_tag")
    assert fz.compute_format_score(turns, ["right", "right", "down", "down"],
                                   MAP_OPEN["layout"], 0, 3,
                                   fmt="reflection_tag") == 0.0


def test_format_score_reflection_invalid_final_neq_last_answer():
    # (7) <FINAL> action sequence != immediately preceding <ANSWER> -> 0.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_FINAL_OPT)
    turns = fz.parse_turns(resp, fmt="reflection_tag")
    assert fz.compute_format_score(turns, ["right", "right", "down", "down"],
                                   MAP_OPEN["layout"], 0, 3,
                                   fmt="reflection_tag") == 0.0


def test_format_score_reflection_invalid_duplicate_answers():
    # (8) Two identical <ANSWER> proposals -> 0.
    dup_with_think = "<THINK>still believe</THINK>" + REFL_ANSWER_BAD
    resp = _join_turns(REFL_ANSWER_BAD, dup_with_think, REFL_FINAL_BAD)
    turns = fz.parse_turns(resp, fmt="reflection_tag")
    assert fz.compute_format_score(turns, ["down", "right"],
                                   MAP_OPEN["layout"], 0, 3,
                                   fmt="reflection_tag") == 0.0


def test_format_score_reflection_invalid_unparseable_turn():
    # (9) Unparseable turn (no recognized tag) -> 0.
    resp = _join_turns(REFL_ANSWER_OPT, "no tags here at all", REFL_FINAL_OPT)
    turns = fz.parse_turns(resp, fmt="reflection_tag")
    # Note: ANY non-parsed turn means compute_format_score returns 0.0
    # because turns include a kind=None entry.
    assert fz.compute_format_score(turns, ["right", "right", "down", "down"],
                                   MAP_OPEN["layout"], 0, 3,
                                   fmt="reflection_tag") == 0.0


def test_compute_score_reflection_em_gets_one():
    # (10) Valid 3-turn trace + BFS-optimal committed answer -> overall = 1.0.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           lambda_deg=1.0, fmt="reflection_tag")
    assert out[0]["em"] == 1.0
    assert out[0]["format"] == 0.1
    assert out[0]["overall"] == 1.0


def test_compute_score_reflection_valid_structure_wrong_answer():
    # (11) Valid trace structure but committed answer is wrong (not EM).
    # overall = R_format + reflect_weight * R_reflect < 1.0.
    # On MAP_OPEN (3x3 open grid, n_opt=4):
    # Trace:
    #   <ANSWER>down right</ANSWER>          (PR 0.5; reaches (1,1) -- 2 of 4)
    #   <THINK><ANSWER>down right down</ANSWER> (PR 0.75; reaches (2,1) -- 3 of 4)
    #   <FINAL>down right down</FINAL>        (committed 3-step trajectory != optimal 4-step)
    # PR_final = 0.75 (3 of 4); but len != n_opt -> NOT EM. R_format = 0.1.
    # reflect = w(+0.25) = 0.25.
    # overall = 0.1 + 0.9*0.25 = 0.325.
    ans1 = "<ANSWER>down right</ANSWER>"
    ans2 = "<THINK>extend by one</THINK><ANSWER>down right down</ANSWER>"
    final2 = "<FINAL>down right down</FINAL>"
    resp = _join_turns(ans1, ans2, final2)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           lambda_deg=1.0, fmt="reflection_tag")
    assert out[0]["em"] == 0.0
    assert out[0]["format"] == 0.1
    assert out[0]["overall"] < 1.0
    # Sanity: overall is R_format + reflect_weight * R_reflect.
    expected = 0.1 + 0.9 * out[0]["reflection"]
    assert abs(out[0]["overall"] - expected) < 1e-9


def test_compute_score_reflection_format_violation_zeros_overall():
    # (12) Format violation (missing <THINK> on revision) -> overall = 0
    # regardless of EM (the committed answer is optimal here).
    resp = _join_turns(REFL_ANSWER_BAD, REFL_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           lambda_deg=1.0, fmt="reflection_tag")
    assert out[0]["format"] == 0.0
    assert out[0]["overall"] == 0.0


def test_parse_turns_reflection_threads_raw_text():
    # parse_turns must include `raw_text` per turn so compute_format_score
    # can inspect <THINK> presence without re-splitting the transcript.
    resp = _join_turns(REFL_ANSWER_OPT, REFL_FINAL_OPT)
    turns = fz.parse_turns(resp, fmt="reflection_tag")
    assert all("raw_text" in t for t in turns)
    assert "<ANSWER>" in turns[0]["raw_text"]
    assert "<FINAL>" in turns[1]["raw_text"]


# ---------------------------------------------------------------------------
# Progress-rate (PR) outcome reward -- alternative to the EM outcome reward.
# Selected by the compute_score kwarg outcome_reward in {"em" (default), "pr"}.
# PR mode replaces the binary EM jackpot for NON-EM rollouts with a graded
#   outcome_weight * PR_final  term, clamped to correct_value.
# ---------------------------------------------------------------------------


def _refl_two_turn(traj):
    """A valid 2-turn no-revision reflection transcript committing `traj`:
    <ANSWER>traj</ANSWER> -> <FINAL>traj</FINAL>. Single route turn, so
    R_reflect == 0 and the reward is driven entirely by the outcome term."""
    return _join_turns("<ANSWER>%s</ANSWER>" % traj, "<FINAL>%s</FINAL>" % traj)


def test_outcome_pr_em_rollout_still_gets_correct_value():
    # EM is the ceiling in PR mode too: a paper-EM committed answer earns
    # exactly correct_value regardless of outcome_reward.
    resp = _refl_two_turn("right right down down")  # optimal on MAP_OPEN
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", outcome_reward="pr",
                           reflect_weight=0.0, outcome_weight=0.9)[0]
    assert out["em"] == 1.0
    assert out["overall"] == 1.0
    assert out["outcome"] == 1.0


def test_outcome_pr_grades_non_em_rollout():
    # outcome-only weighting (reflect_weight=0). A non-EM committed answer with
    # PR_final=0.5 earns 0.1 + 0.9*0.5 = 0.55 under PR, but only the flat 0.1
    # format floor under EM.
    resp = _refl_two_turn("down right")  # PR 0.5 on MAP_OPEN, len 2 != 4 -> not EM
    pr = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                          fmt="reflection_tag", outcome_reward="pr",
                          reflect_weight=0.0, outcome_weight=0.9)[0]
    em = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                          fmt="reflection_tag", outcome_reward="em",
                          reflect_weight=0.0)[0]
    assert pr["em"] == 0.0 and em["em"] == 0.0
    assert abs(pr["pr_final"] - 0.5) < 1e-9
    assert abs(em["overall"] - 0.1) < 1e-9                  # flat format floor
    assert abs(pr["overall"] - (0.1 + 0.9 * 0.5)) < 1e-9    # graded by PR
    assert abs(pr["outcome"] - 0.9 * 0.5) < 1e-9
    assert pr["overall"] > em["overall"]


def test_outcome_pr_zero_progress_matches_em_floor():
    # A committed answer that makes no progress (off-grid first step, PR=0)
    # earns the same 0.1 format floor under both outcome rewards.
    resp = _refl_two_turn("up")  # 'up' from (0,0) leaves the grid -> PR 0.0
    pr = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                          fmt="reflection_tag", outcome_reward="pr",
                          reflect_weight=0.0, outcome_weight=0.9)[0]
    assert pr["pr_final"] == 0.0
    assert abs(pr["overall"] - 0.1) < 1e-9


def test_outcome_pr_is_monotone_in_progress():
    # Strictly increasing PR_final -> strictly increasing PR-mode reward, so a
    # group of non-EM rollouts gets a usable (non-degenerate) gradient.
    trajs = ["up", "right", "down right", "down right down"]  # PR 0, .25, .5, .75
    overalls = []
    for t in trajs:
        out = fz.compute_score([{"response": _refl_two_turn(t),
                                 "ground_truth": MAP_OPEN}],
                               fmt="reflection_tag", outcome_reward="pr",
                               reflect_weight=0.0, outcome_weight=0.9)[0]
        overalls.append(out["overall"])
    assert overalls == sorted(overalls)
    assert len(set(overalls)) == 4  # all distinct


def test_outcome_pr_format_failure_still_zeroes():
    # The format gate overrides the PR outcome: a structural violation
    # (missing <THINK> on a revision turn) zeroes the reward in PR mode too.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", outcome_reward="pr",
                           outcome_weight=0.9)[0]
    assert out["format"] == 0.0
    assert out["overall"] == 0.0


def test_outcome_pr_reflection_aware_adds_pr_to_reflection():
    # reflection-aware weighting (reflect_weight=0.9). A 3-turn trace that
    # improves PR 0.5 -> 0.75 and commits the 0.75 trajectory:
    #   R_reflect = w(+0.25) = 0.25 ; PR_final = 0.75 ; not EM.
    # EM mode:  overall = 0.1 + 0.9*0.25                       = 0.325
    # PR mode:  overall = min(1.0, 0.1 + 0.9*0.25 + 0.5*0.75)  = 0.70
    ans1 = "<ANSWER>down right</ANSWER>"
    ans2 = "<THINK>extend by one</THINK><ANSWER>down right down</ANSWER>"
    final = "<FINAL>down right down</FINAL>"
    resp = _join_turns(ans1, ans2, final)
    em = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                          fmt="reflection_tag", outcome_reward="em",
                          reflect_weight=0.9)[0]
    pr = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                          fmt="reflection_tag", outcome_reward="pr",
                          reflect_weight=0.9, outcome_weight=0.5)[0]
    assert abs(em["overall"] - 0.325) < 1e-9
    assert abs(pr["reflection"] - 0.25) < 1e-9
    assert abs(pr["pr_final"] - 0.75) < 1e-9
    assert abs(pr["overall"] - 0.70) < 1e-9
    assert pr["overall"] > em["overall"]


def test_outcome_pr_clamps_to_correct_value():
    # PR mode never exceeds correct_value for a non-EM rollout: with big
    # weights the raw sum 0.1 + 0.9*0.25 + 0.9*0.75 = 1.0 clamps at the ceiling.
    ans1 = "<ANSWER>down right</ANSWER>"
    ans2 = "<THINK>extend</THINK><ANSWER>down right down</ANSWER>"
    final = "<FINAL>down right down</FINAL>"
    resp = _join_turns(ans1, ans2, final)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", outcome_reward="pr",
                           reflect_weight=0.9, outcome_weight=0.9)[0]
    assert out["em"] == 0.0
    assert out["overall"] <= 1.0
    assert abs(out["overall"] - 1.0) < 1e-9


def test_outcome_reward_unknown_raises():
    import pytest as _pt
    with _pt.raises(ValueError, match="outcome_reward"):
        fz.compute_score([{"response": _refl_two_turn("down right"),
                           "ground_truth": MAP_OPEN}],
                          fmt="reflection_tag", outcome_reward="bogus")


def test_outcome_em_mode_is_default_and_unchanged():
    # Omitting outcome_reward is identical to passing "em" -- the default path
    # is byte-for-byte the legacy EM behavior.
    resp = _refl_two_turn("down right")
    default = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                               fmt="reflection_tag", reflect_weight=0.0)[0]
    explicit = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                                fmt="reflection_tag", reflect_weight=0.0,
                                outcome_reward="em")[0]
    assert default == explicit
    assert abs(default["overall"] - 0.1) < 1e-9
    assert default["outcome"] == 0.0


def test_outcome_pr_revives_dead_group():
    # The research claim, as a test. A GRPO "group" of 8 non-EM rollouts that
    # made varied progress: under the EM outcome reward they all collapse to
    # the same 0.1 floor (zero within-group variance -> zero advantage -> a
    # DEAD group, no gradient). Under the PR outcome reward they spread across
    # distinct reward tiers, restoring a usable gradient.
    trajs = ["up", "up", "right", "right",
             "down right", "down right", "down right down", "down right down"]
    em_scores, pr_scores = [], []
    for t in trajs:
        ri = {"response": _refl_two_turn(t), "ground_truth": MAP_OPEN}
        em_scores.append(fz.compute_score([ri], fmt="reflection_tag",
                                          outcome_reward="em",
                                          reflect_weight=0.0)[0]["overall"])
        pr_scores.append(fz.compute_score([ri], fmt="reflection_tag",
                                          outcome_reward="pr", reflect_weight=0.0,
                                          outcome_weight=0.9)[0]["overall"])
    assert len(set(em_scores)) == 1   # EM outcome: identical floor -> dead group
    assert len(set(pr_scores)) > 1    # PR outcome: distinct tiers -> live gradient


# ---------------------------------------------------------------------------
# pr_overshoot_zero -- piecewise progress rate that zeroes overshoot
# trajectories (the optimal prefix is reached, but len > n_opt). Closes the
# PR=1.0 overshoot reward hack. Threaded compute_score -> _score_one ->
# progress_rate(overshoot_zero=...).
# ---------------------------------------------------------------------------

# On MAP_OPEN (3x3 open, n_opt=4): the optimal 4-step route, then one extra step.
_OVERSHOOT = ["right", "right", "down", "down", "up"]            # extra step on-grid
_OVERSHOOT_OFFGRID = ["right", "right", "down", "down", "right"]  # extra step off-grid


def test_progress_rate_overshoot_zero_flag():
    lay = MAP_OPEN["layout"]
    # flag off (default): an overshoot still scores 1.0 -- the legacy behavior
    assert fz.progress_rate(_OVERSHOOT, lay, 0, 8, 3, 4) == 1.0
    # flag on: matched == n_opt AND len > n_opt -> zeroed
    assert fz.progress_rate(_OVERSHOOT, lay, 0, 8, 3, 4, overshoot_zero=True) == 0.0


def test_progress_rate_overshoot_zero_offgrid_tail():
    # an overshoot whose extra step leaves the grid is still an overshoot
    lay = MAP_OPEN["layout"]
    assert fz.progress_rate(_OVERSHOOT_OFFGRID, lay, 0, 8, 3, 4) == 1.0
    assert fz.progress_rate(_OVERSHOOT_OFFGRID, lay, 0, 8, 3, 4,
                            overshoot_zero=True) == 0.0


def test_progress_rate_overshoot_zero_leaves_em_and_partial_unchanged():
    lay = MAP_OPEN["layout"]
    # exact-optimal (len == n_opt) is NOT an overshoot -> stays 1.0
    assert fz.progress_rate(["right", "right", "down", "down"], lay, 0, 8, 3, 4,
                            overshoot_zero=True) == 1.0
    # partial (len < n_opt) -> unchanged
    assert fz.progress_rate(["down", "right"], lay, 0, 8, 3, 4,
                            overshoot_zero=True) == 0.5
    # zero progress -> unchanged
    assert fz.progress_rate(["up"], lay, 0, 8, 3, 4, overshoot_zero=True) == 0.0


def test_progress_rate_overshoot_zero_defaults_off():
    lay = MAP_OPEN["layout"]
    assert (fz.progress_rate(_OVERSHOOT, lay, 0, 8, 3, 4)
            == fz.progress_rate(_OVERSHOOT, lay, 0, 8, 3, 4, overshoot_zero=False))


def test_compute_score_pr_overshoot_zero_zeroes_outcome():
    # A committed overshoot answer, PR outcome mode, outcome-only weighting.
    #   flag off: pr_final = 1.0 -> outcome 1.0 -> overall clamps to 1.0
    #   flag on : pr_final = 0.0 -> outcome 0.0 -> overall = 0.1 (format floor)
    resp = _refl_two_turn("right right down down up")
    off = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", outcome_reward="pr",
                           reflect_weight=0.0, outcome_weight=1.0)[0]
    on = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                          fmt="reflection_tag", outcome_reward="pr",
                          reflect_weight=0.0, outcome_weight=1.0,
                          pr_overshoot_zero=True)[0]
    assert off["em"] == 0.0 and on["em"] == 0.0
    assert off["pr_final"] == 1.0
    assert abs(off["overall"] - 1.0) < 1e-9
    assert on["pr_final"] == 0.0
    assert on["outcome"] == 0.0
    assert abs(on["overall"] - 0.1) < 1e-9


def test_compute_score_pr_overshoot_zero_defaults_off():
    resp = _refl_two_turn("right right down down up")
    default = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                               fmt="reflection_tag", outcome_reward="pr",
                               reflect_weight=0.0)[0]
    explicit = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                                fmt="reflection_tag", outcome_reward="pr",
                                reflect_weight=0.0, pr_overshoot_zero=False)[0]
    assert default == explicit
    assert default["pr_final"] == 1.0      # legacy: an overshoot scores 1.0


def test_compute_score_pr_overshoot_zero_reflection_rewards_overshoot_fix():
    # Turn 1 proposes an overshoot (piecewise PR 0); the revision cuts the junk
    # steps down to a clean partial route (PR 0.5). reflection sees 0 -> 0.5.
    ans1 = "<ANSWER>right right down down up</ANSWER>"
    ans2 = "<THINK>cut the extra steps</THINK><ANSWER>down right</ANSWER>"
    final = "<FINAL>down right</FINAL>"
    resp = _join_turns(ans1, ans2, final)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", outcome_reward="pr",
                           reflect_weight=0.9, outcome_weight=1.0,
                           lambda_deg=0.5, pr_overshoot_zero=True)[0]
    assert out["em"] == 0.0
    assert abs(out["pr_turn1"] - 0.0) < 1e-9      # turn-1 overshoot zeroed
    assert abs(out["pr_final"] - 0.5) < 1e-9
    assert abs(out["reflection"] - 0.5) < 1e-9    # w(+0.5) = 0.5


# ---------------------------------------------------------------------------
# Tests for em_reflect_bonus_weight
#
# When EM=1 and `em_reflect_bonus_weight > 0`, the reward gets an extra
# discounted reflection bonus on top of `correct_value`:
#   overall = correct_value + em_reflect_bonus_weight * R_reflect
#                                          * em_reflect_step_discount ** n_revisions
# where n_revisions = max(0, len(pr_list) - 1). The default
# em_reflect_bonus_weight=0 preserves legacy behavior (EM=1 -> flat
# correct_value).
# ---------------------------------------------------------------------------

def test_em_reflect_bonus_default_off_preserves_legacy():
    # Default kwargs: EM=1 -> flat correct_value=1.0, no extra bonus even
    # when a revision happened.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", reflect_weight=0.9)[0]
    assert out["em"] == 1.0
    assert out["overall"] == 1.0    # legacy: EM=1 jackpot is the ceiling


def test_em_reflect_bonus_direct_correct_no_bonus():
    # Bonus enabled, but the model proposes optimal in turn 1 and FINALs it
    # in turn 2 (no revision attempted): pr_list has one entry, R_reflect=0,
    # bonus = 0 -> overall = correct_value.
    resp = _join_turns(REFL_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag",
                           em_reflect_bonus_weight=0.15,
                           em_reflect_step_discount=0.7)[0]
    assert out["em"] == 1.0
    assert out["reflection"] == 0.0          # single route turn -> no delta sum
    assert abs(out["overall"] - 1.0) < 1e-9  # no bonus added


def test_em_reflect_bonus_one_revision_pays_discounted_bonus():
    # 1 revision (ANSWER_bad -> THINK+ANSWER_opt -> FINAL_opt):
    # pr_list = [0.5, 1.0] (route turns only; FINAL not in pr_list).
    # n_revisions = max(0, len(pr_list) - 1) = 1.
    # With bonus weight 0.20 and discount 0.7^1 over reflect=0.5:
    #   overall = 1.0 + 0.20 * 0.5 * 0.7 = 1.07
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag",
                           em_reflect_bonus_weight=0.20,
                           em_reflect_step_discount=0.7)[0]
    assert out["em"] == 1.0
    assert abs(out["reflection"] - 0.5) < 1e-9
    expected = 1.0 + 0.20 * 0.5 * (0.7 ** 1)
    assert abs(out["overall"] - expected) < 1e-6


def test_em_reflect_bonus_more_revisions_pay_less():
    # 1-revision path (2 route turns, reflect=0.5, n_rev=1, bonus=0.07)
    #   vs
    # 2-revision path (3 route turns: bad -> bad2[opt] -> opt -> FINAL opt;
    #   pr_list = [0.5, 1.0, 1.0], reflect = 0.5 + 0 = 0.5, n_rev = 2,
    #   bonus = 0.20 * 0.5 * 0.7^2 = 0.049)
    # Both have the same reflection score but the 2-rev path receives a
    # smaller bonus -- the anti-gaming property.
    one_rev = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    two_rev = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_BAD2,
                          REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    one = fz.compute_score([{"response": one_rev, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag",
                           em_reflect_bonus_weight=0.20,
                           em_reflect_step_discount=0.7)[0]
    two = fz.compute_score([{"response": two_rev, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag",
                           em_reflect_bonus_weight=0.20,
                           em_reflect_step_discount=0.7)[0]
    assert one["em"] == 1.0 and two["em"] == 1.0
    assert abs(one["reflection"] - 0.5) < 1e-9
    assert abs(two["reflection"] - 0.5) < 1e-9
    assert abs(one["overall"] - 1.07) < 1e-6
    assert abs(two["overall"] - 1.049) < 1e-6
    assert one["overall"] > two["overall"]


def test_em_reflect_bonus_em_zero_no_bonus_path():
    # When EM=0 we never enter the EM-jackpot branch; the reflection bonus
    # is the LEGACY reflect_weight * R_reflect (already capped), unchanged
    # by em_reflect_bonus_weight. Confirms the new kwarg only modifies the
    # EM=1 branch.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_FINAL_BAD)
    out_legacy = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                                  fmt="reflection_tag", reflect_weight=0.9)[0]
    out_with_bonus = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                                      fmt="reflection_tag", reflect_weight=0.9,
                                      em_reflect_bonus_weight=0.5,
                                      em_reflect_step_discount=0.5)[0]
    assert out_legacy["em"] == 0.0
    assert out_with_bonus["em"] == 0.0
    # The new kwarg should NOT affect the em=0 path -> overall identical.
    assert abs(out_legacy["overall"] - out_with_bonus["overall"]) < 1e-9


# ---------------------------------------------------------------------------
# Tests for prefix_buffer_decision_bonus
#
# Adds an additive bonus / penalty on the FIRST new turn after a buffered
# prefix. Requires the per-sample fields `is_prefix_buffer_rollout=True`,
# `prefix_buffer_type` in {"wrong","right"}, and `prefix_buffer_num_turns>0`.
#
# Decision matrix:
#   wrong: revise+correct -> +B   |   FINAL immediately -> -B
#   right: FINAL+correct  -> +B   |   revise            -> -B
# Skipped silently when not a buffer rollout, when format fails, or when
# the per-sample metadata is missing.
# ---------------------------------------------------------------------------

def test_decision_bonus_default_off_preserves_legacy():
    # With no decision_bonus kwarg, all metadata in place but the feature
    # is disabled: overall must match the legacy formula exactly and
    # `decision_bonus` metric must be 0.0.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{
        "response": resp,
        "ground_truth": MAP_OPEN,
        "is_prefix_buffer_rollout": True,
        "prefix_buffer_type": "wrong",
        "prefix_buffer_num_turns": 1,
    }], fmt="reflection_tag", reflect_weight=0.9)[0]
    assert out["em"] == 1.0
    assert out["overall"] == 1.0
    assert out["decision_bonus"] == 0.0


def test_decision_bonus_wrong_prefix_revise_succeed_plus_bonus():
    # wrong prefix (turn 0 ANSWER_bad), model revises in turn 1 to optimal,
    # then FINAL optimal -> em=1. First "new" turn after prefix is a route ->
    # revise+em=1 -> +bonus. overall = 1.0 + 0.3 = 1.3.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{
        "response": resp,
        "ground_truth": MAP_OPEN,
        "is_prefix_buffer_rollout": True,
        "prefix_buffer_type": "wrong",
        "prefix_buffer_num_turns": 1,
    }], fmt="reflection_tag", reflect_weight=0.9,
       prefix_buffer_decision_bonus=0.3)[0]
    assert out["em"] == 1.0
    assert abs(out["decision_bonus"] - 0.3) < 1e-9
    assert abs(out["overall"] - 1.3) < 1e-9


def test_decision_bonus_wrong_prefix_final_immediately_minus_bonus():
    # wrong prefix (turn 0 ANSWER_bad), model FINALs the bad answer right
    # away (= stuck with wrong) -> -bonus. em=0 so legacy overall = fmt
    # + reflect*0; decision_bonus subtracts on top.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_FINAL_BAD)
    out = fz.compute_score([{
        "response": resp,
        "ground_truth": MAP_OPEN,
        "is_prefix_buffer_rollout": True,
        "prefix_buffer_type": "wrong",
        "prefix_buffer_num_turns": 1,
    }], fmt="reflection_tag", reflect_weight=0.9,
       prefix_buffer_decision_bonus=0.3)[0]
    assert out["em"] == 0.0
    assert abs(out["decision_bonus"] + 0.3) < 1e-9
    # legacy overall on this transcript (em=0, reflect=0 since pr_list=[0.5]
    # has no delta) = fmt_score + 0 = 1.0; with -0.3 bonus -> 0.7.
    expected_legacy = out["format"]
    assert abs(out["overall"] - (expected_legacy - 0.3)) < 1e-9


def test_decision_bonus_right_prefix_final_correctly_plus_bonus():
    # right prefix (turn 0 ANSWER_opt), model FINALs the same optimal
    # answer -> em=1, first new turn is terminate -> +bonus.
    resp = _join_turns(REFL_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{
        "response": resp,
        "ground_truth": MAP_OPEN,
        "is_prefix_buffer_rollout": True,
        "prefix_buffer_type": "right",
        "prefix_buffer_num_turns": 1,
    }], fmt="reflection_tag", reflect_weight=0.9,
       prefix_buffer_decision_bonus=0.3)[0]
    assert out["em"] == 1.0
    assert abs(out["decision_bonus"] - 0.3) < 1e-9
    assert abs(out["overall"] - 1.3) < 1e-9


def test_decision_bonus_right_prefix_revise_unnecessarily_minus_bonus():
    # right prefix (turn 0 ANSWER_opt), model unnecessarily revises (to a
    # different but also-optimal path "down down right right") and FINALs
    # it. em stays 1.0 (both paths are EM-optimal on MAP_OPEN), but the
    # DECISION to revise a correct prefix was wrong -> -bonus. We need to
    # craft a custom FINAL that matches the preceding route's actions in
    # order for the format gate to pass.
    REFL_FINAL_BAD2 = "<FINAL>down down right right</FINAL>"
    resp = _join_turns(REFL_ANSWER_OPT, REFL_THINK_ANSWER_BAD2, REFL_FINAL_BAD2)
    out = fz.compute_score([{
        "response": resp,
        "ground_truth": MAP_OPEN,
        "is_prefix_buffer_rollout": True,
        "prefix_buffer_type": "right",
        "prefix_buffer_num_turns": 1,
    }], fmt="reflection_tag", reflect_weight=0.9,
       prefix_buffer_decision_bonus=0.3)[0]
    assert out["em"] == 1.0
    assert abs(out["decision_bonus"] + 0.3) < 1e-9
    assert abs(out["overall"] - (1.0 - 0.3)) < 1e-9


def test_decision_bonus_skipped_when_not_buffer_rollout():
    # `is_prefix_buffer_rollout=False` -> bonus must NOT apply even when
    # the type/num_turns are populated.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{
        "response": resp,
        "ground_truth": MAP_OPEN,
        "is_prefix_buffer_rollout": False,
        "prefix_buffer_type": "wrong",
        "prefix_buffer_num_turns": 1,
    }], fmt="reflection_tag", reflect_weight=0.9,
       prefix_buffer_decision_bonus=0.3)[0]
    assert out["decision_bonus"] == 0.0


def test_decision_bonus_skipped_when_format_fails():
    # Malformed response -> _floor_result(); decision_bonus must remain 0.
    out = fz.compute_score([{
        "response": "no tags whatsoever",
        "ground_truth": MAP_OPEN,
        "is_prefix_buffer_rollout": True,
        "prefix_buffer_type": "wrong",
        "prefix_buffer_num_turns": 1,
    }], fmt="reflection_tag", reflect_weight=0.9,
       prefix_buffer_decision_bonus=0.3)[0]
    assert out["format"] == 0.0
    assert out["decision_bonus"] == 0.0


def test_decision_bonus_missing_buffer_metadata_is_noop():
    # `is_prefix_buffer_rollout=True` but the optional metadata fields are
    # absent (defensive path: should not crash; should not apply bonus).
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{
        "response": resp,
        "ground_truth": MAP_OPEN,
        "is_prefix_buffer_rollout": True,
        # no prefix_buffer_type, no prefix_buffer_num_turns
    }], fmt="reflection_tag", reflect_weight=0.9,
       prefix_buffer_decision_bonus=0.3)[0]
    assert out["decision_bonus"] == 0.0


# ---------------------------------------------------------------------------
# Tests for step_cost
#
# When the committed answer is EM and step_cost > 0, subtract a per-revision-
# turn penalty from `overall` to discourage unnecessary extra turns. Two modes:
#   step_cost_churn_only=False (default): penalty = step_cost * max(n_route-1, 0)
#                                         (taxes EVERY revision turn)
#   step_cost_churn_only=True : penalty = step_cost * (route turns AFTER the
#                                         first one that already reached PR==1.0)
# Default step_cost=0.0 is a no-op (legacy behavior preserved). The penalty
# only applies on the EM=1 branch; em=0 rollouts are untouched.
# ---------------------------------------------------------------------------

def test_step_cost_default_off_preserves_legacy():
    # Default kwargs: a 1-revision EM rollout scores the flat jackpot 1.0 and
    # the step_cost metric is 0.0.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", reflect_weight=0.9)[0]
    assert out["em"] == 1.0
    assert out["overall"] == 1.0
    assert out["step_cost"] == 0.0


def test_step_cost_direct_correct_no_penalty():
    # Direct-correct (1 route turn, n_revisions=0): no penalty even with
    # step_cost > 0.
    resp = _join_turns(REFL_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", step_cost=0.05)[0]
    assert out["em"] == 1.0
    assert out["step_cost"] == 0.0
    assert abs(out["overall"] - 1.0) < 1e-9


def test_step_cost_one_revision_taxed_flat():
    # 1 revision (pr_list=[0.5, 1.0], n_route=2): churn_only=False taxes
    # max(2-1,0)=1 revision turn -> penalty = 0.05. overall = 1.0 - 0.05.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", step_cost=0.05)[0]
    assert out["em"] == 1.0
    assert abs(out["step_cost"] - 0.05) < 1e-9
    assert abs(out["overall"] - 0.95) < 1e-9


def test_step_cost_more_revisions_taxed_more():
    # 2-revision EM path (pr_list=[0.5,1.0,1.0], n_route=3) pays 2*step_cost
    # vs the 1-revision path's 1*step_cost -- flat per-turn tax.
    one_rev = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    two_rev = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_BAD2,
                          REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    one = fz.compute_score([{"response": one_rev, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", step_cost=0.05)[0]
    two = fz.compute_score([{"response": two_rev, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", step_cost=0.05)[0]
    assert one["em"] == 1.0 and two["em"] == 1.0
    assert abs(one["overall"] - 0.95) < 1e-9
    assert abs(two["overall"] - 0.90) < 1e-9
    assert two["overall"] < one["overall"]


def test_step_cost_churn_only_spares_productive_revision():
    # churn_only=True: a revision that REACHES the optimal route (pr_list=
    # [0.5, 1.0]) has zero post-correct churn -> NO penalty. The flat mode
    # would tax it; churn-only spares it.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    churn = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                             fmt="reflection_tag", step_cost=0.05,
                             step_cost_churn_only=True)[0]
    flat = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                            fmt="reflection_tag", step_cost=0.05)[0]
    assert churn["em"] == 1.0
    assert churn["step_cost"] == 0.0
    assert abs(churn["overall"] - 1.0) < 1e-9
    assert abs(flat["step_cost"] - 0.05) < 1e-9   # flat mode DID tax it


def test_step_cost_churn_only_taxes_post_correct_churn():
    # churn_only=True: pr_list=[0.5, 1.0, 1.0] -> one route turn AFTER the
    # first correct one -> churn=1 -> penalty = step_cost.
    two_rev = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_BAD2,
                          REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{"response": two_rev, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag", step_cost=0.05,
                           step_cost_churn_only=True)[0]
    # NOTE: pr_list here is [0.5, 1.0, 1.0]; first PR==1.0 at idx 1, one route
    # turn after it -> churn 1.
    assert out["em"] == 1.0
    assert abs(out["step_cost"] - 0.05) < 1e-9
    assert abs(out["overall"] - 0.95) < 1e-9


def test_step_cost_em_zero_untouched():
    # A non-EM rollout never enters the EM branch -> step_cost is a no-op.
    resp = _join_turns(REFL_ANSWER_BAD, REFL_FINAL_BAD)
    legacy = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                              fmt="reflection_tag", reflect_weight=0.9)[0]
    with_cost = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                                 fmt="reflection_tag", reflect_weight=0.9,
                                 step_cost=0.5)[0]
    assert legacy["em"] == 0.0 and with_cost["em"] == 0.0
    assert abs(legacy["overall"] - with_cost["overall"]) < 1e-9
    assert with_cost["step_cost"] == 0.0


def test_step_cost_stacks_with_em_reflect_bonus():
    # Arm 1/3 config: bonus on top of jackpot, then step_cost subtracted.
    # 1 revision, reflect=0.5: overall = 1.0 + 0.2*0.5*1.0(discount off) - 0.05
    #                                  = 1.0 + 0.10 - 0.05 = 1.05
    resp = _join_turns(REFL_ANSWER_BAD, REFL_THINK_ANSWER_OPT, REFL_FINAL_OPT)
    out = fz.compute_score([{"response": resp, "ground_truth": MAP_OPEN}],
                           fmt="reflection_tag",
                           em_reflect_bonus_weight=0.2,
                           em_reflect_step_discount=1.0,
                           step_cost=0.05)[0]
    assert out["em"] == 1.0
    assert abs(out["reflection"] - 0.5) < 1e-9
    assert abs(out["step_cost"] - 0.05) < 1e-9
    assert abs(out["overall"] - 1.05) < 1e-9
