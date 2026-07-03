import prefix_buffer_metrics as pbm


def test_splits_by_prefix_type_and_excludes_non_pb():
    # 4 rollouts: idx0 pb-wrong, idx1 pb-right, idx2 non-pb, idx3 pb-wrong
    is_pb = [True, True, False, True]
    pb_types = ["wrong", "right", None, "wrong"]
    reward_metrics = {
        "reflection": [0.2, 0.8, 0.5, 0.4],
        "outcome":    [0.1, 0.9, 0.5, 0.3],
        "em":         [0.0, 1.0, 1.0, 0.0],
    }
    m = pbm.compute_prefix_buffer_quality_metrics(is_pb, pb_types, reward_metrics)
    # wrong prefixes = idx 0 and 3
    assert abs(m["prefix_buffer/reflection_wrong_mean"] - 0.3) < 1e-9
    assert abs(m["prefix_buffer/outcome_wrong_mean"] - 0.2) < 1e-9
    assert abs(m["prefix_buffer/em_wrong_mean"] - 0.0) < 1e-9
    assert m["prefix_buffer/reflection_wrong_count"] == 2
    # right prefixes = idx 1
    assert abs(m["prefix_buffer/reflection_right_mean"] - 0.8) < 1e-9
    assert abs(m["prefix_buffer/em_right_mean"] - 1.0) < 1e-9
    # all prefix-buffer = idx 0,1,3 -- the non-pb idx 2 must be excluded
    assert abs(m["prefix_buffer/em_all_mean"] - (1.0 / 3.0)) < 1e-9
    assert m["prefix_buffer/em_all_count"] == 3


def test_returns_empty_when_no_prefix_buffer_rollouts():
    m = pbm.compute_prefix_buffer_quality_metrics(
        [False, False], [None, None], {"em": [1.0, 0.0]})
    assert m == {}


def test_handles_none_is_pb():
    assert pbm.compute_prefix_buffer_quality_metrics(
        None, None, {"em": [1.0]}) == {}


def test_skips_missing_metric_keys():
    # reward_metrics lacks 'outcome' -> no outcome metrics; others still emitted
    m = pbm.compute_prefix_buffer_quality_metrics(
        [True], ["wrong"], {"reflection": [0.5], "em": [1.0]})
    assert "prefix_buffer/reflection_wrong_mean" in m
    assert "prefix_buffer/em_wrong_mean" in m
    assert not any("outcome" in k for k in m)


def test_handles_none_pb_types():
    # pb_types None -> only the _all_ aggregate, no per-type split
    m = pbm.compute_prefix_buffer_quality_metrics(
        [True, True], None, {"em": [1.0, 0.0]})
    assert abs(m["prefix_buffer/em_all_mean"] - 0.5) < 1e-9
    assert not any("_wrong_" in k or "_right_" in k for k in m)


def test_tolerates_short_value_lists():
    # reward_metrics list shorter than is_pb -> missing entries are skipped
    m = pbm.compute_prefix_buffer_quality_metrics(
        [True, True], ["wrong", "wrong"], {"em": [1.0]})
    assert m["prefix_buffer/em_wrong_count"] == 1
    assert abs(m["prefix_buffer/em_wrong_mean"] - 1.0) < 1e-9


# ============================================================
# New tests: extended default keys + non-PB / all slicing
# ============================================================

def test_default_keys_include_em_turn1_em_final_pr_turn1_pr_final():
    """The default keys must include em_turn1, em_final, pr_turn1, pr_final
    so wandb gets per-turn breakdown for buffer-mode rollouts (user request)."""
    reward_metrics = {
        "em_turn1": [0.0, 1.0, 0.5, 0.0],
        "em_final": [1.0, 1.0, 0.5, 0.0],
        "pr_turn1": [0.2, 0.9, 0.5, 0.0],
        "pr_final": [1.0, 0.9, 0.5, 0.0],
        "reflection": [0.3, 0.0, 0.0, 0.1],
        "n_turns": [3.0, 2.0, 2.0, 4.0],
        "em": [1.0, 1.0, 0.5, 0.0],
    }
    m = pbm.compute_prefix_buffer_quality_metrics(
        [True, True, False, True],
        ["wrong", "right", None, "wrong"],
        reward_metrics)
    # All five "missing" keys should now have wandb entries
    for key in ["em_turn1", "em_final", "pr_turn1", "pr_final", "reflection", "n_turns"]:
        assert f"prefix_buffer/{key}_wrong_mean" in m, f"missing key {key}_wrong_mean"
        assert f"prefix_buffer/{key}_right_mean" in m, f"missing key {key}_right_mean"
        assert f"prefix_buffer/{key}_all_mean" in m, f"missing key {key}_all_mean"
    # Check specific values: em_turn1 wrong = mean(0.0, 0.0) = 0.0
    assert abs(m["prefix_buffer/em_turn1_wrong_mean"] - 0.0) < 1e-9
    # em_final wrong = mean(1.0, 0.0) = 0.5 — buffer wrong rollouts that recovered
    assert abs(m["prefix_buffer/em_final_wrong_mean"] - 0.5) < 1e-9
    # em_final right = mean(1.0) = 1.0 (idx 1, the only right-prefix entry)
    assert abs(m["prefix_buffer/em_final_right_mean"] - 1.0) < 1e-9
    # pr_turn1 wrong = mean(0.2, 0.0) = 0.1
    assert abs(m["prefix_buffer/pr_turn1_wrong_mean"] - 0.1) < 1e-9


# --- compute_non_pb_quality_metrics ---

def test_non_pb_slices_only_non_buffer():
    is_pb = [True, True, False, True, False]
    reward_metrics = {
        "em": [0.0, 1.0, 0.7, 0.0, 0.5],
        "em_turn1": [0.0, 1.0, 1.0, 0.0, 0.0],
    }
    m = pbm.compute_non_pb_quality_metrics(is_pb, reward_metrics)
    # Only idx 2 (em=0.7, em_t1=1.0) and idx 4 (em=0.5, em_t1=0.0) qualify
    assert abs(m["non_prefix_buffer/em_mean"] - 0.6) < 1e-9
    assert m["non_prefix_buffer/em_count"] == 2
    assert abs(m["non_prefix_buffer/em_turn1_mean"] - 0.5) < 1e-9


def test_non_pb_returns_empty_when_all_buffer():
    assert pbm.compute_non_pb_quality_metrics(
        [True, True], {"em": [1.0, 0.0]}) == {}


def test_non_pb_handles_none_is_pb():
    # is_pb None -> treat all rollouts as non-buffer
    m = pbm.compute_non_pb_quality_metrics(None, {"em": [1.0, 0.5, 0.0]})
    assert abs(m["non_prefix_buffer/em_mean"] - 0.5) < 1e-9
    assert m["non_prefix_buffer/em_count"] == 3


def test_non_pb_extended_keys():
    """Should include em_turn1, em_final, pr_turn1, pr_final by default."""
    is_pb = [False, False]
    reward_metrics = {
        "em_turn1": [1.0, 0.0],
        "em_final": [1.0, 1.0],
        "pr_turn1": [1.0, 0.5],
        "pr_final": [1.0, 1.0],
        "reflection": [0.0, 0.3],
        "n_turns": [2.0, 3.0],
    }
    m = pbm.compute_non_pb_quality_metrics(is_pb, reward_metrics)
    for key in ["em_turn1", "em_final", "pr_turn1", "pr_final", "reflection", "n_turns"]:
        assert f"non_prefix_buffer/{key}_mean" in m
        assert f"non_prefix_buffer/{key}_count" in m
    assert abs(m["non_prefix_buffer/em_final_mean"] - 1.0) < 1e-9
    assert abs(m["non_prefix_buffer/reflection_mean"] - 0.15) < 1e-9


# --- compute_all_quality_metrics ---

def test_all_aggregates_over_every_rollout():
    """Should not filter on is_pb — includes both buffer and non-buffer."""
    reward_metrics = {
        "em": [1.0, 0.0, 0.5, 0.5],
        "em_turn1": [1.0, 0.0, 0.5, 0.0],
    }
    m = pbm.compute_all_quality_metrics(reward_metrics)
    assert abs(m["all/em_mean"] - 0.5) < 1e-9
    assert m["all/em_count"] == 4
    assert abs(m["all/em_turn1_mean"] - 0.375) < 1e-9


def test_all_handles_missing_keys():
    m = pbm.compute_all_quality_metrics({"em": [1.0, 0.0]})
    assert "all/em_mean" in m
    assert not any("em_turn1" in k for k in m)  # missing key, no output


def test_all_extended_keys():
    reward_metrics = {
        "em_turn1": [1.0, 0.0],
        "em_final": [1.0, 1.0],
        "pr_turn1": [1.0, 0.5],
        "pr_final": [1.0, 1.0],
        "reflection": [0.0, 0.3],
        "n_turns": [2.0, 3.0],
        "em": [1.0, 1.0],
        "format": [0.1, 0.0],
    }
    m = pbm.compute_all_quality_metrics(reward_metrics)
    for key in ["em", "em_turn1", "em_final", "pr_turn1", "pr_final",
                "reflection", "n_turns", "format"]:
        assert f"all/{key}_mean" in m
        assert f"all/{key}_count" in m
