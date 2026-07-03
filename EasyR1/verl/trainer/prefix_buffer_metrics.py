"""Quality metrics for multi-turn RL rollouts, sliced into prefix-buffer
sub-buckets (wrong/right/all), non-prefix-buffer rollouts, and all rollouts.

The prefix buffer seeds multi-turn rollouts from a stored first-turn prefix
that is either "wrong" (a non-EM proposal the model must revise) or "right".
The functions below aggregate per-rollout reward components (em, em_turn1,
em_final, pr_turn1, pr_final, reflection, n_turns, outcome, format) across:

  - prefix_buffer/{key}_wrong_mean | _right_mean | _all_mean (buffer rollouts)
  - non_prefix_buffer/{key}_mean                            (non-buffer rollouts)
  - all/{key}_mean                                           (every rollout)

Pure (stdlib only) so it is trivially unit-testable.
"""


# Default keys we want a slice for, in every wandb log. The turn-1 vs
# turn-final EM/PR breakdown shows where the buffer slice's em drag comes from
# and whether wrong-prefix rollouts actually recover.
DEFAULT_KEYS = (
    "em", "em_turn1", "em_final",
    "pr_turn1", "pr_final",
    "reflection", "n_turns",
    "outcome", "format",
)


def _emit_slice(out, prefix, key, vals_iter):
    """Helper: compute mean over `vals_iter` and write the wandb pair."""
    vals = list(vals_iter)
    if not vals:
        return
    out["%s/%s_mean" % (prefix, key)] = sum(vals) / len(vals)
    out["%s/%s_count" % (prefix, key)] = len(vals)


def compute_prefix_buffer_quality_metrics(is_pb, pb_types, reward_metrics,
                                          keys=DEFAULT_KEYS,
                                          prefix="prefix_buffer"):
    """Mean of each reward-metric key over prefix-buffer rollouts, split by
    prefix type (wrong vs right) and aggregated.

    Args:
        is_pb: per-rollout iterable of bools -- True if the rollout was seeded
            from the prefix buffer. May be None (returns {} then).
        pb_types: per-rollout iterable; for prefix-buffer rollouts, "wrong" or
            "right"; None/missing for others. May be None (no per-type split).
        reward_metrics: dict[str, list] -- per-rollout score lists keyed by
            metric name, as returned by the reward manager.
        keys: which reward-metric keys to aggregate. Defaults to DEFAULT_KEYS.
        prefix: metric-name prefix for the returned keys.

    Returns:
        dict[str, float|int] of wandb metrics. For each key it emits, when the
        corresponding rollouts exist:
          {prefix}/{key}_all_mean   {prefix}/{key}_all_count
          {prefix}/{key}_wrong_mean {prefix}/{key}_wrong_count
          {prefix}/{key}_right_mean {prefix}/{key}_right_count
        Empty dict if there are no prefix-buffer rollouts.
    """
    out = {}
    if is_pb is None:
        return out
    n = len(is_pb)
    for key in keys:
        vals = reward_metrics.get(key)
        if vals is None:
            continue
        all_vals = []
        by_type = {}
        for i in range(n):
            if not is_pb[i]:
                continue
            if i >= len(vals) or vals[i] is None:
                continue
            v = float(vals[i])
            all_vals.append(v)
            ptype = pb_types[i] if (pb_types is not None
                                    and i < len(pb_types)) else None
            if ptype:
                by_type.setdefault(str(ptype), []).append(v)
        _emit_slice(out, prefix, key + "_all", all_vals)
        for ptype, tv in by_type.items():
            _emit_slice(out, prefix, key + "_" + ptype, tv)
    return out


def compute_non_pb_quality_metrics(is_pb, reward_metrics,
                                   keys=DEFAULT_KEYS,
                                   prefix="non_prefix_buffer"):
    """Mean of each reward-metric key over NON-prefix-buffer rollouts.

    Mirror of compute_prefix_buffer_quality_metrics for the complement slice.
    Useful for separating turn-1-fresh-rollout quality (this) from
    buffer-mode-replay quality.

    Args:
        is_pb: per-rollout iterable of bools. None -> treat every rollout as
            non-buffer (typical for runs with prefix_buffer_mode_weight=0).
        reward_metrics: dict[str, list].
        keys: which reward-metric keys to aggregate.
        prefix: metric-name prefix.

    Returns:
        dict[str, float|int] with `{prefix}/{key}_mean` and `{prefix}/{key}_count`
        for each key that has data. Empty dict if no non-buffer rollouts.
    """
    out = {}
    # Determine the iteration domain: either all entries in reward_metrics, or
    # subset of indices where is_pb is False.
    if is_pb is None:
        # All rollouts qualify as non-buffer.
        for key in keys:
            vals = reward_metrics.get(key)
            if vals is None:
                continue
            clean = [float(v) for v in vals if v is not None]
            _emit_slice(out, prefix, key, clean)
        return out

    n = len(is_pb)
    for key in keys:
        vals = reward_metrics.get(key)
        if vals is None:
            continue
        clean = []
        for i in range(n):
            if is_pb[i]:
                continue
            if i >= len(vals) or vals[i] is None:
                continue
            clean.append(float(vals[i]))
        _emit_slice(out, prefix, key, clean)
    return out


def compute_all_quality_metrics(reward_metrics,
                                keys=DEFAULT_KEYS,
                                prefix="all"):
    """Mean of each reward-metric key over ALL rollouts (no filtering).

    Used as the "ground truth across the whole step" view, including OLF-rejected
    rollouts when called against the unfiltered all_metrics accumulator.

    Args:
        reward_metrics: dict[str, list].
        keys: which reward-metric keys to aggregate.
        prefix: metric-name prefix.

    Returns:
        dict[str, float|int] with `{prefix}/{key}_mean` and `{prefix}/{key}_count`
        for each key that has data. Empty for missing keys.
    """
    out = {}
    for key in keys:
        vals = reward_metrics.get(key)
        if vals is None:
            continue
        clean = [float(v) for v in vals if v is not None]
        _emit_slice(out, prefix, key, clean)
    return out
