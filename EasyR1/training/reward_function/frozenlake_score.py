"""FrozenLake reflection reward for EasyR1 multi-turn GRPO.

Self-contained (stdlib only) so the reward manager can load it by file path.

The committed answer is scored by one of two interchangeable outcome rewards,
selected by the ``outcome_reward`` kwarg:

  outcome_reward="em" (default)   -- binary exact-match jackpot:
      R = correct_value                          if committed answer is EM
      R = R_format + reflect_weight * R_reflect   otherwise

  outcome_reward="pr"             -- graded progress-rate outcome:
      R = correct_value                          if committed answer is EM
      R = R_format + reflect_weight * R_reflect
                   + outcome_weight * PR_final    otherwise  (clamped to correct_value)

R_format == 0 zeroes the total reward in both modes.
"""

import json
import re
from collections import deque

_DIR = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def bfs_dist(layout, src, tgt, L):
    """BFS distance from src to tgt over non-hole cells. None if unreachable."""
    if src == tgt:
        return 0
    visited = {src}
    q = deque([(src, 0)])
    while q:
        p, d = q.popleft()
        r, c = divmod(p, L)
        for dr, dc in _DIR.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < L and 0 <= nc < L):
                continue
            np = nr * L + nc
            if np in visited or layout[nr][nc] == "H":
                continue
            if np == tgt:
                return d + 1
            visited.add(np)
            q.append((np, d + 1))
    return None


def is_em(actions, layout, start, tgt, L, n_opt):
    """Strict EM (matches VisualPlanning convention).

    The trajectory IS an optimal trajectory iff:
      - len(actions) == n_opt          (no passthrough, no short trajectory)
      - every action is in-bounds      (no off-grid attempts)
      - no step lands on a hole
      - the LAST action lands on the goal

    A passthrough -- where the goal is reached at step k < n_opt-1 followed by
    extra actions -- returns False. A trajectory that reaches the goal in
    exactly n_opt actions returns True iff all intermediate steps are legal."""
    if actions is None or n_opt is None:
        return False
    if len(actions) != n_opt:
        return False
    r, c = divmod(start, L)
    for i, act in enumerate(actions):
        if act not in _DIR:
            return False
        dr, dc = _DIR[act]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < L and 0 <= nc < L):
            return False
        r, c = nr, nc
        if layout[r][c] == "H":
            return False
        if layout[r][c] == "G":
            return i == n_opt - 1  # goal must be reached on the LAST action
    return False  # didn't reach goal


def progress_rate(actions, layout, start, tgt, L, n_opt, overshoot_zero=False):
    """(longest strictly-distance-decreasing legal prefix) / n_opt. Range [0, 1].

    If overshoot_zero=True, a trajectory that completes the full optimal prefix
    (matched == n_opt) but carries extra trailing actions (len(actions) > n_opt)
    scores 0.0 instead of 1.0 -- the piecewise rule that closes the PR=1.0
    overshoot reward hack. EM (len == n_opt) and partial (len < n_opt)
    trajectories are unaffected.
    """
    if actions is None or n_opt is None:
        return 0.0
    if n_opt == 0:
        return 1.0
    cur = start
    cur_d = bfs_dist(layout, cur, tgt, L)
    if cur_d is None:
        return 0.0
    matched = 0
    for act in actions:
        if act not in _DIR:
            break
        r, c = divmod(cur, L)
        dr, dc = _DIR[act]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < L and 0 <= nc < L):
            break
        if layout[nr][nc] == "H":
            break
        nb = nr * L + nc
        nd = bfs_dist(layout, nb, tgt, L)
        if nd is None or nd >= cur_d:
            break
        matched += 1
        cur, cur_d = nb, nd
    if overshoot_zero and matched == n_opt and len(actions) > n_opt:
        return 0.0  # optimal prefix reached, but the commit overshoots -> 0
    return matched / n_opt


# ---------------------------------------------------------------------------
# Task 2: Transcript parser (route/terminate turns)
# ---------------------------------------------------------------------------


def _extract_json_blob(text):
    """First JSON object in `text` parsed to a dict, or None.
    Tolerates ```json fences and bare JSON; falls back to json_repair.
    Assumes at most one JSON object per `text` -- callers must split the transcript into individual turns first.
    """
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


def _parse_actions_csv(text):
    """Parse a comma- OR space-separated action string into a list of direction tokens.

    Accepts both legacy CSV (``"up, right"``) and the tag-format space-separated
    (``"up right"``) representations. Returns (actions, ok); ok is False if any
    token is unrecognized.
    """
    if text is None:
        return None, False
    text = str(text).strip()
    if not text:
        return [], True
    # Split on commas first; then any whitespace within each chunk. This way
    # "up, right" -> ["up", "right"] AND "up right" -> ["up", "right"] AND
    # mixed "up,right down" -> ["up", "right", "down"].
    tokens = []
    for chunk in text.split(","):
        for raw in chunk.split():
            tok = raw.strip().lower()
            if not tok:
                continue
            if tok not in _DIR:
                return None, False
            tokens.append(tok)
    return tokens, True


# Tag-format regexes. Non-greedy bodies; DOTALL so <think>...</think> can span
# newlines. The body of <route>/<answer> is captured and then re-split by the
# comma-or-whitespace parser, so messy whitespace inside is tolerated.
_TAG_ROUTE_RE = re.compile(r"<route\b[^>]*>(.*?)</route>", re.DOTALL | re.IGNORECASE)
_TAG_ANSWER_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Reflection-format tags. <FINAL> = commit (terminate); <THINK> = reasoning
# prefix on revision turns; <ANSWER> (in this fmt) = proposal (route). Same
# regex object is reused as _TAG_ANSWER_RE above -- semantics differ by fmt.
_TAG_FINAL_RE = re.compile(r"<final\b[^>]*>(.*?)</final>", re.DOTALL | re.IGNORECASE)
_TAG_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse_tag_turn(turn_text):
    """Parse a tag-format assistant turn.

    Returns (kind, actions, ok) with the same contract as ``_parse_one_turn``.
    ``<answer>`` maps to kind ``"terminate"`` so downstream reward logic is
    unchanged. Returns (None, None, False) if no `<route>` or `<answer>` tag is
    present. If BOTH tags appear in the same turn (malformed), prefers
    `<answer>` -- the model has committed.
    """
    if not turn_text:
        return None, None, False
    m_ans = _TAG_ANSWER_RE.search(turn_text)
    if m_ans is not None:
        actions, ok = _parse_actions_csv(m_ans.group(1))
        return "terminate", actions, ok
    m_route = _TAG_ROUTE_RE.search(turn_text)
    if m_route is not None:
        actions, ok = _parse_actions_csv(m_route.group(1))
        return "route", actions, ok
    return None, None, False


def _parse_reflection_tag_turn(turn_text):
    """Parse a reflection-format assistant turn.

    The reflection corpus uses three tag types:
      - ``<FINAL>X</FINAL>``      -> commit       (kind="terminate")
      - ``<ANSWER>X</ANSWER>``    -> proposal     (kind="route")
      - ``<THINK>...</THINK>``    -> reasoning prefix; metadata only.

    A revision turn pairs a ``<THINK>`` with an ``<ANSWER>``; turn-1 emits a
    bare ``<ANSWER>``; the final turn emits ``<FINAL>`` only. If both FINAL
    and ANSWER appear in the same turn (malformed), prefer FINAL -- the model
    has committed.

    Returns (kind, actions, ok); (None, None, False) if no recognized tag is
    found. Matches the ``_parse_one_turn`` contract so downstream reward logic
    is unchanged.
    """
    if not turn_text:
        return None, None, False
    m_final = _TAG_FINAL_RE.search(turn_text)
    if m_final is not None:
        actions, ok = _parse_actions_csv(m_final.group(1))
        return "terminate", actions, ok
    m_ans = _TAG_ANSWER_RE.search(turn_text)
    if m_ans is not None:
        actions, ok = _parse_actions_csv(m_ans.group(1))
        return "route", actions, ok
    return None, None, False


def _parse_one_turn(turn_text, fmt="tag"):
    """Parse one assistant turn. Returns (kind, actions, ok):
    kind in {'route','terminate',None}; actions is a list or None; ok is bool.

    Args:
        turn_text: raw assistant-turn text.
        fmt: which output format to expect. One of:
            "tag"            -- lowercase ``<route>``/``<answer>`` tag format.
            "reflection_tag" -- reflection format with UPPERCASE
                                ``<ANSWER>``/``<FINAL>`` and optional
                                ``<THINK>`` blocks on revision turns.
            "json"           -- JSON ``function_call`` format.

    The formats are disjoint in practice; the flag picks one path and does NOT
    fall back to the others.
    """
    if fmt == "tag":
        return _parse_tag_turn(turn_text)
    if fmt == "reflection_tag":
        return _parse_reflection_tag_turn(turn_text)
    if fmt == "json":
        return _parse_json_turn(turn_text)
    raise ValueError(
        f"_parse_one_turn: unknown fmt={fmt!r}; expected 'tag', 'reflection_tag', or 'json'"
    )


def _parse_json_turn(turn_text):
    """Parse one assistant turn in the legacy JSON ``function_call`` format.

    Same (kind, actions, ok) contract as ``_parse_one_turn``. Kept as a
    standalone function so the JSON path is preserved for backwards
    compatibility even though ``_parse_one_turn(fmt="tag")`` is the default.
    """
    data = _extract_json_blob(turn_text)
    if not isinstance(data, dict):
        return None, None, False
    fc = data.get("function_call")
    if not isinstance(fc, dict):
        return None, None, False
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
        actions, ok = _parse_actions_csv(args.get("answer", ""))
        return "terminate", actions, ok
    if name == "route":
        actions, ok = _parse_actions_csv(args.get("actions", ""))
        return "route", actions, ok
    return None, None, False


def _split_assistant_turns(response):
    """Return the raw text of each assistant turn in the transcript.

    Turn 1 is bare (the generation prompt put its <|im_start|>assistant marker in
    the prompt, not the response). Defensively strips a leaked run of vision/image
    tokens at the very start before prepending the marker. (`_extract_json_blob`
    already tolerates leading noise, so this is belt-and-suspenders.)
    """
    response = re.sub(
        r"^(?:<\|image_pad\|>|<\|vision_start\|>|<\|vision_end\|>)+", "", response
    )
    if not response.startswith("<|im_start|>assistant"):
        response = "<|im_start|>assistant\n" + response
    pattern = r"<\|im_start\|>assistant\s*\n(.*?)(?=<\|im_end\|>|<\|im_start\|>user|$)"
    return [t for t in re.findall(pattern, response, re.DOTALL) if t.strip()]


def parse_turns(response, fmt="tag"):
    """Parse the multi-turn transcript into a list of
    {'kind': str|None, 'actions': list|None, 'ok': bool, 'raw_text': str}
    dicts, one per turn.

    The 'raw_text' field carries the un-parsed turn body so downstream
    format checks (e.g. detecting a ``<THINK>`` block) can inspect the
    surrounding text without re-splitting the transcript.

    ``fmt`` selects "tag" (lowercase route/answer), "reflection_tag"
    (uppercase ANSWER/FINAL with optional THINK), or "json";
    see ``_parse_one_turn`` for the contract.
    """
    turns = []
    for turn_text in _split_assistant_turns(response):
        kind, actions, ok = _parse_one_turn(turn_text, fmt=fmt)
        turns.append(
            {"kind": kind, "actions": actions, "ok": ok, "raw_text": turn_text}
        )
    return turns


# ---------------------------------------------------------------------------
# Task 3: Illegal-move detection and ground-truth decoding
# ---------------------------------------------------------------------------


def simulate_has_illegal(actions, layout, start, L):
    """True if the trajectory attempts an off-grid move or steps onto a hole.
    Stops at the goal (a pass-through past the goal is not evaluated)."""
    if not actions:
        return False
    r, c = divmod(start, L)
    for act in actions:
        if act not in _DIR:
            return True
        dr, dc = _DIR[act]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < L and 0 <= nc < L):
            return True  # off-grid attempt
        r, c = nr, nc
        if layout[r][c] == "H":
            return True
        if layout[r][c] == "G":
            break
    return False


def trajectory_leaves_grid(actions, layout, start, L):
    """True if the trajectory attempts a move off the grid boundary.

    Off-grid only -- a hole is *on* the map and is not flagged here. Stops at a
    hole or the goal (subsequent moves are not evaluated, matching
    simulate_has_illegal). An unparseable token is treated as invalid."""
    if not actions:
        return False
    r, c = divmod(start, L)
    for act in actions:
        if act not in _DIR:
            return True
        dr, dc = _DIR[act]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < L and 0 <= nc < L):
            return True  # off-grid attempt
        r, c = nr, nc
        if layout[r][c] in ("H", "G"):
            break
    return False


def _decode_ground_truth(gt):
    """Parse the reward-input ground truth into (layout, start, target, level).
    Accepts a dict or a JSON string."""
    if isinstance(gt, str):
        gt = json.loads(gt)
    if not isinstance(gt, dict):
        raise ValueError("ground_truth is not a dict or JSON object")
    layout = gt["layout"]
    start = int(gt["start_pos"])
    target = int(gt["target_pos"])
    level = int(gt["level"])
    return layout, start, target, level


# ---------------------------------------------------------------------------
# Task 4: Delta-form reflection score
# ---------------------------------------------------------------------------


def _reflection_raw_sum(pr_list, lambda_deg=1.0):
    """Un-clipped signed-weighted-delta sum used by both compute_reflection_score
    and the `break` diagnostic. Returns 0.0 when fewer than 2 route turns (no
    revision occurred -- no break possible).
    """
    if len(pr_list) < 2:
        return 0.0
    r = 0.0
    for prev, cur in zip(pr_list, pr_list[1:]):
        d = float(cur) - float(prev)
        r += d if d >= 0.0 else lambda_deg * d
    return r


def compute_reflection_score(pr_list, lambda_deg=1.0, lower_clip=0.0):
    """Pure-improvement reflection score, normalized to [lower_clip, 1].

    Reflection requires at least two route proposals. With fewer than two route
    turns no revision occurred, so the score is 0.0 regardless of lower_clip.

        R_reflect = clip( sum_t w(delta_t), lower_clip, 1 )   (>= 2 route turns)
        delta_t   = PR_{t+1} - PR_t
        w(delta)  = delta             if delta >= 0   (progress across the turn)
                    lambda_deg*delta  if delta <  0   (degeneration, penalized harder)

    PR_1 is intentionally NOT added: R_reflect measures the per-turn improvement
    only, not the final answer's quality. lambda_deg > 1 makes backslides cost
    more than the progress that undid them, so it matters on mixed (some-progress,
    some-backslide) sequences. The running sum is NOT clipped between steps;
    only the final value is clipped.

    lower_clip controls how much net-negative reflection (a "break") shows up in
    the reward. Default 0.0 preserves the legacy behavior (breaks invisible).
    Set negative (e.g. -0.05, -0.1) to surface breaks: a single-revision break
    that drops PR from 1.0 -> 0.5 then yields R_reflect = -0.5 (clamped to
    lower_clip), creating a per-rollout cost for indiscriminate revision.
    Sequences with fewer than two route turns still score 0.0 (no revision
    occurred -- no signal either way).
    """
    if len(pr_list) < 2:
        return 0.0
    r = _reflection_raw_sum(pr_list, lambda_deg=lambda_deg)
    return max(float(lower_clip), min(1.0, r))


# ---------------------------------------------------------------------------
# Task 5: Format/gate score
# ---------------------------------------------------------------------------


def compute_format_score(
    turns, final_actions, layout, start, level, min_turns=1, fmt="tag"
):
    """Return 0.1 if the transcript is structurally valid; else 0.0.

    Trajectory legality (off-grid / hole) is NOT checked here -- the model is
    allowed to step off the grid or onto a hole during reflection and recover.
    Strict EM still rejects illegal committed trajectories (so they cannot earn
    R = 1.0), but the format gate does not punish the attempt.

    Common constraints (all formats):
      - at least one turn, and every turn parses (kind not None, ok True);
      - first turn is `route`; exactly one `terminate`, and it is last;
      - len(turns) >= min_turns;
      - no turn has an empty action list;
      - (constraint 1) terminate matches the immediately preceding route;
      - (constraint 3) no two `route` turns have an identical trajectory;
      - final_actions is not None.

    Additional constraints for fmt == "reflection_tag":
      - The first turn (an ``<ANSWER>`` proposal) MUST NOT contain a
        ``<THINK>`` block.
      - The FINAL turn (a ``<FINAL>`` commit) MUST NOT contain a ``<THINK>``
        block -- ``<FINAL>`` is commit-only, no rationale.
      - Every revision turn (any turn that is not turn-1 and not the FINAL)
        MUST contain a ``<THINK>`` block.
    """
    if not turns:
        return 0.0
    if any((not t["ok"]) or (t["kind"] is None) for t in turns):
        return 0.0
    kinds = [t["kind"] for t in turns]
    if kinds[0] != "route":
        return 0.0
    if kinds.count("terminate") != 1 or kinds[-1] != "terminate":
        return 0.0
    # minimum turn count (shielded by the correct branch in compute_score: an EM
    # rollout scores 1.0 regardless, so this only nudges wrong-and-unreflected ones)
    if len(turns) < min_turns:
        return 0.0
    # no empty trajectories (every turn has parsed `ok` here, so actions is a list)
    if any(not t["actions"] for t in turns):
        return 0.0
    # constraint 1: terminate must commit the immediately preceding route unchanged
    # (structure above guarantees turns[-1] is the lone terminate and turns[-2] a route)
    if turns[-1]["actions"] != turns[-2]["actions"]:
        return 0.0
    # constraint 3: no two route turns share an identical trajectory
    route_sigs = [tuple(t["actions"]) for t in turns if t["kind"] == "route"]
    if len(route_sigs) != len(set(route_sigs)):
        return 0.0
    if final_actions is None:
        return 0.0
    # reflection_tag-specific structural checks on per-turn raw_text.
    if fmt == "reflection_tag":
        n = len(turns)
        for i, t in enumerate(turns):
            raw = t.get("raw_text", "") or ""
            has_think = _TAG_THINK_RE.search(raw) is not None
            if i == 0:
                # turn 1 is the first <ANSWER> proposal -- no <THINK> allowed
                # (no feedback to reflect on yet).
                if has_think:
                    return 0.0
            elif i == n - 1:
                # FINAL turn -- commit only, no <THINK> block.
                if has_think:
                    return 0.0
            else:
                # Revision turn: <THINK> is REQUIRED.
                if not has_think:
                    return 0.0
    return 0.1


# ---------------------------------------------------------------------------
# Task 6: compute_score integration
# ---------------------------------------------------------------------------

_METRIC_KEYS = (
    "overall",
    "format",
    "reflection",
    "em",
    "pr_final",
    "pr_turn1",
    "n_turns",
    "outcome",
    "em_turn1",
    "em_final",
    "break",
    "decision_bonus",
    "step_cost",
)


def _floor_result():
    return {k: 0.0 for k in _METRIC_KEYS}


def _score_one(
    reward_input,
    lambda_deg,
    reflect_weight,
    min_turns,
    correct_value,
    fmt="tag",
    outcome_reward="em",
    outcome_weight=0.9,
    pr_overshoot_zero=False,
    reflect_lower_clip=0.0,
    em_reflect_bonus_weight=0.0,
    em_reflect_step_discount=1.0,
    prefix_buffer_decision_bonus=0.0,
    step_cost=0.0,
    step_cost_churn_only=False,
):
    response = reward_input.get("response", "") or ""

    try:
        layout, start, target, level = _decode_ground_truth(
            reward_input.get("ground_truth")
        )
    except Exception:
        return _floor_result()

    n_opt = bfs_dist(layout, start, target, level)
    if n_opt is None:
        return _floor_result()

    turns = parse_turns(response, fmt=fmt)
    route_turns = [
        t for t in turns if t["kind"] == "route" and t["actions"] is not None
    ]
    pr_list = [
        progress_rate(
            t["actions"],
            layout,
            start,
            target,
            level,
            n_opt,
            overshoot_zero=pr_overshoot_zero,
        )
        for t in route_turns
    ]

    # committed answer: last clean terminate, else last route, else None
    final_actions = None
    for t in turns:
        if t["kind"] == "terminate" and t["actions"] is not None:
            final_actions = t["actions"]
    if final_actions is None:
        for t in turns:
            if t["kind"] == "route" and t["actions"] is not None:
                final_actions = t["actions"]

    em = bool(is_em(final_actions, layout, start, target, level, n_opt))
    pr_final = (
        progress_rate(
            final_actions,
            layout,
            start,
            target,
            level,
            n_opt,
            overshoot_zero=pr_overshoot_zero,
        )
        if final_actions is not None
        else 0.0
    )
    pr_turn1 = pr_list[0] if pr_list else 0.0

    # Per-rollout diagnostics for the wandb reward/* aggregates. em_turn1 is
    # paper-EM on the FIRST <ANSWER>'s actions (turn-1 capability); em_final
    # mirrors `em` (renamed for symmetry). `break` is 1.0 iff a revision was
    # attempted AND the unclipped signed-weighted-delta sum is negative -- the
    # rollout net-degraded its trajectory across route turns. It uses the RAW
    # sum so the metric stays meaningful regardless of reflect_lower_clip.
    em_turn1 = (
        float(
            bool(is_em(route_turns[0]["actions"], layout, start, target, level, n_opt))
        )
        if route_turns
        else 0.0
    )
    em_final = float(em)
    break_flag = (
        1.0 if _reflection_raw_sum(pr_list, lambda_deg=lambda_deg) < 0.0 else 0.0
    )

    fmt_score = compute_format_score(
        turns, final_actions, layout, start, level, min_turns=min_turns, fmt=fmt
    )
    reflect = compute_reflection_score(
        pr_list, lambda_deg=lambda_deg, lower_clip=reflect_lower_clip
    )

    # R_format == 0 zeroes the TOTAL reward -- overriding both outcome modes.
    # A transcript that fails any format constraint (e.g. an off-grid trajectory
    # at any turn) scores 0 regardless of the committed answer's correctness.
    if fmt_score <= 0.0:
        overall = 0.0
        outcome = 0.0
    elif em:
        # EM is the ceiling in BOTH outcome modes: a EM-optimal committed
        # answer earns exactly correct_value.
        # Optionally add a reflection bonus ON TOP of the EM jackpot, gated by
        # `em_reflect_bonus_weight > 0` (default 0.0 preserves legacy behavior:
        # EM=1 -> flat correct_value, reflection ignored). When enabled, the
        # bonus is discounted by `em_reflect_step_discount ** n_revisions` to
        # prevent "deliberately fail turn-1 to harvest the bonus" gaming -- a
        # multi-turn-correct path pays less than a fast direct-correct path
        # once the discount kicks in. n_revisions = number of route turns - 1
        # (i.e., 0 if the model FINALed in turn 1, 1 if one revision, etc.).
        if em_reflect_bonus_weight > 0 and reflect > 0:
            n_revisions = max(0, len(pr_list) - 1)
            bonus = (
                em_reflect_bonus_weight
                * reflect
                * (em_reflect_step_discount**n_revisions)
            )
            overall = correct_value + bonus
        else:
            overall = correct_value
        outcome = correct_value
    elif outcome_reward == "pr":
        # Progress-rate outcome reward. The committed answer earns a graded
        # outcome_weight * pr_final instead of the binary EM jackpot, so two
        # non-EM trajectories that made different progress get different
        # rewards. Clamped to correct_value so EM stays the strict ceiling.
        outcome = outcome_weight * pr_final
        overall = min(correct_value, fmt_score + reflect_weight * reflect + outcome)
    else:
        # EM outcome reward (default). A non-EM committed answer earns no
        # outcome term; the non-EM reward is the format gate plus the
        # reflection bonus only.
        outcome = 0.0
        overall = fmt_score + reflect_weight * reflect

    # --------------------------------------------------------------------
    # Optional `step_cost`. When the committed answer is EM and step_cost > 0,
    # subtract a per-revision-turn penalty from `overall` to discourage
    # unnecessary extra turns (and to bias toward the single-turn path).
    # Default 0.0 = no-op (legacy preserved). Two modes:
    #   step_cost_churn_only=False (default): tax EVERY revision turn
    #       penalty = step_cost * max(len(pr_list) - 1, 0)
    #   step_cost_churn_only=True: tax only the route turns AFTER the first
    #       one that already reached PR == 1.0 (post-correct churn) -- spares
    #       a productive wrong->right revision, penalizes only over-revision.
    # Gated by `em` and `fmt_score > 0` (a malformed-but-EM rollout is already
    # floored to 0 by the format gate; do not push it negative).
    # --------------------------------------------------------------------
    step_cost_penalty = 0.0
    if step_cost > 0.0 and em and fmt_score > 0.0:
        if step_cost_churn_only:
            penalized_turns = 0
            seen_correct = False
            for p in pr_list:
                if seen_correct:
                    penalized_turns += 1
                if p >= 1.0:
                    seen_correct = True
        else:
            penalized_turns = max(len(pr_list) - 1, 0)
        step_cost_penalty = step_cost * penalized_turns
        overall = overall - step_cost_penalty

    # --------------------------------------------------------------------
    # Optional `prefix_buffer_decision_bonus`
    #
    # Pays a small additive bonus / penalty on the FIRST new turn after a
    # buffered prefix, gated by `prefix_buffer_decision_bonus > 0`. Only
    # makes sense when `prefix_buffer_force_pointing=false` (so the model
    # has a real choice between FINAL and a fresh route on its first new
    # turn). Encodes the four-quadrant decision matrix:
    #
    #   prefix_type=wrong:  revise (route) + em=1  -> +bonus
    #                       FINAL immediately      -> -bonus
    #   prefix_type=right:  FINAL immediately + em=1 -> +bonus
    #                       revise (route)           -> -bonus
    #
    # Gated by fmt_score > 0 (no bonus on malformed rollouts) and by the
    # presence of the three per-sample fields populated upstream:
    # `is_prefix_buffer_rollout`, `prefix_buffer_type`, `prefix_buffer_num_turns`.
    # The bonus is ADDED uncapped to overall by design (i.e., a wrong-prefix
    # revise+correct rollout can earn slightly more than correct_value, and
    # a confidently-wrong stick-with-it can pay below the format floor).
    # --------------------------------------------------------------------
    decision_bonus_amt = 0.0
    if (
        prefix_buffer_decision_bonus > 0
        and fmt_score > 0.0
        and reward_input.get("is_prefix_buffer_rollout")
    ):
        prefix_type = reward_input.get("prefix_buffer_type")
        num_prefix_turns = int(reward_input.get("prefix_buffer_num_turns", 0) or 0)
        if (
            num_prefix_turns > 0
            and num_prefix_turns < len(turns)
            and prefix_type in ("wrong", "right")
        ):
            first_new_kind = turns[num_prefix_turns].get("kind")
            revised = first_new_kind == "route"
            finaled = first_new_kind == "terminate"
            if prefix_type == "wrong":
                if revised and em > 0.5:
                    decision_bonus_amt = +prefix_buffer_decision_bonus
                elif finaled:
                    decision_bonus_amt = -prefix_buffer_decision_bonus
            elif prefix_type == "right":
                if finaled and em > 0.5:
                    decision_bonus_amt = +prefix_buffer_decision_bonus
                elif revised:
                    decision_bonus_amt = -prefix_buffer_decision_bonus
    overall = overall + decision_bonus_amt

    return {
        "overall": float(overall),
        "format": float(fmt_score),
        "reflection": float(reflect),
        "em": float(em),
        "pr_final": float(pr_final),
        "pr_turn1": float(pr_turn1),
        "n_turns": float(len(turns)),
        "outcome": float(outcome),
        "em_turn1": em_turn1,
        "em_final": em_final,
        "break": break_flag,
        "decision_bonus": float(decision_bonus_amt),
        "step_cost": float(step_cost_penalty),
    }


def compute_score(
    reward_inputs,
    lambda_deg=1.0,
    reflect_weight=0.9,
    min_turns=1,
    correct_value=1.0,
    log_filename=None,
    fmt="tag",
    outcome_reward="em",
    outcome_weight=0.9,
    pr_overshoot_zero=False,
    reflect_lower_clip=0.0,
    em_reflect_bonus_weight=0.0,
    em_reflect_step_discount=1.0,
    prefix_buffer_decision_bonus=0.0,
    step_cost=0.0,
    step_cost_churn_only=False,
    **kwargs,
):
    """Batch reward for FrozenLake multi-turn route/terminate rollouts.

    The committed answer is scored by one of two interchangeable outcome
    rewards, selected by ``outcome_reward``:

    outcome_reward="em" (default) -- binary exact-match outcome:
        R = 0                                       if R_format == 0
        R = correct_value                           if committed answer is EM-optimal
        R = R_format + reflect_weight * R_reflect    otherwise

    outcome_reward="pr" -- graded progress-rate outcome:
        R = 0                                       if R_format == 0
        R = correct_value                           if committed answer is EM-optimal
        R = min(correct_value,                      otherwise
                R_format + reflect_weight * R_reflect + outcome_weight * PR_final)

    PR mode replaces the all-or-nothing EM jackpot for non-EM rollouts with a
    graded outcome_weight * PR_final term: two failed trajectories that made
    different progress receive different rewards (a denser learning signal).

    R_format checks transcript structure only -- it does NOT punish off-grid or
    hole steps in any turn. The model is allowed to step off the grid or onto
    a hole during reflection and is expected to recover; strict EM still
    rejects illegal committed trajectories so they cannot earn R = 1.0.

    kwargs (set via worker.reward.reward_function_kwargs):
        lambda_deg     -- degeneration penalty multiplier in the reflection score.
                          1.0 = absolute (R_reflect == PR_final); >1.0 = delta form.
        reflect_weight -- weight on R_reflect in the incorrect branch (default 0.9).
        min_turns      -- minimum transcript length for R_format to be 0.1
                          (1 = no-op; 3 = require at least one revision).
        correct_value  -- reward for an exact-match-optimal committed answer.
        outcome_reward -- which outcome reward scores the committed answer:
                          "em" (default) = binary EM jackpot only; "pr" =
                          graded outcome_weight * PR_final for non-EM rollouts.
                          EM still earns correct_value in both modes.
        outcome_weight -- weight on PR_final in the "pr" outcome mode; ignored
                          when outcome_reward="em" (default 0.9).
        pr_overshoot_zero -- if True, progress rate uses the piecewise rule: an
                          overshoot commit (optimal prefix reached but
                          len > n_opt) scores PR = 0 instead of 1. Applies to
                          PR_final, PR_turn1 and every reflection delta. Closes
                          the PR=1.0 overshoot reward hack. Default False.
        reflect_lower_clip -- lower clip on R_reflect (default 0.0 = legacy
                          behavior, breaks invisible). Set negative (e.g.
                          -0.05, -0.1) to let net-negative reflection surface:
                          a single-revision break that drops PR by Δ contributes
                          -lambda_deg*|Δ| (clamped to reflect_lower_clip) to
                          R_reflect, creating a per-rollout cost for
                          indiscriminate revision. Upper bound stays at 1.0.
        log_filename   -- if set, append one JSONL record per reward_input to this
                          path. Rank-suffixed automatically (".rankN") so multiple
                          DP workers can write without conflict. Each record:
                          {"response": str, "ground_truth": str, "score": {...}}.
        fmt            -- which output format to expect from the model:
                          "tag" (lowercase <route>/<answer>),
                          "reflection_tag" (uppercase <ANSWER>/<FINAL> with
                          optional <THINK> on revision turns), or "json"
                          (function_call schema). Set this in the YAML
                          reward_function_kwargs to match the rolled-out model.
    Any other kwargs (e.g. is_random_init_rollout from the reward manager,
    or the legacy `illegal_gate` flag, which is now ignored) are swallowed.
    """
    if outcome_reward not in ("em", "pr"):
        raise ValueError(
            f"compute_score: unknown outcome_reward={outcome_reward!r}; "
            f"expected 'em' or 'pr'"
        )
    results = [
        _score_one(
            ri,
            lambda_deg,
            reflect_weight,
            min_turns,
            correct_value,
            fmt=fmt,
            outcome_reward=outcome_reward,
            outcome_weight=outcome_weight,
            pr_overshoot_zero=pr_overshoot_zero,
            reflect_lower_clip=reflect_lower_clip,
            em_reflect_bonus_weight=em_reflect_bonus_weight,
            em_reflect_step_discount=em_reflect_step_discount,
            prefix_buffer_decision_bonus=prefix_buffer_decision_bonus,
            step_cost=step_cost,
            step_cost_churn_only=step_cost_churn_only,
        )
        for ri in reward_inputs
    ]
    if log_filename:
        import os as _os

        rank = _os.environ.get("RANK", "0")
        path = log_filename + ".rank" + str(rank)
        try:
            _os.makedirs(_os.path.dirname(path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(path, "a", encoding="utf-8") as f:
                for ri, sc in zip(reward_inputs, results):
                    rec = {
                        "response": ri.get("response", ""),
                        "ground_truth": ri.get("ground_truth", ""),
                        "score": sc,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception as _e:
            print("[frozenlake reward] failed to write log:", _e)
    return results
