"""FrozenLake trajectory scoring + base-map rendering.

Self-contained scoring logic for the FrozenLake shortest-path task, used by the
eval client (`api_inference_frozenlake.py`) and, transitively, by the verl
multi-turn rollout.

Public surface (imported elsewhere):
  - `_parse_action_csv`            parse/validate a comma-separated action string
  - `evaluate_frozenlake_trajectory`  format + semantic (paper-EM) judgment
  - `_render_frozenlake_base_rgb`  render the base grid (elf at start) to RGB

Only stdlib is needed at import time; numpy + gymnasium are imported lazily
inside the renderer.
"""

from __future__ import annotations


# Row-major (dr, dc) deltas for the four cardinal moves.
_FL_DIR_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _parse_action_csv(answer: str, parsed_for_log: dict):
    """Helper: split a comma-separated direction string into validated tokens."""
    if not isinstance(answer, str) or not answer.strip():
        return None, parsed_for_log, "answer string is empty"
    tokens = [t.strip().lower() for t in answer.split(",") if t.strip()]
    if not tokens:
        return None, parsed_for_log, "answer empty after splitting on commas"
    bad = [t for t in tokens if t not in _FL_DIR_DELTA]
    if bad:
        return (
            None,
            parsed_for_log,
            (
                f"answer contains invalid direction tokens (expected only "
                f"up/down/left/right): {bad}"
            ),
        )
    return tokens, parsed_for_log, None


def simulate_frozenlake_trajectory(actions, layout, start_pos: int, level: int):
    """Replay `actions` on `layout` starting from `start_pos` (cell index in
    row-major order). Stops on the first hole hit or goal reach. Walls (steps
    that would leave the grid) are treated as no-ops, matching the
    Gymnasium FrozenLake-v1 environment.

    Returns a dict with `positions`, `reached_goal`, `hit_obstacle`,
    `n_actions_consumed`, `n_actions_total`, `wall_hit_steps` (action indices
    where the player tried to leave the grid).
    """
    r, c = divmod(int(start_pos), int(level))
    positions = [r * level + c]
    reached_goal = False
    hit_obstacle = False
    wall_hit_steps: list[int] = []
    n_consumed = 0
    for i, act in enumerate(actions):
        dr, dc = _FL_DIR_DELTA[act]
        nr, nc = r + dr, c + dc
        n_consumed = i + 1
        if not (0 <= nr < level and 0 <= nc < level):
            wall_hit_steps.append(i)
            positions.append(r * level + c)
            continue
        r, c = nr, nc
        cell = layout[r][c]
        positions.append(r * level + c)
        if cell == "H":
            hit_obstacle = True
            break
        if cell == "G":
            reached_goal = True
            break
    return {
        "positions": positions,
        "reached_goal": reached_goal,
        "hit_obstacle": hit_obstacle,
        "wall_hit_steps": wall_hit_steps,
        "n_actions_consumed": n_consumed,
        "n_actions_total": len(actions),
    }


def evaluate_frozenlake_trajectory(actions, layout, start_pos, level, format_error):
    """Combine the format check with the simulator-based semantic check.

    `is_correct` requires (a) no off-grid attempts, (b) no holes hit, and
    (c) the trajectory ends on the goal. The simulator's gym-style wall-as-no-op
    semantics are recorded in `wall_hit_steps`, but for evaluation purposes any
    off-grid attempt is treated as a failure (optimal EM convention). Use
    `sim["reached_goal"]` directly if you want the lenient (gym) judgment.
    """
    valid_format = (format_error is None) and (actions is not None)
    sim = None
    semantic_error = None
    is_correct = False
    if valid_format:
        sim = simulate_frozenlake_trajectory(actions, layout, start_pos, level)
        if sim.get("wall_hit_steps"):
            steps = sim["wall_hit_steps"]
            semantic_error = (
                f"trajectory attempts to leave the grid at step(s) "
                f"{', '.join(str(s + 1) for s in steps)}"
            )
        elif sim["hit_obstacle"]:
            semantic_error = "trajectory steps onto a hole tile"
        elif not sim["reached_goal"]:
            if sim["n_actions_consumed"] < sim["n_actions_total"]:
                semantic_error = "trajectory ended before consuming all actions"
            else:
                semantic_error = "trajectory does not reach the goal"
        else:
            is_correct = True
    return {
        "valid_format": valid_format,
        "format_error": format_error,
        "actions": actions,
        "simulation": sim,
        "semantic_error": semantic_error,
        "is_correct": is_correct,
    }


# Cache the rendered base map per layout key so we don't reload sprites for
# every retry. Keyed by ((layout-as-tuple), start_pos).
_FROZENLAKE_BASE_CACHE: dict = {}


def _render_frozenlake_base_rgb(layout, start_pos, level):
    """Return an HxWx3 RGB ndarray of the base FrozenLake map (elf at start)."""
    import gymnasium as gym
    import numpy as np

    key = (tuple(tuple(row) for row in layout), int(start_pos))
    cached = _FROZENLAKE_BASE_CACHE.get(key)
    if cached is not None:
        return cached
    desc = ["".join(row) for row in layout]
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=False,
        render_mode="rgb_array",
    )
    env.reset()
    env.unwrapped.s = int(start_pos)
    env.unwrapped.lastaction = None
    img = np.asarray(env.render())
    env.close()
    _FROZENLAKE_BASE_CACHE[key] = img
    return img
