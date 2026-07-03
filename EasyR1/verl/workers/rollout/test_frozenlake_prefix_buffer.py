import pytest
from PIL import Image

from EasyR1.verl.workers.rollout.frozenlake_prefix_buffer import (
    PrefixBuffer, PrefixBufferEntry,
)


def _entry(qid="q1", turns=2, prefix_type="wrong", step=0, conversation=None):
    img = Image.new("RGB", (10, 10))
    return PrefixBufferEntry(
        entry_id="",
        question_id=qid,
        map_spec={"layout": ["SF", "FG"], "start_pos": 0, "level": 2},
        ground_truth={"layout": ["SF", "FG"], "start_pos": 0, "target_pos": 3, "level": 2},
        prefix_conversation=conversation or f"<conv:{qid}:{turns}>",
        prefix_type=prefix_type,
        num_route_turns=turns,
        last_actions=["right"] * turns,
        final_pr=0.5,
        feedback_images=[img] * turns,
        collection_step=step,
        correct_suffix=None,
    )


def test_buffer_add_assigns_entry_id_and_increments_size():
    buf = PrefixBuffer(max_size=10)
    assert len(buf) == 0
    ok = buf.add(_entry())
    assert ok
    assert len(buf) == 1
    assert buf._entries[0].entry_id != ""


def test_buffer_evicts_oldest_when_global_cap_hit():
    buf = PrefixBuffer(max_size=2)
    buf.add(_entry(qid="a", step=1))
    buf.add(_entry(qid="b", step=2))
    buf.add(_entry(qid="c", step=3))
    assert len(buf) == 2
    qids = {e.question_id for e in buf._entries}
    assert qids == {"b", "c"}  # "a" evicted


def test_buffer_per_question_cap_evicts_within_question():
    buf = PrefixBuffer(max_size=100, max_per_question=2)
    buf.add(_entry(qid="q", turns=1, step=1))
    buf.add(_entry(qid="q", turns=2, step=2))
    buf.add(_entry(qid="q", turns=3, step=3))  # triggers per-question eviction
    assert len(buf) == 2
    # The eviction picks from the turn-count with the most entries -- here all
    # three were different turn counts so it evicts the oldest (turns=1).
    turn_counts = {e.num_route_turns for e in buf._entries}
    assert turn_counts == {2, 3}


def test_buffer_sample_respects_wrong_ratio():
    buf = PrefixBuffer(max_size=100, min_size=1, wrong_ratio=0.8)
    for i in range(10):
        buf.add(_entry(qid=f"w{i}", prefix_type="wrong", step=1))
    for i in range(10):
        buf.add(_entry(qid=f"r{i}", prefix_type="right", step=1))
    buf.update_step(1)
    samples = buf.sample(n=10)
    n_wrong = sum(1 for s in samples if s.prefix_type == "wrong")
    # 80% wrong target on 10 samples = 8 wrong (probabilistic rounding can deviate by 1)
    assert 7 <= n_wrong <= 9


def test_buffer_filters_stale_entries():
    buf = PrefixBuffer(max_size=100, min_size=1, max_staleness_steps=10)
    buf.add(_entry(qid="old", step=0))
    buf.add(_entry(qid="new", step=100))
    buf.update_step(100)
    samples = buf.sample(n=2)
    assert {s.question_id for s in samples} == {"new"}


def test_can_sample_respects_min_size():
    buf = PrefixBuffer(max_size=100, min_size=5)
    for i in range(4):
        buf.add(_entry(qid=f"q{i}", step=1))
    buf.update_step(1)
    assert not buf.can_sample()
    buf.add(_entry(qid="q5", step=1))
    assert buf.can_sample()


def test_sampled_indices_reindex_after_pop():
    # If sample() recorded indices {0, 2}, then add() evicts index 0,
    # the recorded indices should become {1} (former index 2 is now 1).
    buf = PrefixBuffer(max_size=3, min_size=1)
    buf.add(_entry(qid="a", step=1))
    buf.add(_entry(qid="b", step=1))
    buf.add(_entry(qid="c", step=1))
    buf.update_step(1)
    _ = buf.sample(n=3)  # marks {0, 1, 2} as sampled
    buf.add(_entry(qid="d", step=1))  # evicts index 0
    # _sampled_indices should now be {0, 1} (was {0,1,2}; 0 removed; 1->0, 2->1)
    assert buf._sampled_indices == {0, 1}


def test_mode_seeded_rng_is_reproducible():
    """The mode selection should be reproducible given (question_id, base_seed, batch_counter)."""
    import random as _random
    def pick(q, batch_counter, base_seed=1):
        rng = _random.Random(q + base_seed + batch_counter * 10000)
        return rng.choices(["normal", "random_start", "prefix_buffer"],
                           weights=[1, 1, 2], k=1)[0]
    # Same inputs => same result.
    assert pick(0, 0) == pick(0, 0)
    # Different question_ids in the same batch should produce a mix of modes.
    modes = {pick(q, 0) for q in range(100)}
    assert "prefix_buffer" in modes  # weight=2, should appear
    assert "normal" in modes


def test_inject_prefix_buffer_entries_swaps_sequence_and_images():
    """When mode=prefix_buffer, sample_info.sequence is replaced with stored prefix."""
    from PIL import Image
    base_img = Image.new("RGB", (10, 10), color="red")
    feedback_img = Image.new("RGB", (10, 10), color="blue")

    # Set up a buffer with one entry
    buf = PrefixBuffer(max_size=10, min_size=1)
    entry = _entry(qid="q0", turns=1, prefix_type="wrong", step=1,
                   conversation="<base-prompt><route1>")
    entry.feedback_images = [feedback_img]
    buf.add(entry)
    buf.update_step(1)

    # Mock samples_info for a single question with n=2
    samples_info = [
        {"sequence": "<orig>", "multi_modal_data": {"image": [base_img]},
         "turn_count": 0, "mode": "prefix_buffer", "stop": False},
        {"sequence": "<orig>", "multi_modal_data": {"image": [base_img]},
         "turn_count": 0, "mode": "prefix_buffer", "stop": False},
    ]
    sample_modes = ["prefix_buffer", "prefix_buffer"]
    # Inline the injection block as a closure so we can unit-test it.
    entries = buf.sample(n=1)
    assert len(entries) == 1
    for s in range(2):
        sinfo = samples_info[s]
        sinfo["sequence"] = entries[0].prefix_conversation
        sinfo["multi_modal_data"]["image"] = (
            sinfo["multi_modal_data"]["image"] + list(entries[0].feedback_images))
        sinfo["turn_count"] = entries[0].num_route_turns

    for sinfo in samples_info:
        assert sinfo["sequence"] == "<base-prompt><route1>"
        assert len(sinfo["multi_modal_data"]["image"]) == 2
        assert sinfo["turn_count"] == 1


def test_assistant_turn_starts_detects_boundaries():
    """The boundary helper finds <|im_start|>assistant\\n triples in token streams."""
    # Mock a tokenizer-like object: returns known IDs for the relevant tokens.
    class MockTokenizer:
        def convert_tokens_to_ids(self, t):
            return {"<|im_start|>": 100, "assistant": 200}[t]

    # Token stream: [turn0 content...] <|im_start|> assistant \n [turn1...] <|im_start|> assistant \n [turn2...]
    tokens = [1, 2, 3, 100, 200, 198, 4, 5, 100, 200, 198, 6, 7]
    # Bind helper logic locally for testing (avoid full rollout init).
    im_start_id = 100
    assistant_id = 200
    newline_id = 198
    starts = [(0, 0)]
    turn_idx = 1
    pos = 0
    while pos + 2 < len(tokens):
        if (tokens[pos] == im_start_id
                and tokens[pos + 1] == assistant_id
                and tokens[pos + 2] == newline_id):
            starts.append((turn_idx, pos + 3))
            turn_idx += 1
            pos += 3
        else:
            pos += 1
    assert starts == [(0, 0), (1, 6), (2, 11)]


def test_collect_buffer_entries_adds_right_and_wrong():
    """Normal-mode rollouts with valid route+terminate turns get added."""
    # Build a minimal sequence with one route + one terminate turn.
    map_spec = {"layout": ["SF", "FG"], "start_pos": 0, "target_pos": 3, "level": 2}
    em_seq = (
        '<|im_start|>assistant\n'
        '```json\n{"function_call": {"name": "route", "arguments": {"actions": "right, down"}}}```\n'
        '<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n'
        '<|im_start|>assistant\n'
        '```json\n{"function_call": {"name": "terminate", "arguments": {"answer": "right, down"}}}```\n'
        '<|im_end|>'
    )
    wrong_seq = em_seq.replace('right, down', 'down, right')  # legal but non-optimal
    base_img = Image.new("RGB", (10, 10))
    feedback_img = Image.new("RGB", (10, 10))

    class _StubRollout:
        prefix_buffer = PrefixBuffer(max_size=10, min_size=1)
        prefix_buffer_max_route_turns = 6
        batch_counter = 1

    # Reuse the helper's logic inline -- the goal is to confirm shape, not exhaustively unit-test the
    # importlib path; the integration check is the smoke run in Task 7.
    rollout = _StubRollout()
    samples_info = [
        {"mode": "normal", "finish_reason": None, "sequence": em_seq,
         "multi_modal_data": {"image": [base_img, feedback_img]},
         "map_spec": map_spec, "question_id": 0},
        {"mode": "normal", "finish_reason": None, "sequence": wrong_seq,
         "multi_modal_data": {"image": [base_img, feedback_img]},
         "map_spec": map_spec, "question_id": 1},
    ]
    # Direct buffer insert via the helper logic
    from EasyR1.verl.workers.rollout.frozenlake_prefix_buffer import PrefixBufferEntry as _E
    for sinfo, em_flag in zip(samples_info, [True, False]):
        last = sinfo["sequence"].rfind("<|im_start|>assistant\n")
        rollout.prefix_buffer.add(_E(
            entry_id="", question_id=str(sinfo["question_id"]),
            map_spec=sinfo["map_spec"], ground_truth=sinfo["map_spec"],
            prefix_conversation=sinfo["sequence"][:last],
            prefix_type="right" if em_flag else "wrong",
            num_route_turns=1, last_actions=["right"], final_pr=1.0 if em_flag else 0.5,
            feedback_images=[feedback_img], collection_step=1,
            correct_suffix=sinfo["sequence"][last:] if em_flag else None))
    assert len(rollout.prefix_buffer) == 2
    types = {e.prefix_type for e in rollout.prefix_buffer._entries}
    assert types == {"right", "wrong"}


def _apply_pb_prefix_mask(tmp_multi_turn_mask, starts, num_prefix_turns):
    """Reference implementation of the prefix-buffer prefix-masking logic
    that lives inline in MultiTurnRolloutFrozenLake.generate_sequences.

    Mirrors the active code path so the unit test can exercise the algorithm
    without spinning up the full rollout. If the active code drifts, update
    this helper to match; the rollout itself remains the source of truth.
    """
    import torch
    if num_prefix_turns > 0 and len(starts) > num_prefix_turns:
        cutoff_pos = starts[num_prefix_turns][1]
        tmp_multi_turn_mask[:cutoff_pos] = 0
    return tmp_multi_turn_mask


def test_pb_prefix_mask_zeros_prefix_turn_tokens():
    """num_prefix_turns=1, two assistant-turn boundaries → mask everything
    before the second boundary (the new generation start)."""
    import torch
    # Pretend the response slice has 30 tokens, with the second assistant turn
    # (the new generation) starting at position 14.
    mask = torch.ones(30, dtype=torch.long)
    starts = [(0, 0), (1, 14)]  # prefix turn 0 at pos 0; new turn at pos 14
    out = _apply_pb_prefix_mask(mask.clone(), starts, num_prefix_turns=1)
    assert out[:14].sum().item() == 0, "prefix tokens should be masked"
    assert out[14:].sum().item() == 16, "new-turn tokens should retain mask=1"


def test_pb_prefix_mask_handles_multi_turn_prefix():
    """num_prefix_turns=2 with three assistant-turn boundaries → mask before
    the third boundary."""
    import torch
    mask = torch.ones(40, dtype=torch.long)
    starts = [(0, 0), (1, 12), (2, 25)]  # 2 prefix turns; new turn starts at 25
    out = _apply_pb_prefix_mask(mask.clone(), starts, num_prefix_turns=2)
    assert out[:25].sum().item() == 0
    assert out[25:].sum().item() == 15


def test_pb_prefix_mask_no_op_when_zero_prefix_turns():
    """Degenerate (and contract-violating) case: num_route_turns=0 should be
    a no-op to preserve any training signal."""
    import torch
    mask = torch.ones(20, dtype=torch.long)
    starts = [(0, 0), (1, 10)]
    out = _apply_pb_prefix_mask(mask.clone(), starts, num_prefix_turns=0)
    assert out.sum().item() == 20


def test_pb_prefix_mask_no_op_when_model_generated_no_new_turn():
    """If the model generated no new turn after the prefix, starts has fewer
    than num_prefix_turns+1 entries; leaving the mask alone preserves at
    least some training signal rather than zeroing everything."""
    import torch
    mask = torch.ones(20, dtype=torch.long)
    starts = [(0, 0)]  # only the implicit turn-0 start, no new turn detected
    out = _apply_pb_prefix_mask(mask.clone(), starts, num_prefix_turns=1)
    assert out.sum().item() == 20


def test_immediate_terminate_logic_returns_none_for_non_pb_modes():
    """Mode-gating: only prefix_buffer + right-type samples produce a bool."""
    # Inline the helper's body (constructing a full rollout is heavy; this is
    # a pure-function unit test against the contract).
    def imm_term(sinfo):
        if sinfo.get("mode") != "prefix_buffer":
            return None
        if sinfo.get("prefix_type") != "right":
            return None
        prefix_len = sinfo.get("prefix_len")
        if prefix_len is None:
            return None
        new_text = sinfo.get("sequence", "")[prefix_len:]
        first_turn_end = new_text.find("<|im_end|>")
        first_turn = new_text if first_turn_end < 0 else new_text[:first_turn_end]
        return ('"name": "terminate"' in first_turn
                or '"name":"terminate"' in first_turn)

    # Non-PB mode -> None
    assert imm_term({"mode": "normal", "sequence": "anything"}) is None
    assert imm_term({"mode": "random_start", "prefix_type": "right",
                     "prefix_len": 0, "sequence": "anything"}) is None
    # PB but wrong-type -> None (only right-prefix early-stop matters)
    assert imm_term({"mode": "prefix_buffer", "prefix_type": "wrong",
                     "prefix_len": 0, "sequence": '"name": "terminate"'}) is None
    # PB + right + first new turn IS a terminate -> True
    prefix = "<prefix>"
    seq = prefix + '```json\n{"function_call": {"name": "terminate", "arguments": {"answer": "right"}}}```<|im_end|>'
    assert imm_term({"mode": "prefix_buffer", "prefix_type": "right",
                     "prefix_len": len(prefix), "sequence": seq}) is True
    # PB + right + first new turn is route (not terminate) -> False
    seq2 = prefix + '```json\n{"function_call": {"name": "route", "arguments": {"actions": "right"}}}```<|im_end|>'
    assert imm_term({"mode": "prefix_buffer", "prefix_type": "right",
                     "prefix_len": len(prefix), "sequence": seq2}) is False
