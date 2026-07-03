# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prefix Buffer for on-policy prefix replay in GRPO training.

This module implements a buffer that stores on-policy prefixes from previous rollouts.
These prefixes are used to train the model on:
1. Correction (wrong prefixes): Learning to correct mistakes at any turn
2. Early stopping (right prefixes): Learning when to terminate

The buffer maintains diversity across:
- Different questions
- Different prefix lengths (number of pointing turns)
- Fresh entries (staleness-based eviction)
"""

import random
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image


@dataclass
class PrefixBufferEntry:
    """A single entry in the prefix buffer."""
    
    # Unique identifier for this entry
    entry_id: str
    
    # Question identification
    question_id: str
    
    # Image information
    original_image_path: str
    
    # Ground truth for the question
    ground_truth: Any
    
    # The prefix conversation (multi-turn, WITHOUT final terminate call)
    # This is the full sequence string up to but not including the terminate turn
    prefix_conversation: str
    
    # Type of prefix: "wrong" (terminate was incorrect) or "right" (terminate was correct)
    prefix_type: str  # "wrong" or "right"
    
    # Number of pointing turns in this prefix (1, 2, 3, ...)
    num_pointing_turns: int
    
    # The final pointing coordinate before terminate was called
    final_point: Tuple[int, int]
    
    # Distance from final point to ground truth
    final_distance: float
    
    # Paths to crop images generated during the prefix turns
    crop_paths: List[str]
    
    # The global step when this entry was collected (for staleness tracking)
    collection_step: int
    
    # For "right" entries: the original terminate call content (suffix after the prefix).
    # Used by correct_replay to inject the original correct termination as one of N rollouts.
    correct_suffix: Optional[str] = None
    
    # Multi-modal data (images list including original + crops)
    multi_modal_images: List[Any] = field(default_factory=list)
    
    # Task type (if applicable)
    task: Optional[str] = None


class PrefixBuffer:
    """
    A fixed-size buffer that stores on-policy prefixes for GRPO training.
    
    Features:
    - FIFO eviction when buffer is full
    - Staleness-based filtering (old entries are not sampled)
    - Stratified sampling by prefix type (wrong/right) and number of turns
    - Diversity constraints per question
    - Sampling without replacement within an epoch
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        wrong_ratio: float = 0.8,
        max_staleness_steps: int = 50,
        min_size: int = 32,
        max_per_question: int = 3,
    ):
        """
        Initialize the prefix buffer.
        
        Args:
            max_size: Maximum number of entries in the buffer
            wrong_ratio: Ratio of wrong prefixes to sample (0.8 = 80% wrong, 20% right)
            max_staleness_steps: Maximum steps before an entry is considered stale
            min_size: Minimum buffer size before sampling is enabled
            max_per_question: Maximum entries per question (ensures question diversity).
                             Within a question, prefers turn-count diversity: when the cap
                             is hit, evicts the oldest entry from the most over-represented
                             turn count for that question.
        """
        self.max_size = max_size
        self.wrong_ratio = wrong_ratio
        self.max_staleness_steps = max_staleness_steps
        self.min_size = min_size
        self.max_per_question = max_per_question
        
        # Main storage: list of entries (FIFO order)
        self.entries: List[PrefixBufferEntry] = []
        
        # Index structures for efficient lookup
        self._wrong_entries: List[int] = []  # Indices of wrong entries
        self._right_entries: List[int] = []  # Indices of right entries
        self._entries_by_turns: Dict[int, List[int]] = defaultdict(list)  # turn_count -> indices
        self._entries_by_question: Dict[str, List[int]] = defaultdict(list)  # question_id -> indices
        
        # Sampling state (reset each epoch)
        self._sampled_indices: set = set()
        
        # Entry ID counter
        self._entry_counter = 0
        
        # Current global step (updated externally)
        self.current_step = 0
        
    def _rebuild_indices(self):
        """Rebuild all index structures from scratch."""
        self._wrong_entries = []
        self._right_entries = []
        self._entries_by_turns = defaultdict(list)
        self._entries_by_question = defaultdict(list)
        
        for idx, entry in enumerate(self.entries):
            if entry.prefix_type == "wrong":
                self._wrong_entries.append(idx)
            else:
                self._right_entries.append(idx)
            self._entries_by_turns[entry.num_pointing_turns].append(idx)
            self._entries_by_question[entry.question_id].append(idx)
    
    def add(self, entry: PrefixBufferEntry) -> bool:
        """
        Add an entry to the buffer with per-question diversity enforcement.
        
        When a question already has max_per_question entries:
        - If the new entry brings a turn count not yet represented for this
          question, evict the oldest entry from the most over-represented turn
          count for this question.
        - Otherwise, evict the oldest entry with the same turn count for this
          question.
        This keeps a diverse mix of prefix lengths per question.
        
        Args:
            entry: The prefix buffer entry to add
            
        Returns:
            True if the entry was added, False if it was rejected
        """
        # Assign entry ID
        entry.entry_id = f"prefix_{self._entry_counter}"
        self._entry_counter += 1
        
        # Enforce per-question cap
        q_id = entry.question_id
        q_indices = self._entries_by_question.get(q_id, [])
        if len(q_indices) >= self.max_per_question:
            # Decide which existing entry to evict for this question
            evict_idx = self._pick_question_evict_idx(q_id, entry.num_pointing_turns)
            if evict_idx is not None:
                self._remove_entry(evict_idx)
            else:
                # Shouldn't happen, but fall back to rejecting
                return False
        
        # If buffer is full globally, remove oldest entry (FIFO)
        if len(self.entries) >= self.max_size:
            self._remove_oldest()
        
        # Add the new entry
        idx = len(self.entries)
        self.entries.append(entry)
        
        # Update indices
        if entry.prefix_type == "wrong":
            self._wrong_entries.append(idx)
        else:
            self._right_entries.append(idx)
        self._entries_by_turns[entry.num_pointing_turns].append(idx)
        self._entries_by_question[entry.question_id].append(idx)
        
        return True
    
    def _pick_question_evict_idx(self, question_id: str, new_turn_count: int) -> Optional[int]:
        """
        Pick which entry to evict when a question has hit max_per_question.
        
        Strategy:
        - Count how many entries each turn_count has for this question.
        - If the new turn_count is NOT yet represented → evict the oldest
          entry from the turn count that has the most entries (most
          over-represented).
        - If the new turn_count IS already represented → evict the oldest
          entry with the same turn count for this question.
        """
        q_indices = self._entries_by_question.get(question_id, [])
        if not q_indices:
            return None
        
        # Group existing entries for this question by turn count
        turn_count_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx in q_indices:
            if idx < len(self.entries):
                turn_count_to_indices[self.entries[idx].num_pointing_turns].append(idx)
        
        if new_turn_count not in turn_count_to_indices:
            # New turn count → evict oldest from the most over-represented turn count
            most_common_turn = max(turn_count_to_indices, key=lambda t: len(turn_count_to_indices[t]))
            # Pick the oldest (lowest index = earliest added) in that group
            return min(turn_count_to_indices[most_common_turn])
        else:
            # Same turn count exists → evict oldest with that turn count
            return min(turn_count_to_indices[new_turn_count])
    
    def _remove_entry(self, idx: int):
        """Remove a specific entry by index and rebuild indices."""
        if idx < 0 or idx >= len(self.entries):
            return
        self.entries.pop(idx)
        self._adjust_sampled_indices_after_pop(idx)
        self._rebuild_indices()
    
    def _remove_oldest(self):
        """Remove the oldest entry from the buffer."""
        if not self.entries:
            return
        
        # Remove from main storage
        self.entries.pop(0)
        self._adjust_sampled_indices_after_pop(0)
        
        # Rebuild indices (simpler than updating all indices)
        self._rebuild_indices()
    
    def _adjust_sampled_indices_after_pop(self, removed_idx: int):
        """Adjust _sampled_indices after an entry at removed_idx was popped.
        
        When entries.pop(removed_idx) is called, every entry that was at a
        position > removed_idx shifts down by 1.  We must mirror that shift
        in _sampled_indices so the set still points at the correct entries.
        The removed index itself is dropped.
        """
        new_sampled = set()
        for s_idx in self._sampled_indices:
            if s_idx < removed_idx:
                new_sampled.add(s_idx)
            elif s_idx > removed_idx:
                new_sampled.add(s_idx - 1)
            # s_idx == removed_idx → entry was removed, drop it
        self._sampled_indices = new_sampled
    
    def _get_fresh_indices(self, indices: List[int]) -> List[int]:
        """Filter indices to only include fresh (non-stale) entries."""
        fresh = []
        for idx in indices:
            if idx < len(self.entries):
                entry = self.entries[idx]
                staleness = self.current_step - entry.collection_step
                if staleness <= self.max_staleness_steps:
                    fresh.append(idx)
        return fresh
    
    def _get_unsampled_indices(self, indices: List[int]) -> List[int]:
        """Filter indices to only include unsampled entries."""
        return [idx for idx in indices if idx not in self._sampled_indices]
    
    def sample(self, n: int = 1) -> List[PrefixBufferEntry]:
        """
        Sample entries from the buffer without replacement.
        
        Sampling strategy:
        1. Split sampling between wrong (wrong_ratio) and right (1-wrong_ratio) entries
        2. Within each type, prefer diversity in num_pointing_turns
        3. Only sample fresh (non-stale) entries
        4. Sample without replacement within an epoch
        
        Args:
            n: Number of entries to sample
            
        Returns:
            List of sampled entries
        """
        if len(self.entries) < self.min_size:
            return []
        
        # Calculate how many wrong vs right to sample
        # When n is small (e.g. n=1), int() truncation would always round down
        # to 0 wrong samples.  Use probabilistic rounding so that e.g. with
        # wrong_ratio=0.8 and n=1, we pick a wrong entry 80% of the time.
        n_wrong_float = n * self.wrong_ratio
        n_wrong_floor = int(n_wrong_float)
        frac = n_wrong_float - n_wrong_floor
        if frac > 0 and random.random() < frac:
            n_wrong = n_wrong_floor + 1
        else:
            n_wrong = n_wrong_floor
        n_right = n - n_wrong
        
        sampled_indices: List[int] = []
        sampled_entries: List[PrefixBufferEntry] = []
        
        # Sample wrong entries
        wrong_fresh = self._get_fresh_indices(self._wrong_entries)
        wrong_available = self._get_unsampled_indices(wrong_fresh)
        w_idx, w_ent = self._stratified_sample(wrong_available, n_wrong)
        sampled_indices.extend(w_idx)
        sampled_entries.extend(w_ent)
        
        # Sample right entries
        right_fresh = self._get_fresh_indices(self._right_entries)
        right_available = self._get_unsampled_indices(right_fresh)
        r_idx, r_ent = self._stratified_sample(right_available, n_right)
        sampled_indices.extend(r_idx)
        sampled_entries.extend(r_ent)
        
        # If we couldn't get enough from one type, try to get more from the other
        shortage = n - len(sampled_entries)
        if shortage > 0:
            # Try to fill from whichever type has more available
            already_sampled = set(sampled_indices)
            all_fresh = self._get_fresh_indices(list(range(len(self.entries))))
            all_available = self._get_unsampled_indices(all_fresh)
            remaining_available = [idx for idx in all_available if idx not in already_sampled]
            a_idx, a_ent = self._stratified_sample(remaining_available, shortage)
            sampled_indices.extend(a_idx)
            sampled_entries.extend(a_ent)
        
        # Mark as sampled
        for idx in sampled_indices:
            self._sampled_indices.add(idx)
        
        return sampled_entries
    
    def _stratified_sample(self, available_indices: List[int], n: int) -> Tuple[List[int], List[PrefixBufferEntry]]:
        """
        Sample n entries with stratification by num_pointing_turns.
        
        This ensures we get a diverse mix of prefix lengths.
        
        Returns:
            Tuple of (sampled_indices, sampled_entries)
        """
        if not available_indices or n <= 0:
            return [], []
        
        # Group available indices by num_pointing_turns
        by_turns: Dict[int, List[int]] = defaultdict(list)
        for idx in available_indices:
            if idx < len(self.entries):
                turns = self.entries[idx].num_pointing_turns
                by_turns[turns].append(idx)
        
        if not by_turns:
            return [], []
        
        # Round-robin sample from each turn count.
        # Start at a random position so that every turn count gets a fair
        # chance when n is small (especially n=1, which is the common case
        # when sampling one prefix per question).
        sampled_indices: List[int] = []
        sampled_entries: List[PrefixBufferEntry] = []
        turn_counts = sorted(by_turns.keys())
        turn_idx = random.randint(0, len(turn_counts) - 1)
        
        while len(sampled_entries) < n and any(by_turns.values()):
            turn_count = turn_counts[turn_idx % len(turn_counts)]
            if by_turns[turn_count]:
                idx = random.choice(by_turns[turn_count])
                by_turns[turn_count].remove(idx)
                if idx < len(self.entries):
                    sampled_indices.append(idx)
                    sampled_entries.append(self.entries[idx])
            turn_idx += 1
            
            # Clean up empty lists
            turn_counts = [t for t in turn_counts if by_turns[t]]
            if not turn_counts:
                break
        
        return sampled_indices, sampled_entries
    
    def reset_sampling(self):
        """Reset the sampling state for a new epoch."""
        self._sampled_indices = set()
    
    def update_step(self, step: int):
        """Update the current global step."""
        self.current_step = step
    
    def can_sample(self) -> bool:
        """Check if the buffer has enough entries for sampling."""
        if len(self.entries) < self.min_size:
            return False
        
        # Check if there are fresh, unsampled entries
        all_fresh = self._get_fresh_indices(list(range(len(self.entries))))
        all_available = self._get_unsampled_indices(all_fresh)
        return len(all_available) > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics for logging.
        
        Returns:
            Dict with buffer statistics
        """
        stats = {
            "buffer_size": len(self.entries),
            "buffer_capacity": self.max_size,
            "buffer_utilization": len(self.entries) / self.max_size if self.max_size > 0 else 0,
        }
        
        # Count by prefix type
        n_wrong = len(self._wrong_entries)
        n_right = len(self._right_entries)
        stats["buffer_wrong_count"] = n_wrong
        stats["buffer_right_count"] = n_right
        stats["buffer_wrong_ratio"] = n_wrong / len(self.entries) if self.entries else 0
        
        # Count by number of pointing turns
        turn_counts = defaultdict(int)
        for entry in self.entries:
            turn_counts[entry.num_pointing_turns] += 1
        
        total = len(self.entries) if self.entries else 1
        for turns in sorted(turn_counts.keys()):
            stats[f"buffer_turns_{turns}_count"] = turn_counts[turns]
            stats[f"buffer_turns_{turns}_pct"] = turn_counts[turns] / total * 100
        
        # Fresh entries count
        all_fresh = self._get_fresh_indices(list(range(len(self.entries))))
        stats["buffer_fresh_count"] = len(all_fresh)
        stats["buffer_fresh_pct"] = len(all_fresh) / total * 100 if self.entries else 0
        
        # Unsampled entries count
        unsampled = self._get_unsampled_indices(list(range(len(self.entries))))
        stats["buffer_unsampled_count"] = len(unsampled)
        
        # Unique questions count
        stats["buffer_unique_questions"] = len(self._entries_by_question)
        
        # Per-question stats
        if self._entries_by_question:
            entries_per_q = [len(v) for v in self._entries_by_question.values()]
            stats["buffer_avg_per_question"] = sum(entries_per_q) / len(entries_per_q)
            stats["buffer_max_per_question"] = max(entries_per_q)
        
        return stats
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __repr__(self) -> str:
        return f"PrefixBuffer(size={len(self.entries)}/{self.max_size}, wrong={len(self._wrong_entries)}, right={len(self._right_entries)})"

