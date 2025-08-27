import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cost_func  


def _make_states(T: int, H: int = 6, F: int = 12, baseline_val: int = 3):
    """
    Build states with shape (T, H, F).
    The function uses states[0, :, -1] to compute the baseline via mode, so we fill it.
    """
    states = np.zeros((T, H, F), dtype=float)
    states[0, :, -1] = baseline_val
    return states


def test_no_stable_hours_returns_zero(monkeypatch):
    """
    If is_stable() is always False → denom stays 0 → function must return 0.0.
    """
    T = 4
    states = _make_states(T, baseline_val=3)
    actions = [0.0, 1.0, 1.0, 2.0]  # any values; no step counts toward denom

    monkeypatch.setattr(cost_func, "is_stable", lambda prev_state: False)

    out = cost_func.weaning_score_physician(states, actions)
    assert out == 0.0


def test_all_stable_mixed_changes_correct_average(monkeypatch):
    """
    All steps stable:
      - Proper weaning: decrease by 1 or 2 → add that amount
      - Improper: any increase → subtract 1
      - No change or decrease by >=3 → 0
    Example:
      baseline = 3 (from states[0, :, -1])
      actions = [2, 1, 0, 3]
      Comparisons (with all_actions = [3, *actions]):
        t=1: 2 vs 3 → -1  => +1
        t=2: 1 vs 2 → -1  => +1
        t=3: 0 vs 1 → -1  => +1
        t=4: 3 vs 0 → +3  => -1  (increase)
      total score = 1+1+1-1 = 2 ; denom = 4 → 0.5
    """
    T = 4
    states = _make_states(T, baseline_val=3)
    actions = [2.0, 1.0, 0.0, 3.0]

    monkeypatch.setattr(cost_func, "is_stable", lambda prev_state: True)

    out = cost_func.weaning_score_physician(states, actions)
    assert np.isclose(out, 0.5)


def test_all_stable_increase_and_large_drop_behavior(monkeypatch):
    """
    Check that increases are penalized (-1) and a large drop (>=3) is ignored (0).
    Example:
      baseline = 3
      actions  = [3, 4, 2]
      Steps:
        t=1: 3 vs 3 → 0      => 0
        t=2: 4 vs 3 → +1     => -1 (penalty)
        t=3: 2 vs 4 → -2     => +2 (proper weaning by 2)
      total = -1 + 2 = +1 ; denom = 3 → 1/3
    """
    T = 3
    states = _make_states(T, baseline_val=3)
    actions = [3.0, 4.0, 2.0]

    monkeypatch.setattr(cost_func, "is_stable", lambda prev_state: True)

    out = cost_func.weaning_score_physician(states, actions)
    assert np.isclose(out, 1.0 / 3.0)


def test_uses_mode_of_first_row_last_column(monkeypatch):
    """
    Verify the first action baseline comes from mode(states[0, :, -1]).
    Set baseline to 5 and craft actions to get a predictable score.
      baseline = 5
      actions  = [5, 6, 4]
      Steps:
        t=1: 5 vs 5 → 0    => 0
        t=2: 6 vs 5 → +1   => -1
        t=3: 4 vs 6 → -2   => +2
      total = +1 ; denom = 3 → 1/3
    """
    T = 3
    states = _make_states(T, baseline_val=5)
    actions = [5.0, 6.0, 4.0]

    monkeypatch.setattr(cost_func, "is_stable", lambda prev_state: True)

    out = cost_func.weaning_score_physician(states, actions)
    assert np.isclose(out, 1.0 / 3.0)


def test_mixed_stability_affects_denominator(monkeypatch):
    """
    Make only some steps stable to ensure denom counts only those.
    We'll tag each prev_state with an index in states[t-1, 0, 0] so our patched
    is_stable can decide per-step.
      baseline = 3
      actions  = [3, 4, 2, 2]
      Stability per t (1..4): [False, True, False, True]
      Computed only on stable steps:
        t=2: 4 vs 3 → +1  => -1
        t=4: 2 vs 2 → 0   => 0
      total = -1 ; denom = 2 → -0.5
    """
    T = 4
    states = _make_states(T, baseline_val=3)

    # Tag each previous state (for t=1..T) with t in [0,0].
    # For t=1, prev_state is states[0], we'll store marker=1 in states[0,0,0], etc.
    for t in range(1, T + 1):
        states[t - 1, 0, 0] = t

    actions = [3.0, 4.0, 2.0, 2.0]

    stable_map = {1: False, 2: True, 3: False, 4: True}

    def patched_is_stable(prev_state):
        idx = int(prev_state[0, 0])
        return stable_map.get(idx, True)

    monkeypatch.setattr(cost_func, "is_stable", patched_is_stable)

    out = cost_func.weaning_score_physician(states, actions)
    assert np.isclose(out, -0.5)
