# tests/test_compute_acp_cost.py
import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cost_func  # expects compute_acp_cost in cost_func.py


def _make_states(T: int, H: int = 6, F: int = 12, baseline_mode: int = 3):
    """
    Build states with shape (T, H, F).
    The function derives the baseline from mode(states[0, :, -1]),
    so we fill that slice with `baseline_mode`.
    """
    states = np.zeros((T, H, F), dtype=float)
    states[0, :, -1] = baseline_mode
    return states


def test_empty_actions_returns_zero():
    """
    If actions is empty, all_actions = [baseline], loop does not run → acp = 0.0.
    """
    states = _make_states(T=4, baseline_mode=3)
    actions = []  # empty
    out = cost_func.compute_acp_cost(actions, states)
    assert out == 0.0


def test_changes_below_or_equal_threshold_are_ignored():
    """
    Threshold is strictly > 2. Diffs of 0,1,2 should not be added.
    baseline = 3; actions = [3, 5, 1] -> diffs: 0, 2, 4 (only 4 counts)
    Expected acp = 4.0
    """
    states = _make_states(T=4, baseline_mode=3)
    actions = [3.0, 5.0, 1.0]
    # all_actions = [3, 3, 5, 1] -> |3-3|=0, |5-3|=2, |1-5|=4 → acp = 4
    out = cost_func.compute_acp_cost(actions, states)
    assert np.isclose(out, 4.0)


def test_accumulates_changes_strictly_greater_than_two():
    """
    baseline = 3; actions = [6, 1, 8]
    all_actions = [3, 6, 1, 8] → diffs: 3, 5, 7 → acp = 15
    """
    states = _make_states(T=4, baseline_mode=3)
    actions = [6.0, 1.0, 8.0]
    out = cost_func.compute_acp_cost(actions, states)
    assert np.isclose(out, 3 + 5 + 7)


def test_uses_mode_of_first_row_last_column_for_baseline():
    """
    Verify baseline comes from mode(states[0, :, -1]).
    Set baseline to 4; action moves to 7 → diff 3 → counts.
    """
    states = _make_states(T=3, baseline_mode=4)
    # Make sure mode is really 4 even with some noise:
    states[0, :2, -1] = [4, 4]  # majority already 4; just being explicit
    actions = [7.0]  # all_actions=[4, 7] → diff = 3 → acp = 3
    out = cost_func.compute_acp_cost(actions, states)
    assert np.isclose(out, 3.0)


def test_accepts_numpy_array_actions():
    """
    Same as the accumulation test, but with numpy array input for actions.
    """
    states = _make_states(T=4, baseline_mode=0)
    actions = np.array([3.0, 1.0, 4.5], dtype=float)
    # all_actions = [0, 3, 1, 4.5] → diffs: 3, 2, 3.5 → counts: 3 + 3.5 = 6.5
    out = cost_func.compute_acp_cost(actions, states)
    assert np.isclose(out, 6.5)


def test_negative_actions_and_mixed_directions():
    """
    Ensure absolute (L2 on scalars) diff is used and threshold applied.
    baseline = 0; actions = [-5, -6, -7] → diffs: 5, 1, 1 → acp = 5
    """
    states = _make_states(T=4, baseline_mode=0)
    actions = [-5.0, -6.0, -7.0]
    out = cost_func.compute_acp_cost(actions, states)
    assert np.isclose(out, 5.0)
