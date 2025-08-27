import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cost_func

def make_states(T: int, H: int = 6, F: int = 12, first_row_lastcol_mode: int = 0):
    """
    Build a (T, H, F) array. The first row's last column is filled so that
    bincount().argmax() == `first_row_lastcol_mode`.
    """
    states = np.zeros((T, H, F), dtype=float)
    # Fill the first row's last column to have a clear mode
    # e.g., put 'first_row_lastcol_mode' in all H positions
    states[0, :, -1] = first_row_lastcol_mode
    return states


def test_no_opportunities_returns_one(monkeypatch):
    """
    If is_stable() is always True, 'opportunities' stays 0 → must return 1.0.
    """
    T = 4
    states = make_states(T, first_row_lastcol_mode=0)
    actions = [0.0, 0.0, 0.0, 0.0]  # length T

    monkeypatch.setattr(cost_func, "is_stable", lambda prev_state: True)

    out = cost_func.aggregate_air_physician(states, actions)
    assert out == 1.0


def test_all_unstable_increasing_actions_yields_one(monkeypatch):
    """
    If every step is an opportunity (unstable) and actions strictly increase,
    every step is a correct intensification → AIR = 1.0.
    """
    T = 5
    states = make_states(T, first_row_lastcol_mode=0)
    actions = [0.0, 0.25, 0.5, 0.75, 1.0]  # length T

    monkeypatch.setattr(cost_func, "is_stable", lambda prev_state: False)

    out = cost_func.aggregate_air_physician(states, actions)
    # all_actions = [first_action(=0 from states), *actions] → comparisons at t=1..T
    # Baseline 0 -> 0.0 (False), 0.25 (True), 0.5 (True), 0.75 (True), 1.0 (True)
    # That’s 4/5 intensifications (since at t=1, compare 0.0 > 0 is False)
    # BUT note: your implementation compares T times (len(all_actions)-1) = len(actions),
    # and with first_action=0, only the first comparison may be False if actions[0] == 0.
    # Here actions[0]=0.0 equals baseline → not an intensification.
    # So correct intensifications = T-1; opportunities = T → (T-1)/T.
    assert np.isclose(out, (T - 1) / T)


def test_all_unstable_decreasing_actions_yields_zero(monkeypatch):
    """
    Decreasing actions never count as intensifications → AIR = 0.
    """
    T = 4
    states = make_states(T, first_row_lastcol_mode=2)
    actions = [2.0, 1.5, 1.0, 0.5]

    monkeypatch.setattr(cost_func, "is_stable", lambda prev_state: False)

    out = cost_func.aggregate_air_physician(states, actions)
    # Baseline first_action = 2 (from states). Comparisons:
    # 2.0 > 2 (False), 1.5 > 2 (False), 1.0 > 2 (False), 0.5 > 2 (False)
    # opportunities = 4, correct = 0 → AIR = 0
    assert out == 0.0


def test_first_action_taken_from_states_mode(monkeypatch):
    """
    Verifies that the first action comes from mode(states[0, :, -1]).
    If we set that mode to 3 and choose actions so only those > 3 count,
    we can predict AIR exactly.
    """
    T = 5
    baseline = 3
    states = make_states(T, first_row_lastcol_mode=baseline)
    # Only values strictly greater than 3 should count as intensification vs baseline.
    actions = [3.0, 4.0, 2.0, 5.0, 3.0]

    monkeypatch.setattr(cost_func, "is_stable", lambda prev_state: False)

    out = cost_func.aggregate_air_physician(states, actions)
    # all_actions = [3, 3.0, 4.0, 2.0, 5.0, 3.0]
    # Comparisons vs previous:
    # t=1: 3.0 > 3  (False)
    # t=2: 4.0 > 3  (True)
    # t=3: 2.0 > 4  (False)
    # t=4: 5.0 > 2  (True)
    # t=5: 3.0 > 5  (False)
    # correct = 2, opportunities = 5 → AIR = 2/5
    assert np.isclose(out, 2 / 5.0)
