"""Tests for poker_env.observation: encode_state, hand strength, space bounds."""
import numpy as np
import pytest
from poker_env.observation import encode_state, get_observation_space, OBSERVATION_SHAPE


def _make_state(
    player_id: str = "player_0",
    hole: list = None,
    board: list = None,
    phase: str = "Pre-Flop",
    chips: int = 980,
    bet: int = 20,
    pot: int = 30,
    current_bet: int = 20,
):
    """Build a minimal valid game-state dict."""
    if hole is None:
        hole = ["A♠", "K♥"]
    if board is None:
        board = []
    return {
        "players": [
            {
                "id": player_id,
                "chips": chips,
                "hole": hole,
                "bet": bet,
                "folded": False,
                "all_in": False,
                "sitting_out": False,
                "seat": 0,
                "is_small_blind": False,
                "is_big_blind": True,
            }
        ],
        "board": board,
        "phase": phase,
        "pot": pot,
        "current_bet": current_bet,
        "active_idx": 0,
        "dealer_idx": 1,
        "small_blind": 10,
        "big_blind": 20,
        "starting_chips": 1000,
    }


def test_encode_state_shape():
    obs = encode_state(_make_state(), "player_0")
    assert obs.shape == OBSERVATION_SHAPE
    assert obs.dtype == np.float32


def test_encode_state_unknown_player_returns_zeros():
    obs = encode_state(_make_state(), "ghost_player")
    assert np.all(obs == 0.0)


def test_encode_state_within_observation_space_preflop():
    space = get_observation_space()
    obs = encode_state(_make_state(), "player_0")
    assert space.contains(obs), (
        f"obs outside space bounds: min={obs.min():.4f}, max={obs.max():.4f}"
    )


def test_encode_state_within_observation_space_postflop():
    space = get_observation_space()
    state = _make_state(board=["2♠", "7♥", "K♦"], phase="Flop")
    obs = encode_state(state, "player_0")
    assert space.contains(obs), (
        f"obs outside space bounds (postflop): min={obs.min():.4f}, max={obs.max():.4f}"
    )


def test_hand_strength_in_range_preflop():
    obs = encode_state(_make_state(hole=["A♠", "K♥"]), "player_0")
    assert 0.0 <= float(obs[63]) <= 1.0, f"hand_strength={obs[63]}"


def test_hand_strength_in_range_postflop():
    state = _make_state(board=["2♠", "7♥", "K♦"], phase="Flop")
    obs = encode_state(state, "player_0")
    assert 0.0 <= float(obs[63]) <= 1.0, f"hand_strength={obs[63]}"


def test_hand_strength_ak_on_k72_is_strong():
    """AK on K-7-2 rainbow board should be a strong hand (top pair, top kicker)."""
    state = _make_state(hole=["A♠", "K♥"], board=["K♦", "7♣", "2♥"], phase="Flop")
    obs = encode_state(state, "player_0")
    assert obs[63] > 0.5, f"Expected strong hand, got hand_strength={obs[63]:.4f}"


def test_hand_strength_72o_preflop_is_weak():
    """7-2 offsuit preflop should be below 0.5."""
    state = _make_state(hole=["7♠", "2♥"])
    obs = encode_state(state, "player_0")
    assert obs[63] < 0.5, f"Expected weak hand, got hand_strength={obs[63]:.4f}"


def test_hand_strength_pair_preflop_bonus():
    """Pocket aces preflop should score higher than 72o."""
    state_aces = _make_state(hole=["A♠", "A♥"])
    state_72o  = _make_state(hole=["7♠", "2♥"])
    hs_aces = encode_state(state_aces, "player_0")[63]
    hs_72o  = encode_state(state_72o, "player_0")[63]
    assert hs_aces > hs_72o, f"AA ({hs_aces:.4f}) should beat 72o ({hs_72o:.4f}) preflop"


@pytest.mark.parametrize("phase,expected_idx", [
    ("Pre-Flop", 23),
    ("Flop",     24),
    ("Turn",     25),
    ("River",    26),
    ("Showdown", 27),
])


def test_phase_one_hot(phase, expected_idx):
    state = _make_state(phase=phase)
    obs = encode_state(state, "player_0")
    assert obs[expected_idx] == 1.0, f"Expected obs[{expected_idx}]=1 for phase={phase}"
    assert obs[23:28].sum() == 1.0, f"Phase one-hot sum != 1: {obs[23:28]}"
