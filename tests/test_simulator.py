import pytest
from adapters.simulator.sim_env import SimulatorEnv
from poker_env.observation import OBSERVATION_SHAPE


def test_reset_returns_correct_obs_shape():
    env = SimulatorEnv(num_players=2)
    obs, info = env.reset()
    assert obs.shape == OBSERVATION_SHAPE, f"Unexpected obs shape: {obs.shape}"
    assert "game_state" in info


def test_reset_game_state_has_required_keys():
    env = SimulatorEnv(num_players=2)
    _, info = env.reset()
    state = info["game_state"]
    for key in ("players", "board", "phase", "pot", "current_bet", "dealer_idx"):
        assert key in state, f"Missing key '{key}' in game_state"


def test_reset_agent_receives_two_hole_cards():
    env = SimulatorEnv(num_players=2)
    _, info = env.reset()
    players = info["game_state"]["players"]
    agent = next(p for p in players if p["id"] == "player_0")
    assert len(agent["hole"]) == 2, "Agent should have exactly 2 hole cards"


@pytest.mark.parametrize("action", [0, 1, 2, 3, 4])
def test_step_all_actions_do_not_crash(action):
    env = SimulatorEnv(num_players=2)
    obs, _ = env.reset()
    obs, reward, done, truncated, info = env.step(action)
    assert obs.shape == OBSERVATION_SHAPE
    assert isinstance(float(reward), float)
    assert isinstance(done, bool)
    assert not truncated


@pytest.mark.parametrize("num_players", [2, 4, 6])
def test_multi_episode_no_crash(num_players):
    env = SimulatorEnv(num_players=num_players)
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            obs, reward, done, truncated, info = env.step(env.action_space.sample())
            assert obs.shape == OBSERVATION_SHAPE
            steps += 1


def test_pot_never_negative():
    env = SimulatorEnv(num_players=4)
    for _ in range(5):
        env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            _, _, done, _, info = env.step(env.action_space.sample())
            assert info["game_state"]["pot"] >= 0, "Pot went negative!"
            steps += 1


def test_chips_conservation():
    """Chips + pot must be conserved at every step within a hand.

    Rebuys happen only at the start of _start_new_hand (called from reset).
    Between two consecutive steps inside the same hand, the total must be stable.
    """
    env = SimulatorEnv(num_players=4)
    env.reset()
    done = False
    steps = 0
    while not done and steps < 200:
        before = sum(p["chips"] for p in env.players_list) + env.pot
        _, _, done, _, _ = env.step(env.action_space.sample())
        after = sum(p["chips"] for p in env.players_list) + env.pot
        assert abs(after - before) <= 1, (
            f"Step {steps}: chip total changed from {before} to {after}"
        )
        steps += 1


def test_randomize_players_produces_varied_counts():
    """With min=2, max=6 over 30 episodes, should see ≥2 distinct player counts."""
    env = SimulatorEnv(min_players=2, max_players=6)
    seen_counts = set()
    for _ in range(30):
        env.reset()
        seen_counts.add(len(env.players_list))
    assert len(seen_counts) >= 2, (
        f"Expected varied player counts but only saw: {seen_counts}"
    )


def test_fixed_players_stays_constant():
    """When min==max, table size should never change."""
    env = SimulatorEnv(min_players=3, max_players=3)
    for _ in range(10):
        env.reset()
        assert len(env.players_list) == 3


def test_reward_is_finite():
    env = SimulatorEnv(num_players=2)
    env.reset()
    done = False
    steps = 0
    import math
    while not done and steps < 200:
        _, reward, done, _, _ = env.step(env.action_space.sample())
        assert math.isfinite(reward), f"Non-finite reward at step {steps}: {reward}"
        steps += 1
