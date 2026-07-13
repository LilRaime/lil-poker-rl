import abc
import numpy as np
import gymnasium as gym
from poker_env.action_space import PokerAction, get_action_space
from poker_env.observation import get_observation_space, encode_state


class BasePokerEnv(gym.Env, abc.ABC):
    metadata = {"render_modes": ["human"]}

    def __init__(self, player_id: str, starting_chips: int = 1000):
        super().__init__()
        self.player_id = player_id
        self.starting_chips = starting_chips
        self.action_space = get_action_space()
        self.observation_space = get_observation_space()
        self.last_chips = starting_chips
        self.game_state = {}
        self._prev_potential = 0.0
        self.is_first_action = True

    @abc.abstractmethod
    def _send_action(self, action: int, amount: int = 0) -> None:
        pass

    @abc.abstractmethod
    def _wait_for_my_turn(self) -> dict:
        pass

    @abc.abstractmethod
    def _start_new_hand(self) -> dict:
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        my_status = self._get_player_status(self.game_state)
        current_chips = float(my_status.get("chips", self.starting_chips)) if my_status else self.starting_chips
        self.last_chips = self.starting_chips if current_chips <= 0 else current_chips

        while True:
            self.game_state = self._start_new_hand()
            self.game_state = self._wait_for_my_turn()
            phase = self.game_state.get("phase", "Pre-Flop")
            if phase not in ("Waiting", "Showdown"):
                break
            my_status = self._get_player_status(self.game_state)
            if my_status:
                self.last_chips = float(my_status.get("chips", self.last_chips))

        obs = encode_state(self.game_state, self.player_id)
        self._prev_potential = self._compute_potential(obs)
        self.is_first_action = True
        info = {"game_state": self.game_state}
        return obs, info

    def step(self, action):
        if self.is_first_action:
            self.is_first_action = False
            if action == PokerAction.ALL_IN:
                action = PokerAction.CHECK_CALL

        current_phase = self.game_state.get("phase", "Pre-Flop")
        if current_phase == "Pre-Flop" and action == PokerAction.ALL_IN:
            action = PokerAction.RAISE_POT

        phi_before = self._prev_potential

        action_name, amount = self._map_action(action)
        self._send_action(action, amount)
        self.game_state = self._wait_for_my_turn()
        my_status = self._get_player_status(self.game_state)
        current_chips = float(my_status.get("chips", 0)) if my_status else 0.0

        chip_reward = (current_chips - self.last_chips) / self.starting_chips
        self.last_chips = current_chips

        done = False
        if not my_status or current_chips <= 0 or my_status.get("sitting_out", False):
            done = True

        phase = self.game_state.get("phase", "Waiting")
        if phase == "Waiting":
            done = True

        obs = encode_state(self.game_state, self.player_id)

        phi_after = 0.0 if done else self._compute_potential(obs)
        self._prev_potential = phi_after
        shaped_reward = 0.99 * phi_after - phi_before
        reward = chip_reward + shaped_reward

        info = {"game_state": self.game_state}
        truncated = False
        return obs, reward, done, truncated, info

    def _compute_potential(self, obs: np.ndarray) -> float:
        """Φ(s) = hand_strength × pot_ratio × scale.

        Rewards staying in hands with strong holdings relative to pot size.
        Scale factor (0.05) keeps shaping << chip reward (max ≈ 1.0).
        """
        hand_strength = float(obs[63])
        pot_ratio = float(obs[14])
        return 0.05 * hand_strength * pot_ratio

    def _get_player_status(self, state: dict) -> dict:
        players = state.get("players", []) or []
        for p in players:
            if p.get("id") == self.player_id:
                return p
        return {}

    def _map_action(self, action_idx: int) -> tuple[str, int]:
        current_bet = self.game_state.get("current_bet", 0)
        pot = self.game_state.get("pot", 0)

        my_status = self._get_player_status(self.game_state)
        my_bet = my_status.get("bet", 0) if my_status else 0
        my_chips = my_status.get("chips", 0) if my_status else 0

        to_call = max(0, current_bet - my_bet)

        players = self.game_state.get("players", []) or []
        bets = [float(p.get("bet", 0)) for p in players]
        bets.sort(reverse=True)
        highest_bet = bets[0] if len(bets) > 0 else 0.0
        second_highest_bet = bets[1] if len(bets) > 1 else 0.0
        last_raise_size = highest_bet - second_highest_bet
        big_blind = float(self.game_state.get("big_blind", 20))
        min_raise_to = int(highest_bet + max(big_blind, last_raise_size))

        if action_idx == PokerAction.FOLD:
            return "fold", 0
        elif action_idx == PokerAction.CHECK_CALL:
            if to_call == 0:
                return "check", 0
            else:
                return "call", 0
        elif action_idx == PokerAction.RAISE_MIN:
            amount = min_raise_to
            if amount > my_chips + my_bet:
                return "all_in", my_chips
            return "raise", amount
        elif action_idx == PokerAction.RAISE_POT:
            amount = max(min_raise_to, int(current_bet + max(big_blind, pot)))
            if amount > my_chips + my_bet:
                return "all_in", my_chips
            return "raise", amount
        elif action_idx == PokerAction.ALL_IN:
            return "all_in", my_chips
        return "fold", 0
