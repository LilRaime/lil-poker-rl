"""SelfPlayEnv — SimulatorEnv where bots use a RecurrentPPO model chosen from a pool.

To prevent the agent from collapsing into a degenerate strategy (e.g. always
folding) during self-play, we use fictitious play. The environment maintains a
pool of historical checkpoints in ``models/opponent_pool/``. At the start of each
episode/hand, a random model is selected from the pool to act as the opponent.

A class-level model cache is used to avoid reloading the models from disk on
every hand, preventing training bottlenecks.
"""

import os
import random
import numpy as np

from adapters.simulator.sim_env import SimulatorEnv
from poker_env.action_space import PokerAction
from poker_env.observation import encode_state


class SelfPlayEnv(SimulatorEnv):
    """SimulatorEnv where non-agent players are controlled by a RecurrentPPO model."""

    _MODEL_CACHE = {}

    def __init__(
        self,
        opponent_model_path: str = "models/opponent_model.zip",
        vec_normalize_path: str = "models/vec_normalize.pkl",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._default_opponent_path = opponent_model_path
        self._opponent_model_path = opponent_model_path
        self._vec_normalize_path = vec_normalize_path

        self._opponent_model = None
        self._vec_normalize = None
        self._opponent_loaded = False

    def reset(self, seed=None, options=None):
        self._select_random_opponent()

        return super().reset(seed=seed, options=options)

    def _select_random_opponent(self) -> None:
        """Randomly selects an opponent from the models pool to combat collapse."""
        pool_dir = "models/opponent_pool"
        if not os.path.exists(pool_dir):
            self._opponent_model_path = self._default_opponent_path
            self._opponent_loaded = False
            return

        try:
            models = [f for f in os.listdir(pool_dir) if f.endswith(".zip")]
        except Exception:
            models = []

        if not models:
            self._opponent_model_path = self._default_opponent_path
            self._opponent_loaded = False
            return

        if random.random() < 0.8:
            selected = random.choice(models)
            self._opponent_model_path = os.path.join(pool_dir, selected)
        else:
            self._opponent_model_path = self._default_opponent_path

        self._opponent_loaded = False

    def _ensure_opponent_loaded(self) -> None:
        """Load opponent model from path (using cache) + VecNormalize."""
        if self._opponent_loaded:
            return
        self._opponent_loaded = True

        if not os.path.exists(self._opponent_model_path):
            return

        if self._opponent_model_path in self._MODEL_CACHE:
            self._opponent_model = self._MODEL_CACHE[self._opponent_model_path]
        else:
            try:
                from stable_baselines3 import PPO
                self._opponent_model = PPO.load(self._opponent_model_path, device="cpu")

                if len(self._MODEL_CACHE) >= 15:
                    self._MODEL_CACHE.pop(next(iter(self._MODEL_CACHE)))
                self._MODEL_CACHE[self._opponent_model_path] = self._opponent_model
            except Exception as exc:
                print(f"[SelfPlayEnv] Load error for {self._opponent_model_path}: {exc}")
                self._opponent_model = None

        if self._vec_normalize is None and os.path.exists(self._vec_normalize_path):
            try:
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                dummy = DummyVecEnv([lambda: SimulatorEnv(num_players=2)])
                vn = VecNormalize.load(self._vec_normalize_path, dummy)
                vn.training = False
                vn.norm_reward = False
                self._vec_normalize = vn
            except Exception as exc:
                print(f"[SelfPlayEnv] VecNormalize load error: {exc}")
                self._vec_normalize = None

    def reload_opponent(self) -> None:
        """Force reload (kept for compatibility)."""
        self._opponent_loaded = False
        self._opponent_model = None
        self._ensure_opponent_loaded()

    def _bot_act(self, bot: dict) -> None:
        """Use the frozen PPO model for bot actions; fall back to heuristic."""
        self._ensure_opponent_loaded()

        if self._opponent_model is None:
            super()._bot_act(bot)
            return

        state = self._make_state_dict()
        obs = encode_state(state, bot["id"]).reshape(1, -1)

        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(obs)

        action, _ = self._opponent_model.predict(
            obs,
            deterministic=True
        )

        self._apply_bot_action(bot, int(action[0]))

    def _apply_bot_action(self, bot: dict, action_idx: int) -> None:
        """Translate a discrete action index into a bet for *bot*."""
        to_call = max(0, self.current_bet - bot["bet"])
        min_raise = self.big_blind
        pot = self.pot

        if action_idx == PokerAction.FOLD:
            bot["folded"] = True
            bot["acted"] = True
            self._record_action(bot["id"], "fold", 0)
        elif action_idx == PokerAction.CHECK_CALL:
            added = self._execute_bet(bot, to_call)
            self._record_action(bot["id"], "check" if to_call == 0 else "call", added)
        elif action_idx == PokerAction.RAISE_MIN:
            added = self._execute_bet(bot, to_call + min_raise)
            self._record_action(bot["id"], "raise", added)
        elif action_idx == PokerAction.RAISE_POT:
            added = self._execute_bet(bot, to_call + max(min_raise, pot))
            self._record_action(bot["id"], "raise", added)
        elif action_idx == PokerAction.ALL_IN:
            added = self._execute_bet(bot, bot["chips"])
            self._record_action(bot["id"], "all_in", added)
        else:
            added = self._execute_bet(bot, to_call)
            self._record_action(bot["id"], "check" if to_call == 0 else "call", added)
