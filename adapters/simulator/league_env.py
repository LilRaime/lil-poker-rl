"""LeagueEnv — League Training with multiple opponent types.

Opponent pool (selected once per episode):
  • RandomBot  — picks a random discrete action
  • RuleBot    — heuristic from SimulatorEnv._bot_act
  • PPO pool   — historical snapshots from ``models/opponent_pool/``

Within the PPO pool, models are sampled proportionally to a softmax over
their Elo ratings so stronger opponents are challenged more often.

Also exports ``RandomBotEnv`` — a plain SimulatorEnv override used for
standalone Elo evaluation against a fully random opponent.
"""

import os
import json
import random
import numpy as np

from adapters.simulator.sim_env import SimulatorEnv
from adapters.simulator.self_play_env import LRUModelCache
from poker_env.action_space import PokerAction
from poker_env.observation import encode_state

OPPONENT_RANDOM = "random"
OPPONENT_RULE = "rule"
OPPONENT_PPO = "ppo"

_DEFAULT_ELO = 1500.0

class RandomBotEnv(SimulatorEnv):
    """SimulatorEnv where all non-agent bots pick actions uniformly at random.

    Useful for Elo baseline evaluation; not intended for league training.
    """

    def _bot_act(self, bot: dict) -> None:
        self._apply_random_action(bot)

    def _apply_random_action(self, bot: dict) -> None:
        to_call = max(0, self.current_bet - bot["bet"])
        min_raise = self.big_blind
        pot = self.pot
        action_idx = random.randint(0, 4)

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


class LeagueEnv(SimulatorEnv):
    """SimulatorEnv with league-style multi-opponent training.

    At the start of each episode an opponent type is chosen
    (RandomBot / RuleBot / PPO-from-pool) according to ``*_weight``
    parameters.  Within the PPO pool, models are sampled proportionally
    to a softmax over their Elo ratings so stronger opponents are
    selected more often.

    Args:
        vec_normalize_path: Path to ``VecNormalize`` statistics for PPO bots.
        elo_path:           Path to the Elo registry JSON file.
        ppo_weight:         Relative weight for selecting a PPO opponent.
        rule_weight:        Relative weight for selecting the rule-based heuristic.
        random_weight:      Relative weight for selecting a random opponent.
        **kwargs:           Forwarded to ``SimulatorEnv``.
    """

    _MODEL_CACHE: LRUModelCache = LRUModelCache(maxsize=20)

    def __init__(
        self,
        vec_normalize_path: str = "models/vec_normalize.pkl",
        elo_path: str = "models/elo.json",
        ppo_weight: float = 0.60,
        rule_weight: float = 0.25,
        random_weight: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vec_normalize_path = vec_normalize_path
        self._elo_path = elo_path
        self._base_weights = {
            OPPONENT_PPO: ppo_weight,
            OPPONENT_RULE: rule_weight,
            OPPONENT_RANDOM: random_weight,
        }

        self._opponent_type: str = OPPONENT_RULE
        self._opponent_model_path: str | None = None
        self._opponent_model = None
        self._vec_normalize = None
        self._opponent_loaded: bool = False

    def reset(self, seed=None, options=None):
        self._select_opponent()
        return super().reset(seed=seed, options=options)

    def _select_opponent(self) -> None:
        """Choose opponent type + specific model using Elo-weighted sampling."""
        ppo_models = self._get_ppo_models()

        weights = dict(self._base_weights)
        if not ppo_models:
            weights[OPPONENT_PPO] = 0.0

        total = sum(weights.values())
        if total == 0:
            self._opponent_type = OPPONENT_RULE
            return

        keys = list(weights.keys())
        probs = [weights[k] / total for k in keys]
        opp_type: str = random.choices(keys, weights=probs)[0]

        self._opponent_type = opp_type
        self._opponent_model = None
        self._opponent_model_path = None
        self._opponent_loaded = False

        if opp_type == OPPONENT_PPO:
            self._opponent_model_path = self._elo_weighted_pick(ppo_models)

    def _get_ppo_models(self) -> list[str]:
        """Return list of available PPO model paths (pool + fallback)."""
        paths: list[str] = []
        pool_dir = "models/opponent_pool"
        if os.path.exists(pool_dir):
            try:
                paths = [
                    os.path.join(pool_dir, f)
                    for f in os.listdir(pool_dir)
                    if f.endswith(".zip")
                ]
            except Exception:
                pass
        fallback = "models/opponent_model.zip"
        if os.path.exists(fallback) and fallback not in paths:
            paths.append(fallback)
        return paths

    def _elo_weighted_pick(self, paths: list[str]) -> str:
        """Pick a model path using softmax over Elo ratings.

        Higher Elo → higher selection weight (harder self-play).
        """
        elo_data: dict = {}
        if os.path.exists(self._elo_path):
            try:
                with open(self._elo_path) as f:
                    elo_data = json.load(f).get("ratings", {})
            except Exception:
                pass

        elos = np.array(
            [elo_data.get(os.path.basename(p), _DEFAULT_ELO) for p in paths],
            dtype=float,
        )
        std = elos.std() if elos.std() > 0 else 1.0
        softmax_weights = np.exp((elos - elos.mean()) / (std + 1e-6) * 0.5)
        softmax_weights /= softmax_weights.sum()
        idx = int(np.random.choice(len(paths), p=softmax_weights))
        return paths[idx]

    def _ensure_opponent_loaded(self) -> None:
        if self._opponent_loaded:
            return
        self._opponent_loaded = True

        if self._opponent_type != OPPONENT_PPO or not self._opponent_model_path:
            return

        path = self._opponent_model_path
        if not os.path.exists(path):
            self._opponent_type = OPPONENT_RULE
            return

        cached = self._MODEL_CACHE.get(path)
        if cached is not None:
            self._opponent_model = cached
        else:
            try:
                from stable_baselines3 import PPO
                model = PPO.load(path, device="cpu")
                self._MODEL_CACHE.put(path, model)
                self._opponent_model = model
            except Exception as exc:
                print(f"[LeagueEnv] Load error {path}: {exc}")
                self._opponent_type = OPPONENT_RULE
                return

        if self._vec_normalize is None and os.path.exists(self._vec_normalize_path):
            try:
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
                dummy = DummyVecEnv([lambda: SimulatorEnv(num_players=2)])
                vn = VecNormalize.load(self._vec_normalize_path, dummy)
                vn.training = False
                vn.norm_reward = False
                self._vec_normalize = vn
            except Exception as exc:
                print(f"[LeagueEnv] VecNormalize load error: {exc}")

    def _bot_act(self, bot: dict) -> None:
        self._ensure_opponent_loaded()

        if self._opponent_type == OPPONENT_RANDOM:
            self._apply_bot_action(bot, random.randint(0, 4))
        elif self._opponent_type == OPPONENT_PPO and self._opponent_model is not None:
            self._ppo_bot_act(bot)
        else:
            super()._bot_act(bot)

    def _ppo_bot_act(self, bot: dict) -> None:
        state = self._make_state_dict()
        obs = encode_state(state, bot["id"]).reshape(1, -1)
        if self._vec_normalize is not None:
            obs = self._vec_normalize.normalize_obs(obs)
        action, _ = self._opponent_model.predict(obs, deterministic=True)
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
