"""League Elo rating system for poker agents.

Manages Elo ratings for all pool agents (RandomBot, RuleBot, PPO snapshots).
Persists ratings, win-rates, and full match history to ``models/elo.json``.

Typical usage inside a training callback::

    from agent.league import EloRegistry, evaluate_matchup, AGENT_RULE

    elo = EloRegistry()
    w, l, d = evaluate_matchup(model, lambda: SimulatorEnv(num_players=2))
    elo.record_result("snapshot_100k.zip", AGENT_RULE, w, l, d)
    print(elo.summary())
"""

import os
import json
import time

AGENT_RANDOM = "random_bot"
AGENT_RULE = "rule_bot"

DEFAULT_ELO = 1500.0
K_FACTOR = 32.0


def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Expected score for agent A in a match against agent B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


class EloRegistry:
    """JSON-backed Elo registry for all agents in the pool.

    File layout (``models/elo.json``)::

        {
          "ratings":  {
              "snapshot_100k.zip": 1532.5,
              "rule_bot":          1400.0,
              "random_bot":        1200.0,
              ...
          },
          "winrates": {
              "snapshot_100k.zip_vs_rule_bot": {
                  "wins": 623, "losses": 341, "draws": 36,
                  "n_games": 1000, "winrate": 0.623
              },
              ...
          },
          "history":  [
              {
                  "timestamp": 1753271234,
                  "a": "snapshot_100k.zip",
                  "b": "rule_bot",
                  "wins": 623, "losses": 341, "draws": 36,
                  "elo_a_before": 1500.0, "elo_b_before": 1400.0,
                  "elo_a_after":  1532.5, "elo_b_after":  1367.5
              },
              ...
          ]
        }
    """

    def __init__(self, path: str = "models/elo.json"):
        self._path = path
        self._data = self._load()
        for name, seed in [(AGENT_RANDOM, 1200.0), (AGENT_RULE, 1400.0)]:
            if name not in self._data["ratings"]:
                self._data["ratings"][name] = seed

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    data = json.load(f)
                    data.setdefault("ratings", {})
                    data.setdefault("winrates", {})
                    data.setdefault("history", [])
                    return data
            except Exception:
                pass
        return {"ratings": {}, "winrates": {}, "history": []}

    def save(self) -> None:
        """Write current ratings to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def get_rating(self, name: str) -> float:
        """Return current Elo rating (DEFAULT_ELO if unseen)."""
        return self._data["ratings"].get(name, DEFAULT_ELO)

    def set_rating(self, name: str, rating: float) -> None:
        self._data["ratings"][name] = round(rating, 1)

    def record_result(
        self,
        name_a: str,
        name_b: str,
        wins: int,
        losses: int,
        draws: int,
    ) -> tuple[float, float]:
        """Update Elo ratings for both agents and persist to disk.

        Args:
            name_a:  Identifier for agent A (e.g. ``"snapshot_100k.zip"``).
            name_b:  Identifier for agent B (e.g. ``AGENT_RULE``).
            wins:    Hands won by A.
            losses:  Hands won by B (= losses for A).
            draws:   Hands that ended as a draw (chip-delta == 0).

        Returns:
            ``(new_elo_a, new_elo_b)``
        """
        total = wins + losses + draws
        if total == 0:
            return self.get_rating(name_a), self.get_rating(name_b)

        ra = self.get_rating(name_a)
        rb = self.get_rating(name_b)

        score_a = (wins + 0.5 * draws) / total
        ea = _elo_expected(ra, rb)

        new_ra = ra + K_FACTOR * (score_a - ea)
        new_rb = rb + K_FACTOR * ((1.0 - score_a) - (1.0 - ea))

        self.set_rating(name_a, new_ra)
        self.set_rating(name_b, new_rb)

        winrate = wins / total
        self._data["winrates"][f"{name_a}_vs_{name_b}"] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "n_games": total,
            "winrate": round(winrate, 4),
        }
        self._data["history"].append({
            "timestamp": int(time.time()),
            "a": name_a,
            "b": name_b,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "elo_a_before": round(ra, 1),
            "elo_b_before": round(rb, 1),
            "elo_a_after": round(new_ra, 1),
            "elo_b_after": round(new_rb, 1),
        })
        self.save()
        return new_ra, new_rb

    def summary(self) -> str:
        """Return a formatted Elo leaderboard string."""
        lines = ["=== Elo Ratings ==="]
        sorted_ratings = sorted(
            self._data["ratings"].items(), key=lambda x: -x[1]
        )
        for name, rating in sorted_ratings:
            wr_parts = []
            for key, val in self._data["winrates"].items():
                if key.startswith(f"{name}_vs_"):
                    opp = key.split("_vs_", 1)[1]
                    wr_parts.append(f"vs {opp}: {val['winrate']:.1%}")
            wr_str = "  " + " | ".join(wr_parts) if wr_parts else ""
            lines.append(f"  {name:<42s}  {rating:7.1f}{wr_str}")
        return "\n".join(lines)


def evaluate_matchup(
    model_a,
    env_factory,
    n_games: int = 1000,
) -> tuple[int, int, int]:
    """Simulate *n_games* hands; *model_a* controls player_0.

    The environment's internal ``_bot_act`` controls the opponents, so
    *env_factory* fully determines who player_0 fights against:

    * ``lambda: SimulatorEnv(num_players=2)``   → vs RuleBot
    * ``lambda: RandomBotEnv(num_players=2)``   → vs RandomBot
    * ``lambda: SelfPlayEnv(path, ...)``        → vs a PPO checkpoint

    Win / loss / draw is determined by chip delta at episode end.

    Args:
        model_a:     Loaded SB3 ``PPO`` model with a ``predict`` interface.
        env_factory: Zero-argument callable returning a fresh environment.
        n_games:     Number of complete hands to simulate.

    Returns:
        ``(wins, losses, draws)``
    """
    wins = losses = draws = 0

    env = env_factory()
    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model_a.predict(obs.reshape(1, -1), deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action[0]))
            done = terminated or truncated

        chips = env.players_list[0]["chips"]
        start = env.starting_chips
        if chips > start:
            wins += 1
        elif chips < start:
            losses += 1
        else:
            draws += 1

    env.close()
    return wins, losses, draws
