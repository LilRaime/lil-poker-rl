import random
import numpy as np
from phevaluator import evaluate_cards as _phe_evaluate
from poker_env.base_env import BasePokerEnv
from poker_env.observation import PHE_CARD_CACHE

SUIT_MAP = {'♠': 's', '♥': 'h', '♦': 'd', '♣': 'c'}
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['♠', '♥', '♦', '♣']
DECK_TEMPLATE = [r + s for r in RANKS for s in SUITS]

CARD_CACHE = PHE_CARD_CACHE


class SimulatorEnv(BasePokerEnv):
    def __init__(
        self,
        num_players: int = 6,
        starting_chips: int = 1000,
        min_players: int = 0,
        max_players: int = 0,
    ):
        """Offline Texas Hold'em simulator.

        Args:
            num_players:    Fixed table size when min_players == max_players == 0.
            starting_chips: Chip stack each player starts a session with.
            min_players:    Lower bound for randomised table size per episode.
            max_players:    Upper bound for randomised table size per episode.
                            When either bound is 0 the table size stays fixed at
                            num_players.
        """
        self.min_players = min_players if min_players > 0 else num_players
        self.max_players = max_players if max_players > 0 else num_players
        self.randomize_players = self.min_players != self.max_players

        super().__init__(player_id="player_0", starting_chips=starting_chips)

        initial_players = self.min_players if (min_players > 0 or max_players > 0) else num_players
        self._setup_players(initial_players)

        self.deck = []
        self.board = []
        self.phase = "Waiting"
        self.pot = 0
        self.current_bet = 0
        self.dealer_idx = 0
        self.active_idx = 0
        self.small_blind = 10
        self.big_blind = 20
        self.hand_count = 0

        self.session_stats = {}

        self.aggressor_idx = -1
        self.street_action_counts = {
            "Pre-Flop": {"raise": 0, "call": 0, "fold": 0},
            "Flop": {"raise": 0, "call": 0, "fold": 0},
            "Turn": {"raise": 0, "call": 0, "fold": 0},
            "River": {"raise": 0, "call": 0, "fold": 0},
        }
        self.street_last_actions = [{} for _ in range(4)]
        self.street_contributions = [{} for _ in range(4)]
        self.hand_actions = []

    def _setup_players(self, n: int) -> None:
        """(Re)build the players_list for n seats."""
        self.num_players = n
        self.players_list = []
        for i in range(n):
            self.players_list.append({
                "id": f"player_{i}",
                "name": f"Bot {i}" if i > 0 else "RL Agent",
                "chips": self.starting_chips,
                "hole": [],
                "bet": 0,
                "folded": False,
                "all_in": False,
                "sitting_out": False,
                "seat": i,
                "is_small_blind": False,
                "is_big_blind": False,
                "acted": False,
            })

    def _start_new_hand(self) -> dict:
        self.hand_count += 1

        self.aggressor_idx = -1
        self.street_action_counts = {
            "Pre-Flop": {"raise": 0, "call": 0, "fold": 0},
            "Flop": {"raise": 0, "call": 0, "fold": 0},
            "Turn": {"raise": 0, "call": 0, "fold": 0},
            "River": {"raise": 0, "call": 0, "fold": 0},
        }
        self.street_last_actions = [{} for _ in range(4)]
        self.street_contributions = [{} for _ in range(4)]
        self.hand_actions = []

        if self.randomize_players:
            n = random.randint(self.min_players, self.max_players)
            old_chips = {p["id"]: p["chips"] for p in self.players_list}
            self._setup_players(n)
            for p in self.players_list:
                p["chips"] = old_chips.get(p["id"], self.starting_chips)

        self.deck = DECK_TEMPLATE.copy()
        random.shuffle(self.deck)

        for p in self.players_list:
            p["bet"] = 0
            p["folded"] = False
            p["all_in"] = False
            p["is_small_blind"] = False
            p["is_big_blind"] = False
            p["hole"] = []
            p["acted"] = False
            p["_strength"] = -1.0
            p["chips"] = self.starting_chips

        self.board = []
        self.pot = 0
        self.current_bet = self.big_blind
        self.dealer_idx = (self.dealer_idx + 1) % self.num_players

        sb_idx = (self.dealer_idx + 1) % self.num_players
        bb_idx = (self.dealer_idx + 2) % self.num_players
        self.players_list[sb_idx]["is_small_blind"] = True
        self.players_list[bb_idx]["is_big_blind"] = True

        sb_amount = min(self.small_blind, self.players_list[sb_idx]["chips"])
        self.players_list[sb_idx]["chips"] -= sb_amount
        self.players_list[sb_idx]["bet"] = sb_amount

        bb_amount = min(self.big_blind, self.players_list[bb_idx]["chips"])
        self.players_list[bb_idx]["chips"] -= bb_amount
        self.players_list[bb_idx]["bet"] = bb_amount

        for p in self.players_list:
            p["hole"] = [self.deck.pop(), self.deck.pop()]

        self.phase = "Pre-Flop"
        self.pot = sb_amount + bb_amount
        self.active_idx = (bb_idx + 1) % self.num_players
        return self._make_state_dict()

    def _wait_for_my_turn(self) -> dict:
        while True:
            active_count = 0
            active_winner = None
            for p in self.players_list:
                if not p["folded"]:
                    active_count += 1
                    active_winner = p
                    if active_count > 1:
                        active_winner = None
                        break

            if active_count == 1:
                self._end_hand_early(active_winner)
                return self._make_state_dict()

            if self._is_betting_round_complete():
                self._next_phase()
                if self.phase in ["Showdown", "Waiting"]:
                    return self._make_state_dict()
                continue

            current = self.players_list[self.active_idx]
            if current["folded"] or current["all_in"]:
                self.active_idx = (self.active_idx + 1) % self.num_players
                continue

            if current["id"] == self.player_id:
                return self._make_state_dict()

            self._bot_act(current)
            self.active_idx = (self.active_idx + 1) % self.num_players

    def _send_action(self, action: int, amount: int = 0) -> None:
        agent = self.players_list[0]
        to_call = self.current_bet - agent["bet"]

        if action == 0:
            agent["folded"] = True
            agent["acted"] = True
            self._record_action(agent["id"], "fold", 0)
        elif action == 1:
            added = self._execute_bet(agent, to_call)
            self._record_action(agent["id"], "check" if to_call == 0 else "call", added)
        elif action in [2, 3]:
            added_amount = amount - agent["bet"]
            added = self._execute_bet(agent, max(to_call, added_amount))
            self._record_action(agent["id"], "raise", added)
        elif action == 4:
            added = self._execute_bet(agent, agent["chips"])
            self._record_action(agent["id"], "all_in", added)

        self.active_idx = (self.active_idx + 1) % self.num_players

    def _compute_bot_strength(self, bot: dict) -> float:
        """Return hand strength in [0, 1], cached per phase-transition.

        The treys evaluator is called at most once per (bot, phase) pair.
        A cached value of -1.0 signals "not yet computed this phase".
        """
        cached = bot.get("_strength", -1.0)
        if cached >= 0.0:
            return cached

        hole = bot["hole"]
        board = self.board

        if not board or len(board) < 3:
            ranks = "23456789TJQKA"
            r1 = hole[0][:-1]
            r2 = hole[1][:-1]
            r1_t = 'T' if r1 == '10' else r1
            r2_t = 'T' if r2 == '10' else r2
            r1_idx = ranks.find(r1_t) if r1_t in ranks else 6
            r2_idx = ranks.find(r2_t) if r2_t in ranks else 6
            strength = max(r1_idx, r2_idx) / 12.0
            if r1_t == r2_t:
                strength = min(1.0, strength + 0.3)
            if hole[0][-1] == hole[1][-1]:
                strength = min(1.0, strength + 0.05)
        else:
            try:
                phe_cards = [CARD_CACHE[c] for c in hole + board]
                score = _phe_evaluate(*phe_cards)
                strength = 1.0 - (score - 1) / 7461.0
            except Exception:
                strength = 0.5

        bot["_strength"] = strength
        return strength

    def _bot_act(self, bot: dict) -> None:
        """Improved bot: uses pot-odds and actual hand strength (postflop via treys)."""
        to_call = self.current_bet - bot["bet"]
        strength = self._compute_bot_strength(bot)
        rand_val = random.random()

        if to_call == 0:
            if strength > 0.75 and rand_val < 0.45:
                added = self._execute_bet(bot, max(self.big_blind, self.pot // 2))
                self._record_action(bot["id"], "raise", added)
            elif strength > 0.55 and rand_val < 0.15:
                added = self._execute_bet(bot, self.big_blind)
                self._record_action(bot["id"], "raise", added)
            else:
                bot["acted"] = True
                self._record_action(bot["id"], "check", 0)
        else:
            pot_odds = to_call / max(1, self.pot + to_call)

            if strength < pot_odds * 0.8 and rand_val > strength * 0.5:
                bot["folded"] = True
                bot["acted"] = True
                self._record_action(bot["id"], "fold", 0)
            elif strength > 0.82 and rand_val < 0.35:
                raise_amount = to_call + max(self.big_blind, self.pot // 2)
                added = self._execute_bet(bot, raise_amount)
                self._record_action(bot["id"], "raise", added)
            else:
                added = self._execute_bet(bot, to_call)
                self._record_action(bot["id"], "call", added)

    def _record_action(self, player_id: str, action_name: str, amount: int) -> None:
        phase_map = {"Pre-Flop": 0, "Flop": 1, "Turn": 2, "River": 3}
        action_codes = {
            "fold": 0,
            "check": 1,
            "call": 1,
            "raise": 2,
            "all_in": 3,
        }

        phase = self.phase
        phase_idx = phase_map.get(phase)
        if phase_idx is None:
            return

        action_type = "fold" if action_name == "fold" else ("raise" if action_name in ("raise", "all_in") else "call")
        if phase in self.street_action_counts:
            self.street_action_counts[phase][action_type] += 1

        action_code = action_codes.get(action_name, -1)
        self.street_last_actions[phase_idx][player_id] = action_code

        self.street_contributions[phase_idx][player_id] = self.street_contributions[phase_idx].get(player_id, 0) + amount

        if action_name in ("raise", "all_in"):
            for p in self.players_list:
                if p["id"] == player_id:
                    self.aggressor_idx = p["seat"]
                    break

        self.hand_actions.append((player_id, phase, action_name, amount))

    def _update_session_stats(self) -> None:
        active_players = {p["id"] for p in self.players_list if not p["sitting_out"]}

        pf_vpip = {pid: 0 for pid in active_players}
        pf_pfr = {pid: 0 for pid in active_players}
        calls_count = {pid: 0 for pid in active_players}
        raises_count = {pid: 0 for pid in active_players}
        acted_preflop = set()

        for player_id, phase, action_name, amount in self.hand_actions:
            if player_id not in active_players:
                continue
            if phase == "Pre-Flop":
                acted_preflop.add(player_id)
                if action_name in ("call", "raise", "all_in"):
                    pf_vpip[player_id] = 1
                if action_name in ("raise", "all_in"):
                    pf_pfr[player_id] = 1
            if action_name == "call":
                calls_count[player_id] += 1
            elif action_name in ("raise", "all_in"):
                raises_count[player_id] += 1

        for pid in active_players:
            if pid not in self.session_stats:
                self.session_stats[pid] = {
                    "hands_played": 0,
                    "vpip_opportunities": 0,
                    "vpip_actions": 0,
                    "pfr_opportunities": 0,
                    "pfr_actions": 0,
                    "calls": 0,
                    "bets_or_raises": 0,
                }
            stats = self.session_stats[pid]
            stats["hands_played"] += 1
            if pid in acted_preflop:
                stats["vpip_opportunities"] += 1
                stats["vpip_actions"] += pf_vpip[pid]
                stats["pfr_opportunities"] += 1
                stats["pfr_actions"] += pf_pfr[pid]
            stats["calls"] += calls_count[pid]
            stats["bets_or_raises"] += raises_count[pid]

    def _execute_bet(self, player: dict, amount: int) -> int:
        if amount >= player["chips"]:
            actual_bet = player["chips"]
            player["chips"] = 0
            player["all_in"] = True
            player["bet"] += actual_bet
            self.pot += actual_bet
        else:
            actual_bet = amount
            player["chips"] -= actual_bet
            player["bet"] += actual_bet
            self.pot += actual_bet

        player["acted"] = True

        if player["bet"] > self.current_bet:
            self.current_bet = player["bet"]
            for p in self.players_list:
                if p["id"] != player["id"] and not p["folded"] and not p["all_in"]:
                    p["acted"] = False
        return actual_bet

    def _is_betting_round_complete(self) -> bool:
        for p in self.players_list:
            if p["folded"] or p["sitting_out"] or p["all_in"]:
                continue
            if not p["acted"] or p["bet"] < self.current_bet:
                return False
        return True

    def _next_phase(self) -> None:
        for p in self.players_list:
            p["bet"] = 0
            p["acted"] = False
            p["_strength"] = -1.0
        self.current_bet = 0
        self.active_idx = (self.dealer_idx + 1) % self.num_players

        if self.phase == "Pre-Flop":
            self.phase = "Flop"
            self.board = [self.deck.pop() for _ in range(3)]
        elif self.phase == "Flop":
            self.phase = "Turn"
            self.board.append(self.deck.pop())
        elif self.phase == "Turn":
            self.phase = "River"
            self.board.append(self.deck.pop())
        elif self.phase == "River":
            self.phase = "Showdown"
            self._evaluate_showdown()

    def _evaluate_showdown(self) -> None:
        self._update_session_stats()
        active_players = [p for p in self.players_list if not p["folded"]]
        if not active_players:
            self.phase = "Waiting"
            return

        phe_board = [CARD_CACHE[c] for c in self.board]
        best_score = float('inf')
        winners = []

        for p in active_players:
            phe_hand = [CARD_CACHE[c] for c in p["hole"]]
            score = _phe_evaluate(*(phe_hand + phe_board))
            p["hand_strength_score"] = score

            if score < best_score:
                best_score = score
                winners = [p]
            elif score == best_score:
                winners.append(p)

        win_share = self.pot // len(winners)
        for w in winners:
            w["chips"] += win_share
        self.pot = 0
        self.phase = "Waiting"

    def _end_hand_early(self, winner: dict) -> None:
        self._update_session_stats()
        winner["chips"] += self.pot
        self.pot = 0
        self.phase = "Waiting"

    def _to_treys_card(self, card_str: str) -> str:
        """Return phevaluator card string (e.g. "As") for the given card.

        Name kept for backward compatibility; returns a string now, not an int.
        """
        return CARD_CACHE[card_str]

    def _make_state_dict(self) -> dict:
        players_status = []
        for p in self.players_list:
            players_status.append({
                "id": p["id"],
                "name": p["name"],
                "chips": p["chips"],
                "hole": p["hole"] if p["id"] == self.player_id or self.phase == "Showdown" else [],
                "bet": p["bet"],
                "folded": p["folded"],
                "all_in": p["all_in"],
                "acted": p["acted"],
                "sitting_out": p["sitting_out"],
                "seat": p["seat"],
                "is_small_blind": p["is_small_blind"],
                "is_big_blind": p["is_big_blind"],
            })

        return {
            "players": players_status,
            "board": self.board,
            "phase": self.phase,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "active_idx": self.active_idx,
            "dealer_idx": self.dealer_idx,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "hand_count": self.hand_count,
            "starting_chips": self.starting_chips,
            "aggressor_idx": getattr(self, "aggressor_idx", -1),
            "street_action_counts": getattr(self, "street_action_counts", {}),
            "street_last_actions": getattr(self, "street_last_actions", [{} for _ in range(4)]),
            "street_contributions": getattr(self, "street_contributions", [{} for _ in range(4)]),
            "session_stats": getattr(self, "session_stats", {}),
        }
