from collections import Counter
import numpy as np
import gymnasium as gym
from phevaluator import evaluate_cards as _phe_evaluate
from phevaluator.card import Card as _PheCard

OBSERVATION_SHAPE = (168,)

SUIT_MAP = {'♠': 's', '♥': 'h', '♦': 'd', '♣': 'c'}

PHE_CARD_CACHE = {}
PHE_STR_CACHE = {}
for r in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']:
    for s in ['♠', '♥', '♦', '♣']:
        c_str = r + s
        rank_char = 'T' if r == '10' else r
        phe_str = rank_char + SUIT_MAP[s]
        PHE_STR_CACHE[c_str] = phe_str
        PHE_CARD_CACHE[c_str] = _PheCard(phe_str).id_
CARD_STR_CACHE = PHE_CARD_CACHE

PHE_DICT_CACHE = {}
_PHE_RANKS = "23456789TJQKA"
_PHE_SUITS = ['s', 'h', 'd', 'c']
for r in range(2, 15):
    for s in range(4):
        PHE_DICT_CACHE[(r, s)] = _PheCard(_PHE_RANKS[r - 2] + _PHE_SUITS[s]).id_
CARD_DICT_CACHE = PHE_DICT_CACHE

PARSE_CARD_CACHE = {}
_SUITS_DICT = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
_RANKS_DICT = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8,
    'J': 9, 'Q': 10, 'K': 11, 'A': 12
}
for r in _RANKS_DICT:
    for s in _SUITS_DICT:
        c_str = r + s
        PARSE_CARD_CACHE[c_str] = (_RANKS_DICT[r] / 12.0, _SUITS_DICT[s] / 3.0)


def get_observation_space() -> gym.spaces.Box:
    return gym.spaces.Box(
        low=-1.0,
        high=10.0,
        shape=OBSERVATION_SHAPE,
        dtype=np.float32
    )


def parse_card(card_str: str) -> tuple[float, float]:
    return PARSE_CARD_CACHE.get(card_str, (-1.0, -1.0))


def get_hand_strength(hole: list, board: list) -> float:
    if not hole or len(hole) < 2:
        return 0.0

    if not board or len(board) < 3:
        ranks = "23456789TJQKA"
        def get_card_char(c):
            if isinstance(c, dict):
                r = c.get("Rank")
                s = c.get("Suit")
                r_str = ranks[r-2] if r is not None and 2 <= r <= 14 else '2'
                s_str = ['♠', '♥', '♦', '♣'][s] if s is not None and 0 <= s <= 3 else '♠'
                return r_str, s_str
            return c[:-1], c[-1]

        try:
            r1_str, s1_str = get_card_char(hole[0])
            r2_str, s2_str = get_card_char(hole[1])

            r1_str = 'T' if r1_str == '10' else r1_str
            r2_str = 'T' if r2_str == '10' else r2_str

            r1 = ranks.find(r1_str) if r1_str in ranks else 0
            r2 = ranks.find(r2_str) if r2_str in ranks else 0

            is_pair = r1_str == r2_str
            is_suited = s1_str == s2_str

            score = max(r1, r2) / 12.0
            if is_pair:
                score += 0.25
            if is_suited:
                score += 0.1
            return min(1.0, score)
        except Exception:
            return 0.0

    try:
        def to_phe(c):
            if isinstance(c, dict):
                return PHE_DICT_CACHE.get((c.get("Rank"), c.get("Suit")), "2s")
            return PHE_CARD_CACHE.get(str(c), "2s")

        phe_cards = [to_phe(c) for c in hole + board]
        score = _phe_evaluate(*phe_cards)
        return 1.0 - (score - 1) / 7461.0
    except Exception:
        return 0.0

_SUIT_IDX = {'♠': 0, '♥': 1, '♦': 2, '♣': 3}
_RANK_IDX = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
    '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12,
}

_CARD_SUIT_RANK_CACHE = {}
for r in _RANK_IDX:
    for s in _SUIT_IDX:
        c_str = r + s
        _CARD_SUIT_RANK_CACHE[c_str] = (_SUIT_IDX[s], _RANK_IDX[r])


def get_board_texture(hole: list, board: list) -> tuple[float, float, float, float, float]:
    """Return 5 board-texture features in [0, 1].

    Features (in order):
    1. board_flush_texture  — max same-suit count on board / 4
    2. board_paired         — 1.0 if any rank appears ≥ 2× on board
    3. hero_flush_draw      — 1.0 if hero has ≥ 4 cards to a flush
    4. board_high_rank      — highest rank on board normalised to [0, 1]
    5. board_connectedness  — fraction of a 5-rank window covered by board ranks
    """
    if not board:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    board_suits, board_ranks = [], []
    for card in board:
        if isinstance(card, str):
            res = _CARD_SUIT_RANK_CACHE.get(card)
            if res:
                board_suits.append(res[0])
                board_ranks.append(res[1])

    if not board_suits:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    suit_counts = Counter(board_suits)
    max_suit_count = max(suit_counts.values())
    board_flush_texture = max_suit_count / 4.0

    rank_counts = Counter(board_ranks)
    board_paired = 1.0 if max(rank_counts.values()) >= 2 else 0.0

    hero_flush_draw = 0.0
    if hole and len(hole) >= 2:
        hole_suits = []
        for card in hole:
            if isinstance(card, str):
                res = _CARD_SUIT_RANK_CACHE.get(card)
                if res:
                    hole_suits.append(res[0])
        if hole_suits:
            combined = Counter(hole_suits + board_suits)
            dominant = max(combined, key=combined.get)
            total = combined[dominant]
            hero_contrib = sum(1 for s in hole_suits if s == dominant)
            if total >= 4 and hero_contrib >= 1:
                hero_flush_draw = 1.0

    board_high_rank = max(board_ranks) / 12.0

    unique_ranks = set(board_ranks)
    max_in_window = max(
        sum(1 for r in unique_ranks if start <= r < start + 5)
        for start in range(9)
    )
    board_connectedness = max_in_window / 5.0

    return board_flush_texture, board_paired, hero_flush_draw, board_high_rank, board_connectedness


def encode_state(state: dict, player_id: str) -> np.ndarray:
    obs = np.zeros(OBSERVATION_SHAPE, dtype=np.float32)

    my_status = None
    players = state.get("players", []) or []
    for p in players:
        if p.get("id") == player_id:
            my_status = p
            break

    if my_status is None:
        return obs

    starting_chips = float(state.get("starting_chips", 1000) or 1000)

    hole_cards = my_status.get("hole", []) or []
    for i in range(2):
        if i < len(hole_cards):
            card_str = hole_cards[i]
            if isinstance(card_str, dict):
                r = card_str.get("Rank")
                s = card_str.get("Suit")
                rank_norm = (r - 2) / 12.0 if r is not None else -1.0
                suit_norm = s / 3.0 if s is not None else -1.0
            else:
                rank_norm, suit_norm = parse_card(str(card_str))
            obs[i * 2] = rank_norm
            obs[i * 2 + 1] = suit_norm
        else:
            obs[i * 2] = -1.0
            obs[i * 2 + 1] = -1.0

    board_cards = state.get("board", []) or []
    for i in range(5):
        idx = 4 + i * 2
        if i < len(board_cards):
            card_str = board_cards[i]
            if isinstance(card_str, dict):
                r = card_str.get("Rank")
                s = card_str.get("Suit")
                rank_norm = (r - 2) / 12.0 if r is not None else -1.0
                suit_norm = s / 3.0 if s is not None else -1.0
            else:
                rank_norm, suit_norm = parse_card(str(card_str))
            obs[idx] = rank_norm
            obs[idx + 1] = suit_norm
        else:
            obs[idx] = -1.0
            obs[idx + 1] = -1.0

    obs[14] = float(state.get("pot", 0)) / starting_chips
    current_bet = float(state.get("current_bet", 0))
    my_bet = float(my_status.get("bet", 0))
    to_call = max(0.0, current_bet - my_bet)
    obs[15] = to_call / starting_chips
    obs[16] = float(state.get("small_blind", 10)) / starting_chips
    obs[17] = float(state.get("big_blind", 20)) / starting_chips
    obs[18] = float(my_status.get("chips", 0)) / starting_chips
    obs[19] = my_bet / starting_chips

    my_seat = int(my_status.get("seat", 0))
    dealer_idx = int(state.get("dealer_idx", 0))

    if len(players) <= 2:
        is_sb = my_status.get("is_small_blind", False)
        my_seat_mapped = 0
        dealer_idx_mapped = 1 if is_sb else 0
    else:
        my_seat_mapped = my_seat
        dealer_idx_mapped = dealer_idx

    obs[20] = my_seat_mapped / 7.0
    obs[21] = dealer_idx_mapped / 7.0
    obs[22] = ((my_seat_mapped - dealer_idx_mapped) % 8) / 7.0

    phase = state.get("phase", "Pre-Flop")
    phases = ["Pre-Flop", "Flop", "Turn", "River", "Showdown"]
    phase_idx = phases.index(phase) if phase in phases else 0
    obs[23 + phase_idx] = 1.0

    other_players = [p for p in players if p.get("id") != player_id]
    other_players.sort(key=lambda x: int(x.get("seat", 0)))

    for i in range(7):
        idx = 28 + i * 5
        if i < len(other_players):
            op = other_players[i]
            obs[idx] = float(op.get("chips", 0)) / starting_chips
            obs[idx + 1] = float(op.get("bet", 0)) / starting_chips
            obs[idx + 2] = 0.0 if op.get("folded", False) else 1.0
            obs[idx + 3] = 1.0 if op.get("all_in", False) else 0.0
            obs[idx + 4] = 0.0 if op.get("sitting_out", False) else 1.0
        else:
            obs[idx] = 0.0
            obs[idx + 1] = 0.0
            obs[idx + 2] = 0.0
            obs[idx + 3] = 0.0
            obs[idx + 4] = 0.0

    obs[63] = get_hand_strength(hole_cards, board_cards)

    (
        obs[64],
        obs[65],
        obs[66],
        obs[67],
        obs[68],
    ) = get_board_texture(hole_cards, board_cards)

    phases_list = ["Pre-Flop", "Flop", "Turn", "River", "Showdown"]
    current_phase_idx = phases_list.index(phase) if phase in phases_list else 0
    phase_idx_clamped = min(3, current_phase_idx)

    aggressor_idx = state.get("aggressor_idx", -1)
    street_action_counts = state.get("street_action_counts", {}) or {}
    street_last_actions = state.get("street_last_actions", [{} for _ in range(4)]) or [{} for _ in range(4)]
    street_contributions = state.get("street_contributions", [{} for _ in range(4)]) or [{} for _ in range(4)]
    session_stats = state.get("session_stats", {}) or {}

    pot_val = float(state.get("pot", 0))
    pot_odds = to_call / (pot_val + to_call) if to_call > 0 else 0.0
    obs[69] = pot_odds

    my_chips = float(my_status.get("chips", 0))
    spr = my_chips / max(1.0, pot_val)
    obs[70] = min(spr, 20.0) / 20.0

    num_players_active = len(players)
    if aggressor_idx == -1 or num_players_active <= 1:
        obs[71] = -1.0
    else:
        obs[71] = ((aggressor_idx - my_seat) % num_players_active) / float(num_players_active - 1)

    current_street_counts = street_action_counts.get(phase, {}) or {}
    obs[72] = min(float(current_street_counts.get("raise", 0)), 5.0) / 5.0
    obs[73] = min(float(current_street_counts.get("call", 0)), 5.0) / 5.0
    obs[74] = min(float(current_street_counts.get("fold", 0)), 5.0) / 5.0

    my_last_action_code = street_last_actions[phase_idx_clamped].get(player_id, -1)
    obs[75] = (my_last_action_code + 1.0) / 4.0

    for i in range(7):
        if i < len(other_players):
            op = other_players[i]
            op_id = op.get("id")
            op_last_action_code = street_last_actions[phase_idx_clamped].get(op_id, -1)
            obs[76 + i] = (op_last_action_code + 1.0) / 4.0
        else:
            obs[76 + i] = 0.0

    for i in range(7):
        idx = 83 + i * 3
        if i < len(other_players):
            op = other_players[i]
            op_id = op.get("id")
            stats = session_stats.get(op_id, {}) or {}

            vpip_act = float(stats.get("vpip_actions", 0))
            vpip_opp = float(stats.get("vpip_opportunities", 1))
            vpip = vpip_act / max(1.0, vpip_opp)

            pfr_act = float(stats.get("pfr_actions", 0))
            pfr_opp = float(stats.get("pfr_opportunities", 1))
            pfr = pfr_act / max(1.0, pfr_opp)

            calls = float(stats.get("calls", 0))
            bets_or_raises = float(stats.get("bets_or_raises", 0))
            af = bets_or_raises / max(1.0, calls)
            af_norm = af / (af + 1.0)

            obs[idx] = vpip
            obs[idx + 1] = pfr
            obs[idx + 2] = af_norm
        else:
            obs[idx] = 0.0
            obs[idx + 1] = 0.0
            obs[idx + 2] = 0.0

    player_by_rel_seat = {}
    player_by_rel_seat[0] = player_id

    sorted_players = sorted(players, key=lambda x: int(x.get("seat", 0)))
    my_sorted_idx = -1
    for idx_sp, p in enumerate(sorted_players):
        if p.get("id") == player_id:
            my_sorted_idx = idx_sp
            break

    if my_sorted_idx != -1 and len(players) > 0:
        for r in range(1, 8):
            if r < len(players):
                target_p = sorted_players[(my_sorted_idx + r) % len(players)]
                player_by_rel_seat[r] = target_p.get("id")

    for s_idx in range(4):
        for r in range(8):
            o_idx = 104 + s_idx * 16 + r * 2
            pid = player_by_rel_seat.get(r)
            if pid is not None:
                act_code = street_last_actions[s_idx].get(pid, -1)
                contrib = street_contributions[s_idx].get(pid, 0.0)
                obs[o_idx] = (act_code + 1.0) / 4.0
                obs[o_idx + 1] = float(contrib) / starting_chips
            else:
                obs[o_idx] = 0.0
                obs[o_idx + 1] = 0.0

    return obs
