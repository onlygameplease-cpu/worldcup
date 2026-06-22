import numpy as np
from scipy.stats import poisson

def predict_poisson_market_probs(home_ema_xg, away_ema_xg, max_goals=10):
    home_goals_prob = poisson.pmf(np.arange(max_goals), home_ema_xg)
    away_goals_prob = poisson.pmf(np.arange(max_goals), away_ema_xg)
    score_matrix = np.outer(home_goals_prob, away_goals_prob)
    
    def calc_ev_asian(odds, p_win, p_half_win, p_push, p_half_lose, p_lose):
        return (p_win * (odds - 1)) + (p_half_win * ((odds - 1) / 2.0)) + (p_push * 0) + (p_half_lose * -0.5) + (p_lose * -1.0)

    def get_over_under_ev(line, over_odds, under_odds):
        p_over_win, p_over_hw, p_push, p_over_hl, p_over_lose = 0,0,0,0,0
        for h in range(max_goals):
            for a in range(max_goals):
                total = h + a
                prob = score_matrix[h, a]
                if line % 0.5 != 0:
                    lower = line - 0.25
                    upper = line + 0.25
                    if total > upper:
                        p_over_win += prob
                    elif total == upper:
                        p_over_hw += prob
                    elif total == lower:
                        p_over_hl += prob
                    else:
                        p_over_lose += prob
                else:
                    if total > line:
                        p_over_win += prob
                    elif total == line:
                        p_push += prob
                    else:
                        p_over_lose += prob
                        
        p_under_win = p_over_lose
        p_under_hw = p_over_hl
        p_under_hl = p_over_hw
        p_under_lose = p_over_win
        
        ev_over = calc_ev_asian(over_odds, p_over_win, p_over_hw, p_push, p_over_hl, p_over_lose)
        ev_under = calc_ev_asian(under_odds, p_under_win, p_under_hw, p_push, p_under_hl, p_under_lose)
        return {"over_ev": ev_over, "under_ev": ev_under}

    def get_handicap_ev(home_line, home_odds, away_odds):
        p_home_win, p_home_hw, p_push, p_home_hl, p_home_lose = 0,0,0,0,0
        for h in range(max_goals):
            for a in range(max_goals):
                net_home = h + home_line
                prob = score_matrix[h, a]
                if home_line % 0.5 != 0:
                    lower = home_line - 0.25
                    upper = home_line + 0.25
                    net_lower = h + lower
                    net_upper = h + upper
                    if net_lower > a and net_upper > a:
                        p_home_win += prob
                    elif net_lower == a and net_upper > a:
                        p_home_hw += prob
                    elif net_lower < a and net_upper == a:
                        p_home_hl += prob
                    else:
                        p_home_lose += prob
                else:
                    if net_home > a:
                        p_home_win += prob
                    elif net_home == a:
                        p_push += prob
                    else:
                        p_home_lose += prob
                        
        p_away_win = p_home_lose
        p_away_hw = p_home_hl
        p_away_hl = p_home_hw
        p_away_lose = p_home_win
        
        ev_home = calc_ev_asian(home_odds, p_home_win, p_home_hw, p_push, p_home_hl, p_home_lose)
        ev_away = calc_ev_asian(away_odds, p_away_win, p_away_hw, p_push, p_away_hl, p_away_lose)
        return {"home_ev": ev_home, "away_ev": ev_away}

    return {
        "get_over_under_ev": get_over_under_ev,
        "get_handicap_ev": get_handicap_ev
    }

def filter_main_odds_lines(bookmakers, ev_funcs):
    """
    Returns the best totals and spreads outcomes based on EV.
    """
    best_total_outcomes = None
    best_total_ev = -999.0
    
    best_spread_outcomes = None
    best_spread_ev = -999.0
    
    for book in bookmakers:
        for market in (book.get("markets") or []):
            key = (market.get("key") or "").strip().lower()
            outcomes = market.get("outcomes") or []
            if len(outcomes) < 2:
                continue
                
            if key in {"totals", "total", "over_under", "overunder", "ou"}:
                over = next((x for x in outcomes if str(x.get("name", "")).lower() == "over"), None)
                under = next((x for x in outcomes if str(x.get("name", "")).lower() == "under"), None)
                if over and under and over.get("point") is not None and over.get("price") and under.get("price"):
                    o_price = float(over.get("price"))
                    u_price = float(under.get("price"))
                    # Lọc kèo cân tiền (Ví dụ: 1.80 đến 2.05)
                    if 1.80 <= o_price <= 2.05 and 1.80 <= u_price <= 2.05:
                        evs = ev_funcs["get_over_under_ev"](float(over.get("point")), o_price, u_price)
                        max_ev = max(evs["over_ev"], evs["under_ev"])
                        if max_ev > best_total_ev:
                            best_total_ev = max_ev
                            best_total_outcomes = outcomes
                            
            elif key in {"spreads", "spread", "asian_handicap", "handicap", "ah"}:
                # Usually home is first, away is second, or we can check names. Let's assume point is available
                # Actually outcomes often have name = "Home Team" and "Away Team".
                if outcomes[0].get("point") is not None and outcomes[1].get("point") is not None:
                    p1 = float(outcomes[0].get("price", 0))
                    p2 = float(outcomes[1].get("price", 0))
                    if 1.80 <= p1 <= 2.05 and 1.80 <= p2 <= 2.05:
                        # Find home team line. Typically positive for underdog, negative for favorite.
                        # Wait, we need to know which is home! Let's just assume outcomes[0] is home for this test
                        home_line = float(outcomes[0].get("point"))
                        evs = ev_funcs["get_handicap_ev"](home_line, p1, p2)
                        max_ev = max(evs["home_ev"], evs["away_ev"])
                        if max_ev > best_spread_ev:
                            best_spread_ev = max_ev
                            best_spread_outcomes = outcomes

    return best_total_outcomes, best_spread_outcomes

# Test with dummy bookmakers data
dummy_bookmakers = [
    {
        "markets": [
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": 2.5, "price": 1.95},
                    {"name": "Under", "point": 2.5, "price": 1.85}
                ]
            },
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": 2.25, "price": 1.70},
                    {"name": "Under", "point": 2.25, "price": 2.10}
                ]
            },
            {
                "key": "spreads",
                "outcomes": [
                    {"name": "Home", "point": -0.5, "price": 1.90},
                    {"name": "Away", "point": 0.5, "price": 1.90}
                ]
            }
        ]
    }
]

ev_funcs = predict_poisson_market_probs(1.5, 1.2)
t, s = filter_main_odds_lines(dummy_bookmakers, ev_funcs)
print("Best Totals:", t)
print("Best Spreads:", s)
