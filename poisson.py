import numpy as np
import pandas as pd
import math
from scipy.stats import poisson
from metrics import check_physical_dominance

CIRCUIT_BREAKER_CONFIG = {
    "tier_1_heavy_possession": {
        "possession_floor": 65.0,
        "sq_ceiling": 0.09,
        "fallback_action": "NO BET"
    },
    "default_context": {
        "possession_floor": 60.0,
        "sq_ceiling": 0.075,
        "fallback_action": "NO BET"
    }
}

def calculate_gk_ema(history_prevented_goals, span=3):
    if not history_prevented_goals or len(history_prevented_goals) == 0:
        return 0.0
    series = pd.Series(history_prevented_goals)
    ema_series = series.ewm(span=span, adjust=False).mean()
    return float(ema_series.iloc[-1])

def evaluate_discipline_risk(avg_fouls, avg_yellow_cards, avg_clearances):
    risk_score = 0.0
    if avg_fouls > 13.0: 
        risk_score += 0.35
    if avg_yellow_cards > 2.0: 
        risk_score += 0.35
    if avg_clearances > 25.0: 
        risk_score += 0.2
    return risk_score

def predict_poisson_market_probs(home_ema_xg, away_ema_xg, home_stats=None, away_stats=None, max_goals=10):
    reasons = []
    # --- ADVANCED QUANT ENGINE OVERLAY ---
    if home_stats and away_stats:
        home_gk_history = home_stats.get("history_goals_prevented") or []
        away_gk_history = away_stats.get("history_goals_prevented") or []
        if not home_gk_history: home_gk_history = [float(home_stats.get("avg_Goals_Prevented") or 0.0)]
        if not away_gk_history: away_gk_history = [float(away_stats.get("avg_Goals_Prevented") or 0.0)]
        
        home_gk_perf = calculate_gk_ema(home_gk_history)
        away_gk_perf = calculate_gk_ema(away_gk_history)

        home_eff = float(home_stats.get("avg_Shooting_Efficiency") or 0.0)
        away_eff = float(away_stats.get("avg_Shooting_Efficiency") or 0.0)

        # Lambda tuning
        home_ema_xg = max(0.2, home_ema_xg + home_eff - away_gk_perf)
        away_ema_xg = max(0.2, away_ema_xg + away_eff - home_gk_perf)
        reasons.append({"rule": "gk_efficiency_adj", "impact": f"Home Lambda -> {home_ema_xg:.2f}", "severity": "medium"})
        reasons.append({"rule": "gk_efficiency_adj", "impact": f"Away Lambda -> {away_ema_xg:.2f}", "severity": "medium"})

        # Discipline risk
        home_fouls = float(home_stats.get("avg_Fouls") or 10.0)
        home_yellows = float(home_stats.get("avg_Yellow_Cards") or 1.0)
        home_clearances = float(home_stats.get("avg_Clearances") or 15.0)
        home_disc_risk = evaluate_discipline_risk(home_fouls, home_yellows, home_clearances)

        away_fouls = float(away_stats.get("avg_Fouls") or 10.0)
        away_yellows = float(away_stats.get("avg_Yellow_Cards") or 1.0)
        away_clearances = float(away_stats.get("avg_Clearances") or 15.0)
        away_disc_risk = evaluate_discipline_risk(away_fouls, away_yellows, away_clearances)
        
        if (home_disc_risk + away_disc_risk) >= 0.6:
            home_ema_xg *= 1.08
            away_ema_xg *= 1.08
            reasons.append({"rule": "discipline_risk_fat_tail", "impact": "Lambda x 1.08", "severity": "high"})

        # --- CENTRAL CIRCUIT BREAKER (STERILE POSSESSION IRON WALL) ---
        home_possession = float(home_stats.get("avg_Possession") or 50.0)
        away_possession = float(away_stats.get("avg_Possession") or 50.0)
        home_xg_conceded = float(away_stats.get("avg_xG_Conceded") or 1.25)
        away_xg_conceded = float(home_stats.get("avg_xG_Conceded") or 1.25)

        home_aer = home_ema_xg / max(away_xg_conceded, 0.1)
        away_aer = away_ema_xg / max(home_xg_conceded, 0.1)

        if home_possession > 60.0 and home_aer < 1.15:
            home_ema_xg *= 0.85
            reasons.append({"rule": "sterile_possession_iron_wall", "impact": "Home Lambda scaled by 0.85", "severity": "high"})
            
        if away_possession > 60.0 and away_aer < 1.15:
            away_ema_xg *= 0.85
            reasons.append({"rule": "sterile_possession_iron_wall", "impact": "Away Lambda scaled by 0.85", "severity": "high"})
        # --------------------------------------------------------------

        # 1. GOALKEEPING VOLATILITY FILTER
        is_home_fav = float(home_stats.get("latest_Rank") or 100) < float(away_stats.get("latest_Rank") or 100)
        underdog_stats = away_stats if is_home_fav else home_stats
        underdog_side = "Away" if is_home_fav else "Home"
        fav_side = "Home" if is_home_fav else "Away"
        underdog_gk_history = underdog_stats.get("history_goals_prevented") or []
        if underdog_gk_history:
            recent_gp_mean = np.mean(underdog_gk_history[:3])
            if recent_gp_mean > 0.10:
                dampening_factor = 0.85
                if is_home_fav:
                    home_ema_xg *= dampening_factor
                else:
                    away_ema_xg *= dampening_factor
                reasons.append({
                    "rule": "goalkeeping_volatility_filter",
                    "impact": f"{fav_side} Lambda bị hạ nhiệt do thủ môn {underdog_side} đang vào phom (GP 3 trận gần nhất: {recent_gp_mean:.2f})",
                    "severity": "critical"
                })

        # 2. TRANSITION WEIGHT FILTER
        away_passes_final_third = float(away_stats.get("avg_Passes_Final_Third") or 0)
        if away_passes_final_third > 130:
            boost_factor = 1.25
            away_ema_xg *= boost_factor
            reasons.append({
                "rule": "transition_weight_filter",
                "impact": f"Away Lambda được tăng cường lên {away_ema_xg:.2f} do năng lực chuyển trạng thái cực mạnh khi đá sân khách",
                "severity": "high"
            })
            
    # --- STERILE POSSESSION TRAP OVERLAY ---
    if home_stats and away_stats:
        rank_gap = float(away_stats.get("latest_Rank") or 100) - float(home_stats.get("latest_Rank") or 100)
        home_sample = float(home_stats.get("Sample_Size") or 1)
        away_sample = float(away_stats.get("Sample_Size") or 1)
        
        # Home is favorite
        if rank_gap > 30:
            away_xg_conceded = float(away_stats.get("avg_xG_Conceded") or 1.5)
            if away_xg_conceded < 0.80 and away_sample >= 3:
                home_ema_xg *= 0.85
                reasons.append({"rule": "sterile_possession_iron_wall", "impact": "Home Lambda x 0.85", "severity": "high"})
                
        # Away is favorite
        elif rank_gap < -30:
            home_xg_conceded = float(home_stats.get("avg_xG_Conceded") or 1.5)
            if home_xg_conceded < 0.80 and home_sample >= 3:
                away_ema_xg *= 0.85
                reasons.append({"rule": "sterile_possession_iron_wall", "impact": "Away Lambda x 0.85", "severity": "high"})
    # -------------------------------------

    home_goals_prob = poisson.pmf(np.arange(max_goals), home_ema_xg)
    away_goals_prob = poisson.pmf(np.arange(max_goals), away_ema_xg)
    score_matrix = np.outer(home_goals_prob, away_goals_prob)
    
    if home_stats and away_stats:
        def calc_dominance(stats_a, stats_b):
            t_a = float(stats_a.get("avg_Touches_In_Box") or stats_a.get("Touches_In_Box") or 0)
            t_b = float(stats_b.get("avg_Touches_In_Box") or stats_b.get("Touches_In_Box") or 0)
            t_ratio = t_a / max(t_b, 1.0)
            
            xgot_a = float(stats_a.get("avg_xGOT") or stats_a.get("xGOT") or 0)
            xgot_b = float(stats_b.get("avg_xGOT") or stats_b.get("xGOT") or 0)
            xgot_ratio = xgot_a / max(xgot_b, 0.1)
            
            p_a = float(stats_a.get("avg_Possession") or stats_a.get("Possession") or 50)
            p_b = float(stats_b.get("avg_Possession") or stats_b.get("Possession") or 50)
            p_ratio = p_a / max(p_b, 1.0)
            
            return (t_ratio + xgot_ratio + p_ratio) / 3.0

        d_home = calc_dominance(home_stats, away_stats)
        d_away = calc_dominance(away_stats, home_stats)
        
        rho_base = -0.13
        k = 3.0
        max_d = max(d_home, d_away)
        rho_dyn = rho_base * (1.0 - math.tanh(max_d / k))
        
        l = home_ema_xg
        m = away_ema_xg
        score_matrix[0,0] *= max(0.0, 1.0 - rho_dyn * l * m)
        score_matrix[1,0] *= max(0.0, 1.0 + rho_dyn * m)
        score_matrix[0,1] *= max(0.0, 1.0 + rho_dyn * l)
        score_matrix[1,1] *= max(0.0, 1.0 - rho_dyn)
        
        def calc_b(stats_fav, stats_dog):
            rank_fav = stats_fav.get("latest_Rank")
            rank_dog = stats_dog.get("latest_Rank")
            if rank_fav is None or rank_dog is None or pd.isna(rank_fav) or pd.isna(rank_dog):
                reasons.append({"rule": "handicap_bias", "impact": "Ignored due to missing Rank", "severity": "low"})
                return 0.0
                
            delta_r = max(0, float(rank_dog) - float(rank_fav))
            
            # --- PHYSICAL DOMINANCE OVERRIDE ---
            if check_physical_dominance(stats_fav, stats_dog):
                delta_r = 0.0
                reasons.append({"rule": "physical_dominance_override", "impact": "Handicap Bias Disabled", "severity": "high"})
            # -----------------------------------
            
            t_fav = float(stats_fav.get("avg_Touches_In_Box") or stats_fav.get("Touches_In_Box") or 0)
            t_dog = float(stats_dog.get("avg_Touches_In_Box") or stats_dog.get("Touches_In_Box") or 0)
            t_ratio = t_fav / max(t_dog, 1.0)
            
            if t_ratio <= 1.0: return 0.0
            bias = (delta_r / 100.0) * math.log(t_ratio)
            if bias > 0:
                reasons.append({"rule": "handicap_bias", "impact": f"Fav Bias +{bias:.3f}", "severity": "medium"})
            return bias
            
        b_home = 0.0
        b_away = 0.0
        if home_ema_xg > away_ema_xg:
            b_home = calc_b(home_stats, away_stats)
        else:
            b_away = calc_b(away_stats, home_stats)
            
        if b_home > 0:
            floor_l = int(math.floor(home_ema_xg))
            for x in range(floor_l + 1, max_goals):
                for y in range(max_goals):
                    score_matrix[x, y] *= (1.0 + b_home) ** (x - floor_l)
        
        if b_away > 0:
            floor_m = int(math.floor(away_ema_xg))
            for y in range(floor_m + 1, max_goals):
                for x in range(max_goals):
                    score_matrix[x, y] *= (1.0 + b_away) ** (y - floor_m)
                    
        score_matrix /= np.sum(score_matrix)
    
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

    def get_btts_ev(yes_odds, no_odds):
        p_btts_yes = np.sum(score_matrix[1:, 1:])
        p_btts_no = 1.0 - p_btts_yes
        
        ev_yes = calc_ev_asian(yes_odds, p_btts_yes, 0, 0, 0, p_btts_no)
        ev_no = calc_ev_asian(no_odds, p_btts_no, 0, 0, 0, p_btts_yes)
        return {"yes_ev": ev_yes, "no_ev": ev_no}

    return {
        "get_over_under_ev": get_over_under_ev,
        "get_handicap_ev": get_handicap_ev,
        "get_btts_ev": get_btts_ev,
        "reason_trace": reasons
    }

def filter_main_odds_lines(bookmakers, ev_funcs, is_blowout_risk=False):
    best_total_outcomes = None
    best_total_ev = -999.0
    best_total_pick = None
    best_spread_outcomes = None
    best_spread_ev = -999.0
    best_spread_pick = None
    best_btts_outcomes = None
    best_btts_ev = -999.0
    best_btts_pick = None
    
    debug_lines = []
    MIN_EV = 0.025
    
    for book in bookmakers:
        for market in (book.get("markets") or []):
            key = (market.get("key") or "").strip().lower()
            outcomes = market.get("outcomes") or []
            if len(outcomes) < 2: continue
            
            by_point = {}
            for x in outcomes:
                pt_raw = x.get("point")
                if pt_raw is not None:
                    try:
                        pt_val = abs(float(pt_raw))
                        by_point.setdefault(pt_val, []).append(x)
                    except ValueError:
                        by_point.setdefault(pt_raw, []).append(x)
            
            if not by_point:
                by_point[None] = outcomes
            
            if key in {"totals", "total", "over_under", "overunder", "ou"}:
                # MAIN LINE SELECTION
                main_pt = None
                min_score = 999.0
                for pt, pts_outcomes in by_point.items():
                    if pt is None: continue
                    over = next((x for x in pts_outcomes if str(x.get("name", "")).lower() == "over"), None)
                    under = next((x for x in pts_outcomes if str(x.get("name", "")).lower() == "under"), None)
                    if over and under and over.get("price") and under.get("price"):
                        o_price = float(over.get("price"))
                        u_price = float(under.get("price"))
                        if 1.50 <= o_price <= 2.50 and 1.50 <= u_price <= 2.50:
                            score = abs(o_price - u_price) + 0.25 * abs(((o_price + u_price) / 2) - 1.90)
                            if score < min_score:
                                min_score = score
                                main_pt = pt
                
                if main_pt is not None:
                    pts_outcomes = by_point[main_pt]
                    over = next((x for x in pts_outcomes if str(x.get("name", "")).lower() == "over"), None)
                    under = next((x for x in pts_outcomes if str(x.get("name", "")).lower() == "under"), None)
                    o_price = float(over.get("price"))
                    u_price = float(under.get("price"))
                    debug_lines.append(f"MAIN Totals {main_pt}: {o_price} / {u_price}")
                    
                    evs = ev_funcs["get_over_under_ev"](float(main_pt), o_price, u_price)
                    is_over = evs["over_ev"] > evs["under_ev"]
                    
                    if not is_over and is_blowout_risk and float(main_pt) <= 3.25:
                        ev_funcs.setdefault("reason_trace", []).append({"rule": "circuit_breaker_lock", "impact": f"NO BET Under {main_pt}", "severity": "critical"})
                    else:
                        max_ev = max(evs["over_ev"], evs["under_ev"])
                        if max_ev > best_total_ev and max_ev >= MIN_EV:
                            best_total_ev = max_ev
                            best_total_outcomes = [over, under]
                            best_total_pick = f"Over {main_pt} (@{o_price})" if is_over else f"Under {main_pt} (@{u_price})"
                            
            elif key in {"spreads", "spread", "asian_handicap", "handicap", "ah"}:
                main_pt = None
                min_score = 999.0
                for pt, pts_outcomes in by_point.items():
                    if pt is None: continue
                    home_outcome = next((x for x in pts_outcomes if str(x.get("name", "")).lower() == "home"), None)
                    away_outcome = next((x for x in pts_outcomes if str(x.get("name", "")).lower() == "away"), None)
                    if not home_outcome and not away_outcome and len(pts_outcomes) >= 2:
                        home_outcome, away_outcome = pts_outcomes[0], pts_outcomes[1]
                        
                    if home_outcome and away_outcome:
                        try:
                            p1 = float(home_outcome.get("price", 0))
                            p2 = float(away_outcome.get("price", 0))
                            if 1.50 <= p1 <= 2.50 and 1.50 <= p2 <= 2.50:
                                score = abs(p1 - p2) + 0.25 * abs(((p1 + p2) / 2) - 1.90)
                                if score < min_score:
                                    min_score = score
                                    main_pt = pt
                        except: pass
                        
                if main_pt is not None:
                    pts_outcomes = by_point[main_pt]
                    home_outcome = next((x for x in pts_outcomes if str(x.get("name", "")).lower() == "home"), None)
                    away_outcome = next((x for x in pts_outcomes if str(x.get("name", "")).lower() == "away"), None)
                    if not home_outcome and not away_outcome and len(pts_outcomes) >= 2:
                        home_outcome, away_outcome = pts_outcomes[0], pts_outcomes[1]
                        
                    p1 = float(home_outcome.get("price", 0))
                    p2 = float(away_outcome.get("price", 0))
                    home_line = float(home_outcome.get("point"))
                    debug_lines.append(f"MAIN Spreads {home_line}: {p1} / {p2}")
                    
                    evs = ev_funcs["get_handicap_ev"](home_line, p1, p2)
                    is_home_pick = evs["home_ev"] > evs["away_ev"]
                    
                    locked = False
                    if is_blowout_risk:
                        is_underdog_pick = (is_home_pick and home_line > 0) or (not is_home_pick and home_line < 0)
                        if is_underdog_pick:
                            ev_funcs.setdefault("reason_trace", []).append({"rule": "circuit_breaker_lock", "impact": f"NO BET Underdog Spread {home_line}", "severity": "critical"})
                            locked = True
                            
                    if not locked:
                        max_ev = max(evs["home_ev"], evs["away_ev"])
                        if max_ev > best_spread_ev and max_ev >= MIN_EV:
                            best_spread_ev = max_ev
                            best_spread_outcomes = [home_outcome, away_outcome]
                            away_line = -home_line if home_line != 0 else 0
                            best_spread_pick = f"{home_outcome.get('name')} {home_line} (@{p1})" if is_home_pick else f"{away_outcome.get('name')} {away_line} (@{p2})"
                            
            elif key == "btts":
                yes_outcome = next((x for x in outcomes if str(x.get("name", "")).lower() == "yes"), None)
                no_outcome = next((x for x in outcomes if str(x.get("name", "")).lower() == "no"), None)
                if yes_outcome and no_outcome:
                    try:
                        y_price = float(yes_outcome.get("price"))
                        n_price = float(no_outcome.get("price"))
                        debug_lines.append(f"MAIN BTTS: Yes {y_price} / No {n_price}")
                        if "get_btts_ev" in ev_funcs:
                            evs = ev_funcs["get_btts_ev"](y_price, n_price)
                            is_yes = evs["yes_ev"] > evs["no_ev"]
                            
                            if is_yes and is_blowout_risk:
                                ev_funcs.setdefault("reason_trace", []).append({"rule": "circuit_breaker_lock", "impact": "NO BET BTTS Yes", "severity": "critical"})
                            else:
                                max_ev = max(evs["yes_ev"], evs["no_ev"])
                                if max_ev > best_btts_ev and max_ev >= MIN_EV:
                                    best_btts_ev = max_ev
                                    best_btts_outcomes = [yes_outcome, no_outcome]
                                    best_btts_pick = f"Yes (@{y_price})" if is_yes else f"No (@{n_price})"
                    except (ValueError, TypeError):
                        pass

    if best_total_pick is None: best_total_pick = "NO BET"
    if best_spread_pick is None: best_spread_pick = "NO BET"
    if best_btts_pick is None: best_btts_pick = "NO BET"

    return best_total_outcomes, best_spread_outcomes, best_btts_outcomes, {
        "totals_pick": best_total_pick, 
        "totals_ev": best_total_ev if best_total_ev != -999.0 else 0.0, 
        "spreads_pick": best_spread_pick, 
        "spreads_ev": best_spread_ev if best_spread_ev != -999.0 else 0.0, 
        "btts_pick": best_btts_pick, 
        "btts_ev": best_btts_ev if best_btts_ev != -999.0 else 0.0, 
        "debug": debug_lines,
        "reason_trace": ev_funcs.get("reason_trace", []) if isinstance(ev_funcs, dict) else []
    }
