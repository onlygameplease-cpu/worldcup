"""
Backtest engine cho Poisson Algorithmic Picks.
Chạy: python backtest_poisson.py
Output: Bảng kết quả ROI theo từng market (Totals, Spreads, BTTS).
"""
import pandas as pd
import numpy as np
from metrics import extract_adv_metrics
from poisson import predict_poisson_market_probs, filter_main_odds_lines

def simulate_match_odds(home_score, away_score, home_xg, away_xg):
    """Tạo synthetic odds từ kết quả thực tế.
    
    Vì không có odds lịch sử, ta dùng Poisson của chính model
    với xG thật để tạo "fair odds", rồi thêm margin 5% 
    để mô phỏng bookmaker.
    """
    total_goals = home_score + away_score
    margin = 1.05  # 5% vigorish
    
    # Totals: tạo các line từ 1.5 đến 5.5
    totals_markets = []
    for line in [1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.5]:
        # Fair probability from actual data distribution
        p_over = max(0.1, min(0.9, (home_xg + away_xg - line + 0.5) / 3.0))
        p_under = 1.0 - p_over
        over_odds = round(margin / p_over, 3)
        under_odds = round(margin / p_under, 3)
        totals_markets.append({
            "key": "totals",
            "outcomes": [
                {"name": "Over", "price": str(over_odds), "point": line},
                {"name": "Under", "price": str(under_odds), "point": line}
            ]
        })
    
    # Spreads: tạo các line handicap
    spreads_markets = []
    for hcp in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        p_home_cover = max(0.1, min(0.9, 0.5 + (home_xg - away_xg + hcp) / 4.0))
        p_away_cover = 1.0 - p_home_cover
        h_odds = round(margin / p_home_cover, 3)
        a_odds = round(margin / p_away_cover, 3)
        spreads_markets.append({
            "key": "spreads",
            "outcomes": [
                {"name": "home", "price": str(h_odds), "point": hcp},
                {"name": "away", "price": str(a_odds), "point": -hcp}
            ]
        })
    
    bookmakers = [{"key": "synthetic", "title": "Synthetic", "markets": totals_markets + spreads_markets}]
    return bookmakers


def settle_bet(pick_str, home_score, away_score):
    """Trả về P/L cho 1 unit bet dựa trên kết quả thực tế.
    
    Returns: (pnl, odds) tuple
    """
    import re
    total = home_score + away_score
    net_home = home_score - away_score
    
    if not pick_str:
        return 0, 0
    
    # Parse odds
    odds_match = re.search(r'@([\d.]+)', pick_str)
    if not odds_match:
        return 0, 0
    odds = float(odds_match.group(1))
    
    # Parse line
    line_match = re.search(r'([\d.]+)', pick_str.split('@')[0])
    if not line_match:
        return 0, 0
    line = float(line_match.group(1))
    
    if pick_str.startswith("Over"):
        won = total > line
        half_won = False  # simplified
    elif pick_str.startswith("Under"):
        won = total < line
        half_won = False
    elif "home" in pick_str.lower():
        adj = net_home + line if '-' not in pick_str.split('@')[0] else net_home - line
        won = adj > 0
        half_won = False
    elif "away" in pick_str.lower():
        adj = -net_home + line
        won = adj > 0
        half_won = False
    else:
        return 0, 0
    
    if won:
        return odds - 1.0, odds
    else:
        return -1.0, odds


def run_backtest():
    df = pd.read_csv("dataworldcup/advanced_stats.csv")
    
    teams = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
    
    results = []
    
    for idx, match in df.iterrows():
        home = match["HomeTeam"]
        away = match["AwayTeam"]
        home_score = int(match["HomeScore"])
        away_score = int(match["AwayScore"])
        
        # Get stats from all OTHER matches (leave-one-out)
        other = df.drop(idx)
        home_matches = other[(other["HomeTeam"] == home) | (other["AwayTeam"] == home)]
        away_matches = other[(other["HomeTeam"] == away) | (other["AwayTeam"] == away)]
        
        if home_matches.empty or away_matches.empty:
            continue
        
        home_stats = extract_adv_metrics(home_matches, home)
        away_stats = extract_adv_metrics(away_matches, away)
        
        home_xg = float(home_stats.get("xG") or home_stats.get("avg_xG") or 1.5)
        away_xg = float(away_stats.get("xG") or away_stats.get("avg_xG") or 1.2)
        
        # Generate synthetic odds
        bookmakers = simulate_match_odds(home_score, away_score, home_xg, away_xg)
        
        # Run Poisson
        ev_funcs = predict_poisson_market_probs(home_xg, away_xg, home_stats=home_stats, away_stats=away_stats)
        t_out, s_out, btts_out, picks = filter_main_odds_lines(bookmakers, ev_funcs)
        
        t_pick = picks.get("totals_pick")
        s_pick = picks.get("spreads_pick")
        
        t_pnl, t_odds = settle_bet(t_pick, home_score, away_score)
        s_pnl, s_odds = settle_bet(s_pick, home_score, away_score)
        
        results.append({
            "match": f"{home} vs {away}",
            "score": f"{home_score}-{away_score}",
            "home_xg_raw": home_stats.get("xG"),
            "away_xg_raw": away_stats.get("xG"),
            "home_xg_sos": home_stats.get("avg_xG"),
            "away_xg_sos": away_stats.get("avg_xG"),
            "totals_pick": t_pick,
            "totals_pnl": t_pnl,
            "spreads_pick": s_pick,
            "spreads_pnl": s_pnl,
        })
    
    rdf = pd.DataFrame(results)
    
    # Summary
    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total matches tested: {len(rdf)}")
    print()
    
    # Totals
    t_bets = rdf[rdf["totals_pick"].notna()]
    if not t_bets.empty:
        t_wins = (t_bets["totals_pnl"] > 0).sum()
        t_total = len(t_bets)
        t_roi = t_bets["totals_pnl"].sum() / t_total * 100
        t_over = t_bets[t_bets["totals_pick"].str.startswith("Over", na=False)]
        t_under = t_bets[t_bets["totals_pick"].str.startswith("Under", na=False)]
        print(f"TOTALS: {t_wins}/{t_total} wins ({t_wins/t_total*100:.1f}%) | ROI: {t_roi:.2f}%")
        print(f"  Over picks: {len(t_over)} | Under picks: {len(t_under)}")
    
    # Spreads
    s_bets = rdf[rdf["spreads_pick"].notna()]
    if not s_bets.empty:
        s_wins = (s_bets["spreads_pnl"] > 0).sum()
        s_total = len(s_bets)
        s_roi = s_bets["spreads_pnl"].sum() / s_total * 100
        s_home = s_bets[s_bets["spreads_pick"].str.contains("home", case=False, na=False)]
        s_away = s_bets[s_bets["spreads_pick"].str.contains("away", case=False, na=False)]
        print(f"SPREADS: {s_wins}/{s_total} wins ({s_wins/s_total*100:.1f}%) | ROI: {s_roi:.2f}%")
        print(f"  Home picks: {len(s_home)} | Away picks: {len(s_away)}")
    
    print()
    print("BIAS CHECK:")
    if not t_bets.empty:
        print(f"  Over/Under ratio: {len(t_over)}/{len(t_under)}")
    if not s_bets.empty:
        print(f"  Home/Away ratio: {len(s_home)}/{len(s_away)}")
    
    # Save detailed results
    rdf.to_csv("backtest_results.csv", index=False)
    print(f"\nDetailed results saved to backtest_results.csv")
    
    return rdf

if __name__ == "__main__":
    run_backtest()
