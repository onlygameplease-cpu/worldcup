import pandas as pd
import numpy as np

def extract_epl_metrics(df: pd.DataFrame, team_name: str, lookback=10) -> dict:
    df_sub = df[((df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name)) & (df["FTHG"].notna())].copy()
    
    if len(df_sub) == 0:
        return {}
        
    df_sub = df_sub.tail(lookback)
    sample_size = len(df_sub)
    
    trust = min(sample_size / 4.0, 1.0)
    BASELINE_XG = 1.35
    BASELINE_XGOT = 1.35
    BASELINE_TOUCHES_IN_BOX = 20.0
    BASELINE_XG_CONCEDED = 1.35
    
    def get_avg(col_home, col_away, is_conceded=False):
        if col_home not in df_sub.columns or col_away not in df_sub.columns:
            return None
        vals = []
        for _, row in df_sub.iterrows():
            if row["HomeTeam"] == team_name:
                v = row[col_away] if is_conceded else row[col_home]
            else:
                v = row[col_home] if is_conceded else row[col_away]
            if pd.notna(v):
                vals.append(float(v))
        return float(np.nanmean(vals)) if vals else None

    # Try exact advanced metrics if user provides them later (Flashscore)
    # If not present, fallback to proxy metrics
    xG = get_avg("xG_Home", "xG_Away")
    if xG is None:
        shots = get_avg("HS", "AS") or 10.0
        sot = get_avg("HST", "AST") or 4.0
        # Proxy xG: each SOT is ~0.25 xG, each off-target shot is ~0.05 xG
        xG = (sot * 0.25) + ((max(0, shots - sot)) * 0.05)
        
    xG_Conceded = get_avg("xG_Away", "xG_Home") # wait, col_home is xG_Away? Let's use get_avg logic properly
    if xG_Conceded is None:
        shots_c = get_avg("HS", "AS", is_conceded=True) or 10.0
        sot_c = get_avg("HST", "AST", is_conceded=True) or 4.0
        xG_Conceded = (sot_c * 0.25) + ((max(0, shots_c - sot_c)) * 0.05)
        
    xGOT = get_avg("xGOT_Home", "xGOT_Away")
    if xGOT is None:
        # Proxy xGOT: Actual Goals Scored
        xGOT = get_avg("FTHG", "FTAG") or 1.0
        
    touches = get_avg("Touches_In_Box_Home", "Touches_In_Box_Away")
    if touches is None:
        corners = get_avg("HC", "AC") or 5.0
        shots = get_avg("HS", "AS") or 10.0
        # Proxy Touches: Corners * 1.5 + Shots * 1.2
        touches = (corners * 1.5) + (shots * 1.2)
        
    fouls = get_avg("HF", "AF") or 10.0
    yellows = get_avg("HY", "AY") or 1.5
    clearances = get_avg("Clearances_Home", "Clearances_Away") or 15.0 # proxy default
    
    goals_scored = get_avg("FTHG", "FTAG") or 1.0
    goals_conceded = get_avg("FTHG", "FTAG", is_conceded=True) or 1.0
    
    shoot_eff = get_avg("Shooting_Efficiency_Home", "Shooting_Efficiency_Away")
    if shoot_eff is None:
        shoot_eff = (goals_scored - xG) / max(xG, 0.1)
        
    gk_perf = get_avg("Goalkeeping_Overperformance_Home", "Goalkeeping_Overperformance_Away")
    if gk_perf is None:
        gk_perf = (xG_Conceded - goals_conceded) / max(xG_Conceded, 0.1)
        
    # Mean Reversion
    xG = xG * trust + BASELINE_XG * (1 - trust)
    xGOT = xGOT * trust + BASELINE_XGOT * (1 - trust)
    touches = touches * trust + BASELINE_TOUCHES_IN_BOX * (1 - trust)
    xG_Conceded = xG_Conceded * trust + BASELINE_XG_CONCEDED * (1 - trust)

    metrics = {
        "Sample_Size": sample_size,
        "xG": round(xG, 2),
        "avg_xG": round(xG, 2), # EPL no SOS adjustment for now
        "xGOT": round(xGOT, 2),
        "avg_xGOT": round(xGOT, 2),
        "Touches_In_Box": round(touches, 2),
        "avg_Touches_In_Box": round(touches, 2),
        "avg_xG_Conceded": round(xG_Conceded, 2),
        "avg_Fouls": round(fouls, 2),
        "avg_Yellow_Cards": round(yellows, 2),
        "avg_Clearances": round(clearances, 2),
        "avg_Shooting_Efficiency": round(shoot_eff, 2),
        "avg_Goalkeeping_Overperformance": round(gk_perf, 2),
        "avg_Goals_Prevented": round(gk_perf, 2), # Simplified
        "latest_Rank": 100 # Mock rank for EPL, since teams are equal-ish league-wise
    }
    
    return metrics
