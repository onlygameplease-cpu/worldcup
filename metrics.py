import numpy as np
import pandas as pd

def check_physical_dominance(stats_fav, stats_dog):
    tackles_fav = float(stats_fav.get("avg_Tackles") or 0)
    tackles_dog = float(stats_dog.get("avg_Tackles") or 0)
    duels_fav = float(stats_fav.get("avg_Duels_Won") or 0)
    duels_dog = float(stats_dog.get("avg_Duels_Won") or 0)
    clear_fav = float(stats_fav.get("avg_Clearances") or 0)
    clear_dog = float(stats_dog.get("avg_Clearances") or 0)
    
    dog_physical_wins = 0
    if tackles_dog > tackles_fav and tackles_dog > 0: dog_physical_wins += 1
    if duels_dog > duels_fav and duels_dog > 0: dog_physical_wins += 1
    if clear_dog > clear_fav and clear_dog > 0: dog_physical_wins += 1
    
    return dog_physical_wins >= 3

def extract_adv_metrics(df_sub, team_name):
    if df_sub is None or df_sub.empty:
        return {}
    
    metrics = {}

    def get_avg(col_home, col_away, apply_multiplier=False):
        if col_home not in df_sub.columns or col_away not in df_sub.columns:
            return None
            
        vals = []
        for _, row in df_sub.iterrows():
            if row["HomeTeam"] == team_name:
                raw_stat = row[col_home]
                opp_rank_col = "Away_Rank"
                own_rank_col = "Home_Rank"
            else:
                raw_stat = row[col_away]
                opp_rank_col = "Home_Rank"
                own_rank_col = "Away_Rank"
                
            if pd.isna(raw_stat):
                continue
                
            if apply_multiplier and opp_rank_col in row and not pd.isna(row[opp_rank_col]) and own_rank_col in row and not pd.isna(row[own_rank_col]):
                opp_rank = float(row[opp_rank_col])
                own_rank = float(row[own_rank_col])
                delta = own_rank - opp_rank
                
                if abs(delta) <= 5:
                    modifier = 1.0
                elif delta > 5:
                    modifier = 1.0 + (delta - 5) * 0.008
                else:
                    modifier = 1.0 + (delta + 5) * 0.008
                    
                modifier = max(0.5, min(1.5, modifier))
                adjusted_stat = float(raw_stat) * modifier
            else:
                adjusted_stat = float(raw_stat)
                
            vals.append(adjusted_stat)
            
        if not vals:
            return None
        return round(float(np.nanmean(vals)), 2)

    sample_size = len(df_sub)
    metrics["Sample_Size"] = sample_size
    trust = min(sample_size / 4.0, 1.0)
    
    BASELINE_XG = 1.25
    BASELINE_XGOT = 1.25
    BASELINE_TOUCHES_IN_BOX = 16.0
    BASELINE_XG_CONCEDED = 1.25

    # Raw metrics (no SOS adjustment) — for Poisson input
    metrics["xG"] = get_avg("xG_Home", "xG_Away", apply_multiplier=False)
    metrics["xGOT"] = get_avg("xGOT_Home", "xGOT_Away", apply_multiplier=False)
    metrics["Touches_In_Box"] = get_avg("Touches_In_Box_Home", "Touches_In_Box_Away", apply_multiplier=False)

    # SOS-adjusted metrics — for Circuit Breaker & comparative analysis
    metrics["avg_xG"] = get_avg("xG_Home", "xG_Away", apply_multiplier=True)
    metrics["avg_xGOT"] = get_avg("xGOT_Home", "xGOT_Away", apply_multiplier=True)
    
    # Audit Metrics (DR Congo)
    metrics["avg_xG_Conceded"] = get_avg("xG_Away", "xG_Home", apply_multiplier=False)
    metrics["avg_Tackles"] = get_avg("Tackles_Home", "Tackles_Away", apply_multiplier=False)
    metrics["avg_Duels_Won"] = get_avg("Duels_Won_Home", "Duels_Won_Away", apply_multiplier=False)
    metrics["avg_Fouls"] = get_avg("Fouls_Home", "Fouls_Away", apply_multiplier=False)
    metrics["avg_Yellow_Cards"] = get_avg("Yellow_Cards_Home", "Yellow_Cards_Away", apply_multiplier=False)
    metrics["avg_Clearances"] = get_avg("Clearances_Home", "Clearances_Away", apply_multiplier=False)
    metrics["avg_Shot_Quality"] = get_avg("Shot_Quality_Home", "Shot_Quality_Away")
    metrics["avg_Shooting_Efficiency"] = get_avg("Shooting_Efficiency_Home", "Shooting_Efficiency_Away")
    metrics["avg_Goalkeeping_Overperformance"] = get_avg("Goalkeeping_Overperformance_Home", "Goalkeeping_Overperformance_Away")
    metrics["avg_Touches_In_Box"] = get_avg("Touches_In_Box_Home", "Touches_In_Box_Away", apply_multiplier=True)
    metrics["avg_Goals_Prevented"] = get_avg("Goals_Prevented_Home", "Goals_Prevented_Away")
    
    # --- SAMPLE-SIZE MEAN REVERSION ---
    if metrics["xG"] is not None:
        metrics["xG"] = round(metrics["xG"] * trust + BASELINE_XG * (1 - trust), 2)
    if metrics["xGOT"] is not None:
        metrics["xGOT"] = round(metrics["xGOT"] * trust + BASELINE_XGOT * (1 - trust), 2)
    if metrics["Touches_In_Box"] is not None:
        metrics["Touches_In_Box"] = round(metrics["Touches_In_Box"] * trust + BASELINE_TOUCHES_IN_BOX * (1 - trust), 2)
        
    if metrics["avg_xG"] is not None:
        metrics["avg_xG"] = round(metrics["avg_xG"] * trust + BASELINE_XG * (1 - trust), 2)
    if metrics["avg_xGOT"] is not None:
        metrics["avg_xGOT"] = round(metrics["avg_xGOT"] * trust + BASELINE_XGOT * (1 - trust), 2)
    if metrics["avg_Touches_In_Box"] is not None:
        metrics["avg_Touches_In_Box"] = round(metrics["avg_Touches_In_Box"] * trust + BASELINE_TOUCHES_IN_BOX * (1 - trust), 2)
        
    if metrics["avg_xG_Conceded"] is not None:
        metrics["avg_xG_Conceded"] = round(metrics["avg_xG_Conceded"] * trust + BASELINE_XG_CONCEDED * (1 - trust), 2)
    # ----------------------------------

    def get_latest(col_home, col_away):
        if col_home not in df_sub.columns or col_away not in df_sub.columns:
            return None
        vals = []
        for _, row in df_sub.iterrows():
            if row["HomeTeam"] == team_name:
                raw_stat = row[col_home]
            else:
                raw_stat = row[col_away]
            if pd.notna(raw_stat):
                vals.append(float(raw_stat))
        return vals[-1] if vals else None

    metrics["latest_Rank"] = get_latest("Home_Rank", "Away_Rank")

    def get_history(col_home, col_away, limit=5):
        if col_home not in df_sub.columns or col_away not in df_sub.columns:
            return []
        vals = []
        for _, row in df_sub.iterrows():
            if row["HomeTeam"] == team_name:
                raw_stat = row[col_home]
            else:
                raw_stat = row[col_away]
            if pd.notna(raw_stat):
                vals.append(float(raw_stat))
        return vals[-limit:]

    # Missing metrics that exist in CSV but never extracted
    metrics["avg_Possession"] = get_avg("Possession_Home", "Possession_Away")
    metrics["avg_Passes_Final_Third"] = get_avg("Passes_Final_Third_Home", "Passes_Final_Third_Away")
    metrics["history_goals_prevented"] = get_history("Goals_Prevented_Home", "Goals_Prevented_Away", limit=5)

    return metrics
