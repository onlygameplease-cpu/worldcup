import pandas as pd
from metrics import check_physical_dominance

def check_high_blowout_risk(stats_a, stats_b):
    ra = stats_a.get("latest_Rank")
    rb = stats_b.get("latest_Rank")
    
    rank_a = float(ra) if ra is not None and not pd.isna(ra) else None
    rank_b = float(rb) if rb is not None and not pd.isna(rb) else None
    
    xg_a = float(stats_a.get("avg_xG") or stats_a.get("xG") or 1.0)
    xg_b = float(stats_b.get("avg_xG") or stats_b.get("xG") or 1.0)
    
    if rank_a is not None and rank_b is not None:
        delta_rank = abs(rank_a - rank_b)
        if rank_a < rank_b:
            fav, dog = stats_a, stats_b
            xg_fav, xg_dog = xg_a, xg_b
        else:
            fav, dog = stats_b, stats_a
            xg_fav, xg_dog = xg_b, xg_a
    else:
        delta_rank = 0
        # If rank is missing, determine favorite by xG
        if xg_a > xg_b:
            fav, dog = stats_a, stats_b
            xg_fav, xg_dog = xg_a, xg_b
        else:
            fav, dog = stats_b, stats_a
            xg_fav, xg_dog = xg_b, xg_a
            
    touches_fav = float(fav.get("avg_Touches_In_Box") or fav.get("Touches_In_Box") or 0)
    touches_dog = float(dog.get("avg_Touches_In_Box") or dog.get("Touches_In_Box") or 1)
    
    xgot_fav = float(fav.get("avg_xGOT") or fav.get("xGOT") or 0)
    xgot_dog = float(dog.get("avg_xGOT") or dog.get("xGOT") or 0.1)
    danger_dominance = (xgot_fav / max(xgot_dog, 0.1)) >= 3.0
    
    pass_fav = float(fav.get("avg_Passes_Final_Third") or 0)
    passing_dominance = pass_fav >= 120 or (float(fav.get("avg_Possession") or 50) >= 60)
    
    macro_mismatch = (delta_rank >= 30) or (xg_dog > 0 and (xg_fav / xg_dog) >= 3.0)
    
    # --- AUDIT OVERRIDE: SUPPRESS BLOWOUT IF UNDERDOG IS IRON WALL ---
    dog_xg_conceded = float(dog.get("avg_xG_Conceded") or 1.5)
    dog_sample = float(dog.get("Sample_Size") or 1)
    fav_sample = float(fav.get("Sample_Size") or 1)
    
    iron_wall = (dog_xg_conceded < 0.80) and (dog_sample >= 3)
    
    if iron_wall and (xg_fav / max(xg_dog, 0.1) < 2.5):
        macro_mismatch = False
    elif check_physical_dominance(fav, dog) and min(dog_sample, fav_sample) >= 3:
        macro_mismatch = False
    # ------------------------------------------------------------------
    
    box_ratio = touches_fav / max(touches_dog, 1)
    penetration_dominance = (touches_fav >= 25) and (box_ratio >= 3.0)
    
    live_vectors = sum([passing_dominance, penetration_dominance, danger_dominance])
    return macro_mismatch and (live_vectors >= 1)

