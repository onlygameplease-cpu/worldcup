import pandas as pd
import numpy as np
import os
from teams import normalize_team_name, get_team_rank

def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tự động tính toán các biến phái sinh chiến thuật cho cả đội Home và Away.
    """
    res_df = df.copy()
    
    # Sanitize team names
    if 'HomeTeam' in res_df.columns:
        res_df['HomeTeam'] = res_df['HomeTeam'].apply(lambda x: normalize_team_name(x))
    if 'AwayTeam' in res_df.columns:
        res_df['AwayTeam'] = res_df['AwayTeam'].apply(lambda x: normalize_team_name(x))
    
    # Ép kiểu an toàn để tránh lỗi nếu dữ liệu không phải số
    for col in ['Total_Shots_Home', 'Total_Shots_Away', 'xG_Home', 'xG_Away', 'xGOT_Home', 'xGOT_Away', 'HomeScore', 'AwayScore']:
        if col in res_df.columns:
            res_df[col] = pd.to_numeric(res_df[col], errors='coerce').fillna(0)
        else:
            res_df[col] = 0.0

    # 1. Shot Quality
    res_df['Shot_Quality_Home'] = np.where(res_df['Total_Shots_Home'] > 0, res_df['xG_Home'] / res_df['Total_Shots_Home'], 0.0)
    res_df['Shot_Quality_Away'] = np.where(res_df['Total_Shots_Away'] > 0, res_df['xG_Away'] / res_df['Total_Shots_Away'], 0.0)
    
    # 2. Shooting Efficiency
    res_df['Shooting_Efficiency_Home'] = res_df['xGOT_Home'] - res_df['xG_Home']
    res_df['Shooting_Efficiency_Away'] = res_df['xGOT_Away'] - res_df['xG_Away']
    
    # 3. Goalkeeping Overperformance (Home thủ môn đối mặt với xGOT_Away)
    res_df['Goalkeeping_Overperformance_Home'] = res_df['xGOT_Away'] - res_df['AwayScore']
    res_df['Goalkeeping_Overperformance_Away'] = res_df['xGOT_Home'] - res_df['HomeScore']
    
    # 4. Tích hợp Bảng Xếp Hạng FIFA (Elo proxy)
    if 'Home_Rank' not in res_df.columns:
        res_df['Home_Rank'] = np.nan
    if 'Away_Rank' not in res_df.columns:
        res_df['Away_Rank'] = np.nan
        
    res_df['Rank_Diff'] = np.nan
    
    if os.path.exists("dataworldcup/fifa_ranking.csv"):
        fifa_df = pd.read_csv("dataworldcup/fifa_ranking.csv")
        
        for idx, row in res_df.iterrows():
            home = str(row.get('HomeTeam', ''))
            away = str(row.get('AwayTeam', ''))
            
            # Chỉ fallback vào CSV nếu OCR không quét được (NaN hoặc bằng 0)
            if pd.isna(row.get('Home_Rank')) or row.get('Home_Rank') == 0:
                rank = get_team_rank(home, fifa_df)
                if rank is not None:
                    res_df.at[idx, 'Home_Rank'] = rank
                else:
                    res_df.at[idx, 'Home_Rank'] = np.nan
                    
            if pd.isna(row.get('Away_Rank')) or row.get('Away_Rank') == 0:
                rank = get_team_rank(away, fifa_df)
                if rank is not None:
                    res_df.at[idx, 'Away_Rank'] = rank
                else:
                    res_df.at[idx, 'Away_Rank'] = np.nan
                    
            # Rank_Diff dương nghĩa là Home có hạng cao hơn (số nhỏ hơn) -> Home mạnh hơn
            hr = res_df.at[idx, 'Home_Rank']
            ar = res_df.at[idx, 'Away_Rank']
            if pd.notna(hr) and pd.notna(ar):
                res_df.at[idx, 'Rank_Diff'] = ar - hr

    fill_cols = [
        'Shot_Quality_Home', 'Shot_Quality_Away', 
        'Shooting_Efficiency_Home', 'Shooting_Efficiency_Away',
        'Goalkeeping_Overperformance_Home', 'Goalkeeping_Overperformance_Away'
    ]
    res_df[fill_cols] = res_df[fill_cols].fillna(0.0)
    
    return res_df
