import difflib

TEAM_ALIASES = {
    "cộng hòa séc": "Czech Republic",
    "czechia": "Czech Republic",
    "thụy sĩ": "Switzerland",
    "ma rốc": "Morocco",
    "thổ n. k.": "Turkey",
    "thổ nhĩ kỳ": "Turkey",
    "türkiye": "Turkey",
    "turkiye": "Turkey",
    "bờ biển ngà": "Ivory Coast",
    "thụy điển": "Sweden",
    "ai cập": "Egypt",
    "ả rập saudi": "Saudi Arabia",
    "na uy": "Norway",
    "áo": "Austria",
    "d.r. congo": "Congo DR",
    "dr congo": "Congo DR",
    "nam phi": "South Africa",
    "úc": "Australia",
    "mỹ": "United States",
    "usa": "United States",
    "hàn quốc": "South Korea",
    "korea republic": "South Korea",
    "nhật bản": "Japan",
    "bỉ": "Belgium",
    "bồ đào nha": "Portugal",
    "tây ban nha": "Spain",
    "pháp": "France",
    "đức": "Germany",
    "hà lan": "Netherlands",
    "anh": "England",
    "north korea": "Korea DPR",
    "uae": "United Arab Emirates",
    "bosnia & herzegovina": "Bosnia-Herzegovina",
    "bosnia and herzegovina": "Bosnia-Herzegovina",
    "curaçao": "Curacao",
    "ir iran": "Iran",
    "cabo verde": "Cape Verde"
}

def normalize_team_name(name: str) -> str:
    """Normalize team name by stripping common suffixes, lowercasing, and looking up aliases."""
    if not name:
        return ""
    norm = str(name).strip().lower()
    
    # Do not strip 'united', 'city', 'real', 'sporting', 'athletic' to preserve full names.
    tokens = norm.split()
    tokens = [
        t for t in tokens
        if t not in {
            "fc", "cf", "afc", "club", "de", "cd", "ud", "rc", "sd", "sc", "as", "ss",
            "fs", "fk", "nk", "ks", "bv", "sv"
        }
    ]
    normalized = " ".join(tokens)
    if not normalized:
        normalized = norm
        
    # Check if the exact original lower case has an alias first
    if norm in TEAM_ALIASES:
        return TEAM_ALIASES[norm]
        
    return TEAM_ALIASES.get(normalized, normalized.title() if normalized == norm else " ".join(t.capitalize() for t in tokens))

def get_team_rank(team_name: str, fifa_df) -> float:
    """Returns the FIFA rank for a team. Returns None if not found."""
    if not team_name or fifa_df is None or fifa_df.empty:
        return None
        
    team_name_norm = normalize_team_name(team_name)
    teams_list = fifa_df['Team'].tolist()
    
    # Exact match first
    exact_match = fifa_df[fifa_df['Team'].str.lower() == team_name_norm.lower()]
    if not exact_match.empty:
        return float(exact_match['Rank'].values[0])
        
    # Fuzzy match fallback
    matches = difflib.get_close_matches(team_name_norm, teams_list, n=1, cutoff=0.6)
    if matches:
        return float(fifa_df[fifa_df['Team'] == matches[0]]['Rank'].values[0])
        
    return None
