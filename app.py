import os
import json
from pathlib import Path
import pandas as pd
import streamlit as st

# ---------- Gemini API (hardcoded, hidden from UI) ----------
# Paste your Gemini API key here (or set env var GEMINI_API_KEY). Not shown in the Streamlit UI.
# Model IDs used here:
#   - gemini-3-flash-preview
#   - gemini-3-pro-preview
# If you write gemini-3-flash or gemini-3-pro, the code will auto-fix to *-preview.


def normalize_gemini_model(model: str) -> str:
    """Normalize Gemini model IDs.

    - Accepts both "models/<id>" and "<id>"
    - Auto-fixes Gemini 3 shorthand to preview IDs (required).
    """
    model = (model or "").strip()
    if model.startswith("models/"):
        model = model.split("/", 1)[1]

    if model.startswith("gemini-3-") and "preview" not in model:
        if model.endswith("-flash"):
            return "gemini-3-flash-preview"
        if model.endswith("-pro"):
            return "gemini-3-pro-preview"
        if model.endswith("-pro-image"):
            return "gemini-3-pro-image-preview"
        return model + "-preview"

    return model
GEMINI_API_VERSION = "v1beta"  # "v1beta" (recommended) or "v1"
GEMINI_MODEL = "gemini-3-flash-preview"
GEMINI_MODEL = normalize_gemini_model(GEMINI_MODEL)

# Prefer env var so you don't hardcode secrets in code:
# ---------------------------------------------------------
# GEMINI API CONFIGURATION
# ---------------------------------------------------------
# CẢNH BÁO: KHÔNG dán cứng API Key vào đây để tránh bị lộ trên GitHub.
# Vui lòng dùng biến môi trường (Environment Variable) hoặc Streamlit Secrets.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2", "").strip()

AI_API_URL = f"https://generativelanguage.googleapis.com/{GEMINI_API_VERSION}/models/{GEMINI_MODEL}:generateContent"
AI_API_KEY = GEMINI_API_KEY

# ---------- OpenAI API (hidden from UI) ----------
# Set env var OPENAI_API_KEY before running (recommended) so you don't hardcode secrets.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODELS = ["gpt-5.4"]
OPENAI_DEFAULT_MODEL = "gpt-5.4"
GOOGLE_RECORD_SHEET_URL = "https://docs.google.com/spreadsheets/d/1J4Oplqlon91j-FEtcZiaNkeaIUwQcMT4Y4F5q0umSnE/edit?usp=sharing"

# ---------- Config ----------
DATA_FILES = {
    "EPL": "data/E0.csv",
    "LaLiga": "data/SP1.csv",
    "Serie A": "data/I1.csv",
    "Bundesliga": "data/D1.csv",
    "Ligue 1": "data/F1.csv",
    "Greece": "data/G1.csv",
    "Netherlands": "data/N1.csv",
    "Portugal": "data/P1.csv",
    "Scotland": "data/SC0.csv",
    "Turkey": "data/T1.csv",
    "Belgium": "data/B1.csv",
    "Championship": "data/E1.csv",
    "France Ligue 2": "data/F2.csv",
    "Italy Serie B": "data/I2.csv",
    "Germany 2.Bundesliga": "data/D2.csv",
    "Spain Segunda": "data/SP2.csv",
    # "Europa League": "data/EL.csv",  # má»Ÿ náº¿u cÃ³ file nÃ y
    # "Conference League": "data/ECL.csv",  # má»Ÿ náº¿u cÃ³ file nÃ y
    # "UCL": "data/CL.csv",  # mở nếu có file này
    "ALL (gộp 5 giải)": None,
}
ALL5_LEAGUES = ["EPL", "LaLiga", "Serie A", "Bundesliga", "Ligue 1", "Greece", "Netherlands", "Portugal", "Scotland", "Turkey", "Belgium"]
ODDS_API_IO_LEAGUE_SLUGS = {
    "EPL": "england-premier-league",
    "LaLiga": "spain-la-liga",
    "Serie A": "italy-serie-a",
    "Bundesliga": "germany-bundesliga",
    "Ligue 1": "france-ligue-1",
    "Greece": "greece-super-league",
    "Netherlands": "netherlands-eredivisie",
    "Portugal": "portugal-primeira-liga",
    "Scotland": "scotland-premiership",
    "Turkey": "turkey-super-lig",
    "Belgium": "belgium-first-division-a",
    "Championship": "england-championship",
    "France Ligue 2": "france-ligue-2",
    "Italy Serie B": "italy-serie-b",
    "Germany 2.Bundesliga": "germany-2-bundesliga",
    "Spain Segunda": "spain-segunda-division",
    "Champions League": "uefa-champions-league",
    "Europa League": "uefa-europa-league",
    "Conference League": "uefa-conference-league",
}

GOAL_LINES = [2, 2.5, 3, 3.5, 4]
H1_LINES   = [0.75, 1.0, 1.25, 1.5]
COR_LINES  = [9, 9.5, 10, 10.5]
CARD_LINES = [3.5, 4.5, 5.5]

st.set_page_config(page_title="EPL DB (Personal)", layout="wide")
st.title("EPL Database (Personal) — Goals / 1H / Corners / Cards / AH")

# ---------- Helpers ----------
def _num(s):
    return pd.to_numeric(s, errors="coerce")

def hit_rate(df: pd.DataFrame, col: str, lines: list[float]) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    x = _num(df[col]).dropna()
    out = []
    for ln in lines:
        out.append({
            "Line": ln,
            "Over%": float((x > ln).mean() * 100) if len(x) else 0.0,
            "Under%": float((x <= ln).mean() * 100) if len(x) else 0.0,
            "Samples": int(len(x))
        })
    return pd.DataFrame(out)



def _btts_rate(df: pd.DataFrame) -> float:
    """Return BTTS rate (0..1), NaN-safe."""
    if df is None or df.empty or "BTTS" not in df.columns:
        return float("nan")
    s = df["BTTS"].dropna()
    if len(s) == 0:
        return float("nan")
    # Ensure boolean (True=1, False=0). Avoid bool(np.nan)==True traps via dropna above.
    try:
        s = s.astype(bool)
    except Exception:
        pass
    return float(s.mean())



def build_btts_summary(a_df: pd.DataFrame, b_df: pd.DataFrame, h2h_df: pd.DataFrame) -> dict:
    """BTTS summary + simple auto-pick.

    Returns a dict that is easy to display and also safe to send to AI payload.
    """
    ra = _btts_rate(a_df)
    rb = _btts_rate(b_df)
    rh = _btts_rate(h2h_df)

    vals = [x for x in [ra, rb, rh] if x == x]  # filter NaN
    blend = float(sum(vals) / len(vals)) if vals else float("nan")

    # Simple rule:
    #  >= 0.60  -> BTTS Yes
    #  <= 0.40  -> BTTS No
    #  else     -> Lean
    pick = "Lean"
    if blend == blend:
        if blend >= 0.60:
            pick = "BTTS Yes"
        elif blend <= 0.40:
            pick = "BTTS No"

    return {
        "A_BTTS%": round(ra * 100, 1) if ra == ra else None,
        "B_BTTS%": round(rb * 100, 1) if rb == rb else None,
        "H2H_BTTS%": round(rh * 100, 1) if rh == rh else None,
        "Blended%": round(blend * 100, 1) if blend == blend else None,
        "Pick": pick,
        "Samples": {
            "A": int(len(a_df)) if a_df is not None else 0,
            "B": int(len(b_df)) if b_df is not None else 0,
            "H2H": int(len(h2h_df)) if h2h_df is not None else 0,
        },
    }

def last_n(df: pd.DataFrame, team: str, n: int) -> pd.DataFrame:
    sub = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    sub = sub.sort_values("Date", ascending=False)
    return sub.head(n)

def last_n_home(df: pd.DataFrame, team: str, n: int) -> pd.DataFrame:
    """Last N matches where team played at HOME."""
    sub = df[df["HomeTeam"] == team].copy()
    sub = sub.sort_values("Date", ascending=False)
    return sub.head(n)

def last_n_away(df: pd.DataFrame, team: str, n: int) -> pd.DataFrame:
    """Last N matches where team played AWAY."""
    sub = df[df["AwayTeam"] == team].copy()
    sub = sub.sort_values("Date", ascending=False)
    return sub.head(n)

def h2h_strict(df: pd.DataFrame, home: str, away: str, n: int) -> pd.DataFrame:
    """Last N head-to-head matches with fixed venue: home vs away."""
    sub = df[(df["HomeTeam"] == home) & (df["AwayTeam"] == away)].copy()
    sub = sub.sort_values("Date", ascending=False)
    return sub.head(n)

def h2h(df: pd.DataFrame, a: str, b: str, n: int) -> pd.DataFrame:
    sub = df[
        ((df["HomeTeam"] == a) & (df["AwayTeam"] == b)) |
        ((df["HomeTeam"] == b) & (df["AwayTeam"] == a))
    ].copy()
    sub = sub.sort_values("Date", ascending=False)
    return sub.head(n)


def _normalize_team_name(name: str) -> str:
    name = (name or "").strip().lower()
    if not name:
        return ""

    alias_map = {
        "man utd": "manchester united",
        "man united": "manchester united",
        "man city": "manchester city",
        "m gladbach": "monchengladbach",
        "mgladbach": "monchengladbach",
        "borussia m gladbach": "monchengladbach",
        "borussia monchengladbach": "monchengladbach",
        "monchengladbach": "monchengladbach",
        "spurs": "tottenham hotspur",
        "tottenham": "tottenham hotspur",
        "nottm forest": "nottingham forest",
        "nott'm forest": "nottingham forest",
        "nottm forest": "nottingham forest",
        "nottingham forest": "nottingham forest",
        "wolves": "wolverhampton wanderers",
        "newcastle": "newcastle united",
        "forest": "nottingham forest",
        "west brom": "west bromwich albion",
        "sheff utd": "sheffield united",
        "sheff wed": "sheffield wednesday",
        "sheffield weds": "sheffield wednesday",
        "qpr": "queens park rangers",
        "brighton": "brighton and hove albion",
        "nurnberg": "nuremberg",
        "nuremberg": "nuremberg",
        "nurnberg fc": "nuremberg",
        "1 fc nurnberg": "nuremberg",
        "fc nurnberg": "nuremberg",
        "waregem": "zulte waregem",
        "zulte waregem": "zulte waregem",
        "sv zulte waregem": "zulte waregem",
        "dresden": "dynamo dresden",
        "dynamo dresden": "dynamo dresden",
        "leeds": "leeds united",
        "leicester": "leicester city",
        "norwich": "norwich city",
        "stoke": "stoke city",
        "swansea": "swansea city",
        "cardiff": "cardiff city",
        "birmingham": "birmingham city",
        "bristol city": "bristol city",
        "coventry": "coventry city",
        "ipswich": "ipswich town",
        "luton": "luton town",
        "preston": "preston north end",
        "derby": "derby county",
        "real sociedad san sebastian": "real sociedad",
        "ath madrid": "atletico madrid",
        "atletico": "atletico madrid",
        "ath bilbao": "athletic club",
        "athletic bilbao": "athletic club",
        "athletic club": "athletic club",
        "alaves": "deportivo alaves",
        "deportivo alaves": "deportivo alaves",
        "ein frankfurt": "eintracht frankfurt",
        "eintracht frankfurt": "eintracht frankfurt",
        "hamburg": "hamburger sv",
        "hamburger sv": "hamburger sv",
        "union berlin": "union berlin",
        "fc koln": "koln",
        "koln": "koln",
        "1 fc koln": "koln",
        "fc cologne": "koln",
        "1 fc cologne": "koln",
        "cologne": "koln",
        "leverkusen": "bayer leverkusen",
        "bayer leverkusen": "bayer leverkusen",
        "mainz": "mainz 05",
        "mainz 05": "mainz 05",
        "fsv mainz 05": "mainz 05",
        "1 fsv mainz 05": "mainz 05",
        "mainz 05": "mainz 05",
        "st pauli": "st pauli",
        "fc st pauli": "st pauli",
        "paris sg": "paris saint germain",
        "lorient": "fc lorient",
        "fc lorient": "fc lorient",
        "espanol": "espanyol barcelona",
        "espanyol": "espanyol barcelona",
        "espanyol barcelona": "espanyol barcelona",
        "sp lisbon": "sporting cp",
        "sporting lisbon": "sporting cp",
        "sporting cp": "sporting cp",
        "guimaraes": "vitoria sc guimaraes",
        "vitoria guimaraes": "vitoria sc guimaraes",
        "vitoria sc guimaraes": "vitoria sc guimaraes",
        "hearts": "heart of midlothian",
        "heart of midlothian": "heart of midlothian",
        "rangers": "glasgow rangers",
        "glasgow rangers": "glasgow rangers",
        "inter": "inter milan",
        "internazionale": "inter milan",
        "milan": "ac milan",
        "psg": "paris saint germain",
    }

    repl = {
        "&": " and ",
        ".": " ",
        "-": " ",
        "_": " ",
        "'": "",
    }
    for a, b in repl.items():
        name = name.replace(a, b)

    tokens = [
        tok
        for tok in name.split()
        if tok
        and not tok.isdigit()
        and tok not in {
            "fc",
            "cf",
            "afc",
            "sc",
            "ac",
            "club",
            "de",
            "ssc",
            "ss",
            "as",
            "us",
            "fk",
            "nk",
            "ks",
            "bv",
            "sv",
        }
    ]
    normalized = " ".join(tokens)
    return alias_map.get(normalized, normalized)


def _team_names_match(left: str, right: str) -> bool:
    left_norm = _normalize_team_name(left)
    right_norm = _normalize_team_name(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if left_norm in right_norm or right_norm in left_norm:
        return True

    left_tokens = set(left_norm.split())
    right_tokens = set(right_norm.split())
    if not left_tokens or not right_tokens:
        return False

    overlap = left_tokens & right_tokens
    min_tokens = min(len(left_tokens), len(right_tokens))
    return len(overlap) >= max(1, min_tokens)


def _event_teams_from_odds_item(item: dict) -> tuple[str, str]:
    return (
        _normalize_team_name(item.get("home_team", "")),
        _normalize_team_name(item.get("away_team", "")),
    )


def find_matching_odds_event(odds_snapshot: dict | None, home_team: str, away_team: str) -> dict | None:
    if not isinstance(odds_snapshot, dict):
        return None

    events = odds_snapshot.get("odds") or []
    if not isinstance(events, list) or not events:
        return None

    home_norm = _normalize_team_name(home_team)
    away_norm = _normalize_team_name(away_team)
    if not home_norm or not away_norm:
        return None

    def _score_event(item: dict) -> int:
        ev_home, ev_away = _event_teams_from_odds_item(item)
        score = 0

        if ev_home == home_norm:
            score += 5
        elif home_norm and ev_home and (home_norm in ev_home or ev_home in home_norm):
            score += 3

        if ev_away == away_norm:
            score += 5
        elif away_norm and ev_away and (away_norm in ev_away or ev_away in away_norm):
            score += 3

        if ev_home == away_norm:
            score -= 2
        if ev_away == home_norm:
            score -= 2

        return score

    scored = [(item, _score_event(item)) for item in events]
    scored = [x for x in scored if x[1] > 0]
    if not scored:
        return None

    scored.sort(key=lambda x: x[1], reverse=True)
    best_item, best_score = scored[0]
    if best_score < 6:
        return None

    return best_item


def find_matching_odds_api_io_event(events: list | None, home_team: str, away_team: str) -> dict | None:
    if not isinstance(events, list) or not events:
        return None

    best_item = None
    best_score = -1
    for item in events:
        home_name = item.get("home", "")
        away_name = item.get("away", "")
        score = 0

        if _team_names_match(home_name, home_team):
            score += 5
        elif _normalize_team_name(home_name) in _normalize_team_name(home_team) or _normalize_team_name(home_team) in _normalize_team_name(home_name):
            score += 3

        if _team_names_match(away_name, away_team):
            score += 5
        elif _normalize_team_name(away_name) in _normalize_team_name(away_team) or _normalize_team_name(away_team) in _normalize_team_name(away_name):
            score += 3

        if _team_names_match(home_name, away_team):
            score -= 2
        if _team_names_match(away_name, home_team):
            score -= 2

        if score > best_score:
            best_item = item
            best_score = score

    if best_score < 6:
        return None
    return best_item


def _normalize_league_text(text: str) -> str:
    text = (text or "").strip().lower()
    for ch in ["-", ".", "_", "/", "'", "&", "(", ")"]:
        text = text.replace(ch, " ")
    return " ".join(text.split())


def filter_odds_api_io_events_by_league(events: list | None, league_slug: str, league_label: str) -> list:
    if not isinstance(events, list) or not events:
        return []
    if not league_slug and not league_label:
        return events

    out = []
    label_aliases = {
        "laliga": ["la liga", "laliga"],
        "ligue 1": ["ligue 1"],
        "serie a": ["serie a"],
        "bundesliga": ["bundesliga"],
        "germany 2 bundesliga": ["2 bundesliga", "2. bundesliga"],
        "france ligue 2": ["ligue 2"],
        "italy serie b": ["serie b"],
        "spain segunda": ["segunda division", "segunda"],
        "champions league": ["champions league", "uefa champions league"],
        "europa league": ["europa league", "uefa europa league"],
        "conference league": ["conference league", "uefa conference league"],
    }
    label_norm = _normalize_league_text(league_label)
    candidate_names = {label_norm} if label_norm else set()
    for alias in label_aliases.get(label_norm, []):
        candidate_names.add(_normalize_league_text(alias))

    for item in events:
        league_obj = item.get("league") or {}
        slug = str(league_obj.get("slug") or "").strip().lower()
        name = _normalize_league_text(str(league_obj.get("name") or ""))
        if league_slug and slug == str(league_slug).strip().lower():
            out.append(item)
            continue
        if candidate_names and any(candidate and candidate in name for candidate in candidate_names):
            out.append(item)
            continue
    return out


def build_ai_odds_payload(odds_snapshot: dict | None, home_team: str, away_team: str) -> dict | None:
    if not isinstance(odds_snapshot, dict):
        return None

    if odds_snapshot.get("provider") == "odds-api-io":
        event = odds_snapshot.get("event")
        event_home = ((event or {}).get("home_team") or (event or {}).get("home") or "")
        event_away = ((event or {}).get("away_team") or (event or {}).get("away") or "")
        is_match = bool(event) and _team_names_match(event_home, home_team) and _team_names_match(event_away, away_team)
        payload = {
            "provider": "odds-api-io",
            "sport_key": odds_snapshot.get("sport_key"),
            "league": odds_snapshot.get("league"),
            "bookmakers": odds_snapshot.get("bookmakers"),
            "requested_match": {"home": home_team, "away": away_team},
            "matched": is_match,
        }
        if is_match:
            payload["event"] = event
        else:
            payload["match_error"] = odds_snapshot.get("match_error") or "No matched odds-api.io event was stored for the selected teams."
        return payload

    matched = find_matching_odds_event(odds_snapshot, home_team, away_team)
    payload = {
        "sport_key": odds_snapshot.get("sport_key"),
        "regions": odds_snapshot.get("regions"),
        "markets": odds_snapshot.get("markets"),
        "days_ahead": odds_snapshot.get("days_ahead"),
        "requested_match": {"home": home_team, "away": away_team},
        "matched": bool(matched),
    }
    if matched:
        payload["event"] = matched
    else:
        payload["match_error"] = "No matching odds event found for the selected teams in the current fetched odds snapshot."
    return payload


def _event_has_usable_markets(event: dict | None) -> bool:
    if not isinstance(event, dict):
        return False
    books = _normalize_bookmakers_payload(event)
    if not books:
        return False
    return any((book.get("markets") or []) for book in books)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _format_num(x: float | None) -> str:
    if x is None:
        return ""
    if abs(x - int(x)) < 1e-9:
        return str(int(x))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def _normalize_bookmakers_payload(event: dict) -> list[dict]:
    raw_books = (event or {}).get("bookmakers") or []
    if isinstance(raw_books, list):
        return raw_books
    if not isinstance(raw_books, dict):
        return []

    normalized_books = []
    for book_name, book_payload in raw_books.items():
        if not isinstance(book_payload, dict):
            continue

        if isinstance(book_payload.get("markets"), list):
            normalized_books.append({"title": book_name, "markets": book_payload.get("markets") or []})
            continue

        markets = []
        for market_key, market_payload in book_payload.items():
            outcomes = None

            if isinstance(market_payload, dict):
                if isinstance(market_payload.get("outcomes"), list):
                    outcomes = market_payload.get("outcomes")
                else:
                    guessed = []
                    point = market_payload.get("line") or market_payload.get("point") or market_payload.get("hdp")
                    point_num = _safe_float(point)
                    if "over" in market_payload or "under" in market_payload:
                        over_obj = market_payload.get("over")
                        under_obj = market_payload.get("under")
                        if isinstance(over_obj, dict):
                            guessed.append({"name": "Over", "price": over_obj.get("price"), "point": point})
                        elif over_obj is not None:
                            guessed.append({"name": "Over", "price": over_obj, "point": point})
                        if isinstance(under_obj, dict):
                            guessed.append({"name": "Under", "price": under_obj.get("price"), "point": point})
                        elif under_obj is not None:
                            guessed.append({"name": "Under", "price": under_obj, "point": point})
                    elif "home" in market_payload or "away" in market_payload:
                        home_name = event.get("home_team") or event.get("home") or "Home"
                        away_name = event.get("away_team") or event.get("away") or "Away"
                        home_obj = market_payload.get("home")
                        away_obj = market_payload.get("away")
                        if isinstance(home_obj, dict):
                            guessed.append({"name": home_name, "price": home_obj.get("price"), "point": point})
                        elif home_obj is not None:
                            guessed.append({"name": home_name, "price": home_obj, "point": point})
                        away_point = -point_num if point_num is not None else point
                        if isinstance(away_obj, dict):
                            guessed.append({"name": away_name, "price": away_obj.get("price"), "point": away_point})
                        elif away_obj is not None:
                            guessed.append({"name": away_name, "price": away_obj, "point": away_point})
                    if guessed:
                        outcomes = guessed
            elif isinstance(market_payload, list):
                outcomes = market_payload

            if outcomes:
                markets.append({"key": market_key, "outcomes": outcomes})

        if markets:
            normalized_books.append({"title": book_name, "markets": markets})

    return normalized_books


def extract_input_lines_from_matched_odds(matched_odds: dict | None) -> dict:
    out = {
        "goals": "",
        "handicap": "",
        "corners": "",
    }
    if not isinstance(matched_odds, dict) or not matched_odds.get("matched"):
        return out

    event = matched_odds.get("event") or {}
    bookmakers = _normalize_bookmakers_payload(event)
    if not bookmakers:
        return out

    totals_best = None
    spreads_best = None
    h2h_best = None

    for book in bookmakers:
        for market in (book.get("markets") or []):
            key = (market.get("key") or "").strip().lower()
            outcomes = market.get("outcomes") or []
            if key in {"totals", "total", "over_under", "overunder", "ou"} and not totals_best and len(outcomes) >= 2:
                totals_best = outcomes
            elif key in {"spreads", "spread", "asian_handicap", "handicap", "ah"} and not spreads_best and len(outcomes) >= 2:
                spreads_best = outcomes
            elif key == "h2h" and not h2h_best and len(outcomes) >= 2:
                h2h_best = outcomes

    if totals_best:
        over = next((x for x in totals_best if str(x.get("name", "")).lower() == "over"), None)
        under = next((x for x in totals_best if str(x.get("name", "")).lower() == "under"), None)
        point = _safe_float((over or {}).get("point"))
        if point is None:
            point = _safe_float((under or {}).get("point"))
        over_price = _safe_float((over or {}).get("price"))
        under_price = _safe_float((under or {}).get("price"))
        if point is not None and (over_price is not None or under_price is not None):
            out["goals"] = f"Goals: {_format_num(point)} | Over {_format_num(over_price)} | Under {_format_num(under_price)}"

    if spreads_best:
        home = event.get("home_team") or event.get("home") or ""
        away = event.get("away_team") or event.get("away") or ""
        home_side = next((x for x in spreads_best if str(x.get("name", "")).strip().lower() == str(home).strip().lower()), None)
        away_side = next((x for x in spreads_best if str(x.get("name", "")).strip().lower() == str(away).strip().lower()), None)
        line = _safe_float((home_side or {}).get("point"))
        if line is None and away_side is not None:
            away_line = _safe_float(away_side.get("point"))
            line = -away_line if away_line is not None else None
        home_price = _safe_float((home_side or {}).get("price"))
        away_price = _safe_float((away_side or {}).get("price"))
        if line is not None and (home_price is not None or away_price is not None):
            out["handicap"] = f"Handicap (home): {_format_num(line)} | Home {_format_num(home_price)} | Away {_format_num(away_price)}"

    # Some books expose corners as alternate totals/spreads, but TheOddsAPI default soccer payload often won't.
    # Keep corners blank unless a reliable market is found later.

    return out


def extract_structured_odds_lines(matched_odds: dict | None) -> dict:
    out = {
        "goals": {"line": None, "over_price": None, "under_price": None},
        "handicap": {"line": None, "home_price": None, "away_price": None},
    }
    if not isinstance(matched_odds, dict) or not matched_odds.get("matched"):
        return out

    event = matched_odds.get("event") or {}
    bookmakers = _normalize_bookmakers_payload(event)
    if not bookmakers:
        return out

    totals_best = None
    spreads_best = None

    for book in bookmakers:
        for market in (book.get("markets") or []):
            key = (market.get("key") or "").strip().lower()
            outcomes = market.get("outcomes") or []
            if key in {"totals", "total", "over_under", "overunder", "ou"} and not totals_best and len(outcomes) >= 2:
                totals_best = outcomes
            elif key in {"spreads", "spread", "asian_handicap", "handicap", "ah"} and not spreads_best and len(outcomes) >= 2:
                spreads_best = outcomes

    if totals_best:
        over = next((x for x in totals_best if str(x.get("name", "")).lower() == "over"), None)
        under = next((x for x in totals_best if str(x.get("name", "")).lower() == "under"), None)
        out["goals"] = {
            "line": _safe_float((over or {}).get("point")) if over else _safe_float((under or {}).get("point")),
            "over_price": _safe_float((over or {}).get("price")),
            "under_price": _safe_float((under or {}).get("price")),
        }

    if spreads_best:
        home = event.get("home_team") or event.get("home") or ""
        away = event.get("away_team") or event.get("away") or ""
        home_side = next((x for x in spreads_best if str(x.get("name", "")).strip().lower() == str(home).strip().lower()), None)
        away_side = next((x for x in spreads_best if str(x.get("name", "")).strip().lower() == str(away).strip().lower()), None)
        line = _safe_float((home_side or {}).get("point"))
        if line is None and away_side is not None:
            away_line = _safe_float(away_side.get("point"))
            line = -away_line if away_line is not None else None
        out["handicap"] = {
            "line": line,
            "home_price": _safe_float((home_side or {}).get("price")),
            "away_price": _safe_float((away_side or {}).get("price")),
        }

    return out


def build_prompt_odds_block(matched_odds: dict | None, input_lines: dict | None) -> str:
    if not isinstance(matched_odds, dict):
        return "ODDS SNAPSHOT:\n- Not loaded.\n"

    if not matched_odds.get("matched"):
        return "ODDS SNAPSHOT:\n- Loaded but not matched to this exact fixture.\n"

    event = matched_odds.get("event") or {}
    provider = matched_odds.get("provider") or "unknown"
    bookmakers = matched_odds.get("bookmakers") or ""
    home = (
        event.get("home_team")
        or event.get("home")
        or (matched_odds.get("requested_match") or {}).get("home")
        or "?"
    )
    away = (
        event.get("away_team")
        or event.get("away")
        or (matched_odds.get("requested_match") or {}).get("away")
        or "?"
    )

    lines = ["ODDS SNAPSHOT:"]
    lines.append(f"- Source: {provider} | Bookmakers: {bookmakers}")
    lines.append(f"- Matched fixture: {home} vs {away}")

    goals_line = (input_lines or {}).get("goals")
    handicap_line = (input_lines or {}).get("handicap")
    corners_line = (input_lines or {}).get("corners")
    if goals_line:
        lines.append(f"- {goals_line}")
    if handicap_line:
        lines.append(f"- {handicap_line}")
    if corners_line:
        lines.append(f"- {corners_line}")

    if provider == "odds-api-io" and not any([goals_line, handicap_line, corners_line]):
        lines.append("- Use the raw matched event odds object in payload if structured input lines are unavailable.")

    return "\n".join(lines) + "\n"


def build_raw_market_summary(matched_odds: dict | None) -> str:
    if not isinstance(matched_odds, dict) or not matched_odds.get("matched"):
        return ""

    event = matched_odds.get("event") or {}
    bookmakers = _normalize_bookmakers_payload(event)
    if not bookmakers:
        return ""

    lines = []
    for book in bookmakers[:3]:
        book_title = book.get("title") or "Bookmaker"
        for market in (book.get("markets") or [])[:8]:
            key = (market.get("key") or "").strip()
            outcomes = market.get("outcomes") or []
            pretty = []
            for item in outcomes[:4]:
                name = str(item.get("name") or "").strip()
                point = _format_num(_safe_float(item.get("point")))
                price = _format_num(_safe_float(item.get("price")))
                bit = name
                if point:
                    bit += f" {point}"
                if price:
                    bit += f" @{price}"
                pretty.append(bit.strip())
            if pretty:
                lines.append(f"- {book_title} | {key}: " + " ; ".join(pretty))
    return "\n".join(lines)


def pick_strength_score(row: dict) -> float:
    if not isinstance(row, dict):
        return -999.0
    pick = str(row.get("Pick", "")).upper()
    if pick not in {"OVER", "UNDER"}:
        return -999.0

    score = 0.0
    tag = str(row.get("Tag", "")).upper()
    confidence = str(row.get("Confidence", "")).upper()
    if tag == "STRONG":
        score += 20
    elif tag == "EDGE":
        score += 10

    score += {"HIGH": 12, "MEDIUM": 8, "LOW": 3}.get(confidence, 0)

    over = _safe_float(row.get("Over%"))
    under = _safe_float(row.get("Under%"))
    if pick == "OVER" and over is not None:
        score += max(0.0, over - 50.0)
    elif pick == "UNDER" and under is not None:
        score += max(0.0, under - 50.0)
    return score


def best_pick_from_summary(sum_df: pd.DataFrame) -> dict | None:
    if sum_df is None or sum_df.empty:
        return None
    rows = [r for r in sum_df.to_dict("records") if str(r.get("Pick", "")).upper() in {"OVER", "UNDER"}]
    if not rows:
        return None
    strong_rows = [r for r in rows if str(r.get("Tag", "")).upper() == "STRONG"]
    edge_rows = [r for r in rows if str(r.get("Tag", "")).upper() == "EDGE"]
    candidate_rows = strong_rows or edge_rows or rows

    # Practical preference:
    # 1) If there is any STRONG pick, use STRONG first.
    # 2) Inside the same strength tier, prefer the lower line.
    # 3) Use confidence / strength score only as a tie-breaker.
    candidate_rows.sort(
        key=lambda r: (
            _safe_float(r.get("Line")) if _safe_float(r.get("Line")) is not None else float("inf"),
            -pick_strength_score(r),
        )
    )
    return candidate_rows[0]


def line_mismatch(book_line: float | None, suggested_line: float | None, tolerance: float = 0.5) -> bool:
    if book_line is None or suggested_line is None:
        return False
    try:
        return abs(float(book_line) - float(suggested_line)) > float(tolerance)
    except Exception:
        return False


def verdict_from_pick(row: dict | None, book_line: float | None = None, tolerance: float = 0.5) -> dict:
    if not row:
        return {"verdict": "NO BET", "why": "No strong market signal", "book_line": book_line}

    pick = str(row.get("Pick", "")).upper()
    tag = str(row.get("Tag", "")).upper()
    confidence = str(row.get("Confidence", "")).upper()
    suggested_line = _safe_float(row.get("Line"))
    notes = str(row.get("Notes", "") or "").strip()

    if pick not in {"OVER", "UNDER"}:
        return {"verdict": "NO BET", "why": notes or "Rule did not produce a playable side", "book_line": book_line}

    if line_mismatch(book_line, suggested_line, tolerance=tolerance):
        return {
            "verdict": "LEAN ONLY",
            "why": f"Stats like {pick} {_format_num(suggested_line)} but current book line is {_format_num(book_line)}",
            "book_line": book_line,
        }

    if tag == "STRONG" and confidence in {"MEDIUM", "HIGH"}:
        return {
            "verdict": "PLAYABLE",
            "why": notes or f"{tag} signal with {confidence.lower()} confidence",
            "book_line": book_line,
        }

    if tag in {"STRONG", "EDGE"}:
        return {
            "verdict": "LEAN",
            "why": notes or f"{tag} signal but confidence is only {confidence.lower() or 'unknown'}",
            "book_line": book_line,
        }

    return {"verdict": "NO BET", "why": notes or "No edge", "book_line": book_line}


def render_market_verdict(title: str, row: dict | None, verdict: dict):
    st.markdown(f"### {title}")
    if not row:
        st.write("Pick: **NO BET**")
        st.caption(verdict.get("why", "No usable signal"))
        return

    pick = row.get("Pick", "")
    line = row.get("Line")
    confidence = row.get("Confidence", "")
    tag = row.get("Tag", "")
    book_line = verdict.get("book_line")

    st.write(f"Pick: **{pick} {_format_num(_safe_float(line))}**")
    st.write(f"Confidence: **{confidence or '—'}** | Tag: **{tag or '—'}**")
    if book_line is not None:
        st.write(f"Book line: **{_format_num(_safe_float(book_line))}**")
    st.write(f"Verdict: **{verdict.get('verdict', 'NO BET')}**")
    st.caption(verdict.get("why", ""))


def team_picker(label: str, teams: list[str], key_prefix: str, default_team: str | None = None) -> str:
    teams = [t for t in teams if isinstance(t, str) and t.strip()]
    if not teams:
        return ""

    selection_key = f"{key_prefix}_selected"
    search_key = f"{key_prefix}_search"

    if selection_key not in st.session_state or st.session_state[selection_key] not in teams:
        st.session_state[selection_key] = default_team if default_team in teams else teams[0]
    if search_key not in st.session_state:
        st.session_state[search_key] = ""

    search_value = st.text_input(
        f"{label} search",
        key=search_key,
        placeholder="Type team name, Ctrl+A to clear",
    ).strip()

    filtered = [t for t in teams if search_value.lower() in t.lower()] if search_value else list(teams)
    if not filtered:
        st.caption("No team matched your search. Showing full list again.")
        filtered = list(teams)

    current = st.session_state.get(selection_key)
    index = filtered.index(current) if current in filtered else 0
    chosen = st.selectbox(label, filtered, index=index, key=f"{key_prefix}_selectbox")
    st.session_state[selection_key] = chosen
    return chosen

def ah_points(goals_for: float, goals_against: float, line: float) -> float:
    """
    Trả về điểm cược Asian Handicap cho 'đội mình' theo line:
    +1 = win, +0.5 = half-win, 0 = push, -0.5 = half-lose, -1 = lose
    """
    if pd.isna(line) or pd.isna(goals_for) or pd.isna(goals_against):
        return float("nan")

    q = int(round(line * 4))  # quarter units

    def _single(line_half: float) -> float:
        adj = goals_for + line_half
        if adj > goals_against:
            return 1.0
        if adj == goals_against:
            return 0.0
        return -1.0

    # line .0 hoặc .5 => 1 bet
    if q % 2 == 0:
        return _single(q / 4)

    # line .25 hoặc .75 => split 2 bet
    l1 = (q - 1) / 4
    l2 = (q + 1) / 4
    return (_single(l1) + _single(l2)) / 2  # => {1,0.5,0,-0.5,-1}

def ah_label(p: float) -> str:
    if pd.isna(p): return ""
    return {1.0:"W", 0.5:"HW", 0.0:"P", -0.5:"HL", -1.0:"L"}.get(float(p), "")

def _safe_mean(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.mean()) if len(s) else float("nan")

def _prob_over(s: pd.Series, line: float) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float((s > line).mean()) if len(s) else float("nan")

def build_ou_summary(
    a_last: pd.DataFrame,
    b_last: pd.DataFrame,
    h2h_df: pd.DataFrame,
    col: str,
    lines: list[float],
    w_h2h: float = 0.15,
    w_a: float = 0.42,
    w_b: float = 0.42,
    edge: float = 0.58,
    strong: float = 0.64,
):
    """Blend O/U stats from H2H + Team A last-N + Team B last-N.

    Fixes NaN propagation: if a source has no samples, its weight is ignored and
    remaining weights are re-normalized.
    Returns: (summary_df, avg_value, sample_info)
    """

    def _get_series(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty or col not in df.columns:
            return pd.Series(dtype="float")
        return pd.to_numeric(df[col], errors="coerce").dropna()

    s_h2h = _get_series(h2h_df)
    s_a = _get_series(a_last)
    s_b = _get_series(b_last)

    n_h2h, n_a, n_b = len(s_h2h), len(s_a), len(s_b)

    def _blend(vals: list[tuple[float, float]]) -> float:
        total = 0.0
        wsum = 0.0
        for v, w in vals:
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass
            if w <= 0:
                continue
            total += float(v) * float(w)
            wsum += float(w)
        return total / wsum if wsum > 0 else float("nan")

    # Avg value (for debugging/extra context)
    avg_value = _blend([
        (s_h2h.mean(), w_h2h) if n_h2h else (float("nan"), 0.0),
        (s_a.mean(), w_a) if n_a else (float("nan"), 0.0),
        (s_b.mean(), w_b) if n_b else (float("nan"), 0.0),
    ])

    def _prob_over(series: pd.Series, ln: float) -> float:
        if series is None or len(series) == 0:
            return float("nan")
        return float((series > ln).mean())  # 0..1

    rows = []
    for ln in lines:
        p_over = _blend([
            (_prob_over(s_h2h, ln), w_h2h) if n_h2h else (float("nan"), 0.0),
            (_prob_over(s_a, ln), w_a) if n_a else (float("nan"), 0.0),
            (_prob_over(s_b, ln), w_b) if n_b else (float("nan"), 0.0),
        ])

        total_samples = n_h2h + n_a + n_b
        strongest_source = max(n_h2h, n_a, n_b)

        if pd.isna(p_over):
            over_pct = None
            under_pct = None
            pick = "NO DATA"
            tag = ""
            confidence = "NO DATA"
            notes = "Missing numeric history"
        else:
            p_under = 1.0 - p_over
            over_pct = round(p_over * 100.0, 1)
            under_pct = round(p_under * 100.0, 1)
            edge_pct = abs(p_over - 0.5) * 100.0
            confidence = "LOW"
            notes = []

            if total_samples < 18:
                pick, tag = "NO BET", "THIN"
                notes.append("thin sample")
            elif strongest_source < 6:
                pick, tag = "NO BET", "UNSTABLE"
                notes.append("weak source depth")
            elif edge_pct < 8:
                pick, tag = "NO BET", "TIGHT"
                notes.append("edge too small")
            elif p_over >= strong:
                pick, tag = "OVER", "STRONG"
            elif p_over >= edge:
                pick, tag = "OVER", "EDGE"
            elif p_under >= strong:
                pick, tag = "UNDER", "STRONG"
            elif p_under >= edge:
                pick, tag = "UNDER", "EDGE"
            else:
                pick, tag = "NO BET", ""
                notes.append("below threshold")

            if total_samples >= 36 and strongest_source >= 10 and edge_pct >= 14:
                confidence = "HIGH"
            elif total_samples >= 24 and strongest_source >= 8 and edge_pct >= 10:
                confidence = "MEDIUM"

            notes = ", ".join(notes) if notes else ""

        rows.append({
            "Line": ln,
            "Over%": over_pct,
            "Under%": under_pct,
            "Pick": pick,
            "Tag": tag,
            "Confidence": confidence,
            "Notes": notes,
        })

    sum_df = pd.DataFrame(rows)

    sample_info = {
        "n_h2h": int(n_h2h),
        "n_a": int(n_a),
        "n_b": int(n_b),
        "w_h2h": float(w_h2h),
        "w_a": float(w_a),
        "w_b": float(w_b),
        "edge": float(edge),
        "strong": float(strong),
    }
    return sum_df, avg_value, sample_info


def build_ou_one(df1: pd.DataFrame, col: str, lines: list[float], strong: float = 0.65, edge: float = 0.58):
    """Auto-pick using a SINGLE dataframe (e.g. H2H only)."""
    n = len(df1) if col in df1.columns else 0
    if n == 0:
        empty = pd.DataFrame([{"Line": ln, "Over%": None, "Under%": None, "Pick": "NO DATA", "Tag": "", "Confidence": "NO DATA", "Notes": "Missing numeric history"} for ln in lines])
        return empty, float("nan"), {"n": 0}

    rows = []
    avg = float(df1[col].mean())
    for ln in lines:
        over_pct = float((df1[col] > ln).mean())
        under_pct = float((df1[col] < ln).mean())
        pick = "NO BET"
        tag = ""
        confidence = "LOW"
        notes = []
        edge_pct = abs(over_pct - 0.5) * 100.0
        if n < 8:
            pick, tag = "NO BET", "THIN"
            notes.append("thin sample")
        elif edge_pct < 10:
            pick, tag = "NO BET", "TIGHT"
            notes.append("edge too small")
        elif over_pct >= strong:
            pick, tag = "OVER", "STRONG"
        elif under_pct >= strong:
            pick, tag = "UNDER", "STRONG"
        elif over_pct >= edge:
            pick, tag = "OVER", "EDGE"
        elif under_pct >= edge:
            pick, tag = "UNDER", "EDGE"
        if n >= 16 and edge_pct >= 15:
            confidence = "HIGH"
        elif n >= 10 and edge_pct >= 12:
            confidence = "MEDIUM"
        rows.append({
            "Line": ln,
            "Over%": round(over_pct * 100, 1),
            "Under%": round(under_pct * 100, 1),
            "Pick": pick,
            "Tag": tag,
            "Confidence": confidence,
            "Notes": ", ".join(notes) if notes else "",
        })

    summary_df = pd.DataFrame(rows)
    return summary_df, avg, {"n": n}

def render_ou_cards(
    title: str,
    sum_df: pd.DataFrame,
    avg_value: float,
    sample_info: dict | None = None,
    per_row_cols: int = 4,
):
    sample_info = sample_info or {}

    avg_str = "—" if pd.isna(avg_value) else f"{float(avg_value):.2f}"

    # Backward + forward compatible sample/weight caption
    if "samples" in sample_info:
        samples_str = str(sample_info.get("samples", ""))
        weights_str = str(sample_info.get("weights", ""))
    else:
        n_h2h = int(sample_info.get("n_h2h", 0) or 0)
        n_a = int(sample_info.get("n_a", 0) or 0)
        n_b = int(sample_info.get("n_b", 0) or 0)
        samples_str = f"H2H={n_h2h}, A={n_a}, B={n_b}"

        if all(k in sample_info for k in ("w_h2h", "w_a", "w_b")):
            weights_str = f"{sample_info['w_h2h']:.2f}/{sample_info['w_a']:.2f}/{sample_info['w_b']:.2f}"
        else:
            weights_str = ""

    st.subheader(title)
    cap = f"Avg: {avg_str}"
    if samples_str:
        cap += f" | samples: {samples_str}"
    if weights_str:
        cap += f" | weights: {weights_str}"
    st.caption(cap)

    if sum_df is None or sum_df.empty:
        st.info("No data for this metric.")
        return

    rows = sum_df.to_dict("records")
    for i in range(0, len(rows), per_row_cols):
        cols = st.columns(per_row_cols)
        for j in range(per_row_cols):
            if i + j >= len(rows):
                break
            r = rows[i + j]
            line = r.get("Line", "")
            over = r.get("Over%")
            under = r.get("Under%")
            pick = r.get("Pick", "")
            tag = r.get("Tag", "")
            confidence = r.get("Confidence", "")
            notes = r.get("Notes", "")

            over_str = "—" if (over is None or pd.isna(over)) else f"{float(over):.1f}%"
            under_str = "—" if (under is None or pd.isna(under)) else f"{float(under):.1f}%"
            pick_str = pick if not tag else f"{pick} ({tag})"

            with cols[j]:
                st.markdown(f"### O/U {line}")
                st.write(f"Over: **{over_str}**")
                st.write(f"Under: **{under_str}**")
                st.write(f"**{pick_str}**")
                if confidence:
                    st.caption(f"Confidence: {confidence}")
                if notes:
                    st.caption(f"Notes: {notes}")


def ah_winrate(df_team: pd.DataFrame) -> dict:
    """Return AH win-rate stats from a team-view AH dataframe."""
    if df_team is None or df_team.empty or "Result" not in df_team.columns:
        return {"N": 0, "W": 0, "HW": 0, "P": 0, "HL": 0, "L": 0, "Win%": 0.0}
    counts = df_team["Result"].fillna("").value_counts().to_dict()
    W = int(counts.get("W", 0))
    HW = int(counts.get("HW", 0))
    P = int(counts.get("P", 0))
    HL = int(counts.get("HL", 0))
    L = int(counts.get("L", 0))
    N = int(len(df_team))
    win_pct = (W + 0.5 * HW) / N * 100 if N else 0.0
    return {"N": N, "W": W, "HW": HW, "P": P, "HL": HL, "L": L, "Win%": round(win_pct, 1)}

def team_ah_last(df: pd.DataFrame, team: str, n: int, side_filter: str | None = None) -> pd.DataFrame:
    """Build last-N Asian Handicap results from raw match dataframe (home/away mixed)."""
    cols = ["Date", "Side", "Opp", "Score", "Line", "Result", "Pts"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        return pd.DataFrame(columns=cols)
    sub = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()
    # Optional venue filter: "H" = only Home matches, "A" = only Away matches
    if side_filter in ("H", "A"):
        if side_filter == "H":
            sub = sub[sub["HomeTeam"] == team]
        else:
            sub = sub[sub["AwayTeam"] == team]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    if "Date" in sub.columns:
        sub = sub.sort_values("Date", ascending=False)
    sub = sub.head(int(n))
    if "AHh" not in sub.columns:
        return pd.DataFrame(columns=cols)

    def _row(r: pd.Series) -> pd.Series:
        if r.get("HomeTeam") == team:
            side = "H"
            opp = r.get("AwayTeam", "")
            line = r.get("AHh", float("nan"))
            pts = r.get("AH_home_pts", float("nan"))
            res = r.get("AH_home_res", "")
            gf = r.get("FTHG", float("nan"))
            ga = r.get("FTAG", float("nan"))
        else:
            side = "A"
            opp = r.get("HomeTeam", "")
            line = -r.get("AHh", float("nan"))
            pts = r.get("AH_away_pts", float("nan"))
            res = r.get("AH_away_res", "")
            gf = r.get("FTAG", float("nan"))
            ga = r.get("FTHG", float("nan"))

        score = ""
        try:
            if pd.notna(gf) and pd.notna(ga):
                score = f"{int(gf)}-{int(ga)}"
        except Exception:
            score = f"{gf}-{ga}"

        return pd.Series({
            "Date": r.get("Date", pd.NaT),
            "Side": side,
            "Opp": opp,
            "Score": score,
            "Line": line,
            "Result": res,
            "Pts": pts,
        })

    out = sub.apply(_row, axis=1)
    out = out.dropna(subset=["Line", "Pts"])
    return out[cols] if all(c in out.columns for c in cols) else out


def team_cah_last(df: pd.DataFrame, team: str, n: int, side_filter: str | None = None) -> pd.DataFrame:
    """Build last-N Corner Asian Handicap results from raw match dataframe (home/away mixed).

    Uses:
      - AHCh (corner handicap line for Home)
      - HC/AC (home/away corners)
      - Derived columns from load_df: CAH_home_pts/res, CAH_away_pts/res
    """
    cols = ["Date", "Side", "Opp", "Score", "Line", "Result", "Pts"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    if "HomeTeam" not in df.columns or "AwayTeam" not in df.columns:
        return pd.DataFrame(columns=cols)

    sub = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].copy()

    # Optional venue filter: "H" = only Home matches, "A" = only Away matches
    if side_filter in ("H", "A"):
        if side_filter == "H":
            sub = sub[sub["HomeTeam"] == team]
        else:
            sub = sub[sub["AwayTeam"] == team]

    if sub.empty:
        return pd.DataFrame(columns=cols)

    if "Date" in sub.columns:
        sub = sub.sort_values("Date", ascending=False)
    sub = sub.head(int(n))

    # Need corner handicap line + corners
    if "AHCh" not in sub.columns or not {"HC", "AC"}.issubset(sub.columns):
        return pd.DataFrame(columns=cols)

    def _row(r: pd.Series) -> pd.Series:
        if r.get("HomeTeam") == team:
            side = "H"
            opp = r.get("AwayTeam", "")
            line = r.get("AHCh", float("nan"))
            pts = r.get("CAH_home_pts", float("nan"))
            res = r.get("CAH_home_res", "")
            gf = r.get("HC", float("nan"))
            ga = r.get("AC", float("nan"))
        else:
            side = "A"
            opp = r.get("HomeTeam", "")
            line = -r.get("AHCh", float("nan"))
            pts = r.get("CAH_away_pts", float("nan"))
            res = r.get("CAH_away_res", "")
            gf = r.get("AC", float("nan"))
            ga = r.get("HC", float("nan"))

        score = ""
        try:
            if pd.notna(gf) and pd.notna(ga):
                score = f"{int(gf)}-{int(ga)}"
        except Exception:
            score = f"{gf}-{ga}"

        return pd.Series({
            "Date": r.get("Date", pd.NaT),
            "Side": side,
            "Opp": opp,
            "Score": score,
            "Line": line,
            "Result": res,
            "Pts": pts,
        })

    out = sub.apply(_row, axis=1)
    out = out.dropna(subset=["Line", "Pts"])
    return out[cols] if all(c in out.columns for c in cols) else out

def render_ah_panel(team: str, df_team: pd.DataFrame, show_table: bool = True):
    st.markdown(f"### {team}")
    stats = ah_winrate(df_team)
    m1, m2, m3 = st.columns(3)
    m1.metric("AH Win%", f"{stats['Win%']}%")
    m2.metric("W-HW-P", f"{stats['W']}-{stats['HW']}-{stats['P']}")
    m3.metric("HL-L", f"{stats['HL']}-{stats['L']}")
    if show_table and df_team is not None and not df_team.empty:
        st.dataframe(df_team, use_container_width=True)


def build_market_health_report(
    team_a: str,
    team_b: str,
    n: int,
    venue_mode: bool,
    h2h_df: pd.DataFrame,
    matched_odds: dict | None,
    ou_infos: list[tuple[str, dict]],
    btts: dict,
    ah_a: pd.DataFrame,
    ah_b: pd.DataFrame,
) -> list[str]:
    warnings = []

    if matched_odds is None:
        warnings.append("No fetched odds snapshot is attached. Picks are running only on historical stats.")
    elif not matched_odds.get("matched"):
        warnings.append("Fetched odds did not match the selected teams exactly, so market context is incomplete.")

    for label, info in ou_infos:
        total_samples = int(info.get("n_h2h", 0) or 0) + int(info.get("n_a", 0) or 0) + int(info.get("n_b", 0) or 0)
        if total_samples < 18:
            warnings.append(f"{label}: sample depth is thin ({total_samples} combined rows).")

    if h2h_df is None or len(h2h_df) < 3:
        warnings.append("H2H sample is very thin, so matchup-specific reads are weak.")

    blend = btts.get("Blended%")
    if blend is None:
        warnings.append("BTTS signal has missing data.")
    elif 45 <= float(blend) <= 55:
        warnings.append("BTTS is near 50/50, so it should not be treated as a strong edge.")

    ah_wr_a = ah_winrate(ah_a).get("Win%", 0.0)
    ah_wr_b = ah_winrate(ah_b).get("Win%", 0.0)
    if abs(float(ah_wr_a) - float(ah_wr_b)) < 8:
        warnings.append(f"AH form between {team_a} and {team_b} is close, so handicap edge is weak.")

    if venue_mode and n > 20:
        warnings.append("Venue mode with large N can mix old context into current form. Re-check with smaller N like 8-15.")

    return warnings


def evaluate_ou_pick(actual_value: float, line: float, pick: str) -> float | None:
    if pd.isna(actual_value):
        return None
    if pick == "OVER":
        return 1.0 if float(actual_value) > float(line) else 0.0
    if pick == "UNDER":
        return 1.0 if float(actual_value) <= float(line) else 0.0
    return None


def run_ou_backtest(
    df: pd.DataFrame,
    col: str,
    lines: list[float],
    last_n_window: int,
    min_history_matches: int = 30,
    require_tag: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    if "Date" in work.columns:
        work = work.sort_values("Date", ascending=True).reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    records = []
    for idx, row in work.iterrows():
        if idx < min_history_matches:
            continue
        home = row.get("HomeTeam")
        away = row.get("AwayTeam")
        actual_value = row.get(col)
        if pd.isna(actual_value):
            continue

        history = work.iloc[:idx].copy()
        a_hist = last_n_home(history, home, last_n_window)
        b_hist = last_n_away(history, away, last_n_window)
        h_hist = h2h(history, home, away, last_n_window)
        summary_df, _, info = build_ou_summary(a_hist, b_hist, h_hist, col=col, lines=lines)
        if summary_df.empty:
            continue

        for rec in summary_df.to_dict("records"):
            pick = rec.get("Pick")
            tag = rec.get("Tag")
            if pick not in {"OVER", "UNDER"}:
                continue
            if require_tag and tag not in {"EDGE", "STRONG"}:
                continue
            result = evaluate_ou_pick(actual_value, rec.get("Line"), pick)
            if result is None:
                continue
            total_samples = int(info.get("n_h2h", 0) or 0) + int(info.get("n_a", 0) or 0) + int(info.get("n_b", 0) or 0)
            records.append({
                "MarketCol": col,
                "Line": rec.get("Line"),
                "Pick": pick,
                "Tag": tag,
                "Confidence": rec.get("Confidence"),
                "Win": result,
                "Samples": total_samples,
            })

    if not records:
        return pd.DataFrame()

    bt = pd.DataFrame(records)
    grouped = (
        bt.groupby(["MarketCol", "Line", "Pick", "Tag", "Confidence"], dropna=False)
        .agg(Bets=("Win", "size"), HitRate=("Win", "mean"), AvgSamples=("Samples", "mean"))
        .reset_index()
    )
    grouped["HitRate"] = (grouped["HitRate"] * 100).round(1)
    grouped["AvgSamples"] = grouped["AvgSamples"].round(1)
    grouped = grouped.sort_values(["Bets", "HitRate"], ascending=[False, False])
    return grouped


def canonical_backtest_line(market_col: str, line: float, pick: str) -> float:
    try:
        line = float(line)
    except Exception:
        return line

    integer_like_markets = {"TG", "H1G", "TC", "TCards"}
    if market_col not in integer_like_markets:
        return line

    if abs(line - round(line)) < 1e-9:
        if str(pick).upper() == "UNDER":
            return line + 0.5
        return line
    if abs((line * 2) - round(line * 2)) < 1e-9 and abs(line - int(line) - 0.5) < 1e-9:
        if str(pick).upper() == "OVER":
            return line - 0.5
        return line
    return line


def merge_equivalent_backtest_lines(bt_df: pd.DataFrame) -> pd.DataFrame:
    if bt_df is None or bt_df.empty:
        return bt_df

    work = bt_df.copy()
    work["CanonicalLine"] = work.apply(
        lambda r: canonical_backtest_line(r.get("MarketCol"), r.get("Line"), r.get("Pick")),
        axis=1,
    )

    merged = (
        work.groupby(["MarketCol", "CanonicalLine", "Pick", "Tag", "Confidence"], dropna=False)
        .agg(
            Bets=("Bets", "sum"),
            WeightedHits=("HitRate", lambda s: 0.0),
            AvgSamples=("AvgSamples", "mean"),
            SourceLines=("Line", lambda s: sorted({float(x) for x in s})),
        )
        .reset_index()
    )

    # Recompute weighted hit rate from original rows inside each grouped bucket.
    weighted_rows = []
    for _, row in merged.iterrows():
        subset = work[
            (work["MarketCol"] == row["MarketCol"])
            & (work["CanonicalLine"] == row["CanonicalLine"])
            & (work["Pick"] == row["Pick"])
            & (work["Tag"] == row["Tag"])
            & (work["Confidence"] == row["Confidence"])
        ]
        total_bets = float(subset["Bets"].sum())
        weighted_hit = ((subset["HitRate"] * subset["Bets"]).sum() / total_bets) if total_bets else 0.0
        weighted_rows.append(weighted_hit)

    merged["HitRate"] = [round(x, 1) for x in weighted_rows]
    merged["AvgSamples"] = merged["AvgSamples"].round(1)
    merged["Line"] = merged["CanonicalLine"]
    merged["SourceLines"] = merged["SourceLines"].apply(lambda xs: ", ".join(str(int(x)) if float(x).is_integer() else str(x) for x in xs))
    merged = merged.drop(columns=["CanonicalLine", "WeightedHits"])
    merged = merged[["MarketCol", "Line", "Pick", "Tag", "Confidence", "Bets", "HitRate", "AvgSamples", "SourceLines"]]
    merged = merged.sort_values(["Bets", "HitRate"], ascending=[False, False]).reset_index(drop=True)
    return merged

def call_ai_api(api_url: str, api_key: str, payload: dict):
    """Small HTTP client for the AI panel.

    - Default: POST JSON `payload` to `api_url`, with optional Bearer token (api_key)
    - Gemini (generativelanguage.googleapis.com):
        * Auto-attach API key (query param + x-goog-api-key header)
        * Convert payload into Gemini `contents` format
        * Auto-fix common model issues (e.g., gemini-3-* requires *-preview model IDs)
        * If model is not found, call models.list and retry with the closest match
    """
    import json
    import re
    from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

    import requests

    api_url = (api_url or "").strip()
    api_key = (api_key or "").strip()

    if not api_url:
        return {"ok": False, "error": "Missing API URL"}

    is_gemini = "generativelanguage.googleapis.com" in api_url
    headers = {"Content-Type": "application/json"}

    def _attach_key(u: str) -> str:
        if not api_key:
            return u
        uu = urlparse(u)
        q = dict(parse_qsl(uu.query, keep_blank_values=True))
        if "key" not in q:
            q["key"] = api_key
            uu = uu._replace(query=urlencode(q))
        return urlunparse(uu)

    def _parse_gemini(u: str):
        """Return (version, model_id, method) from a Gemini REST url, or (None,None,None)."""
        uu = urlparse(u)
        m = re.search(r"/(v1beta|v1)/models/([^:/?#]+)(?::([^/?#]+))?", uu.path)
        if not m:
            return None, None, None
        version = m.group(1)
        model_id = m.group(2)
        method = m.group(3) or "generateContent"
        return version, model_id, method

    def _normalize_gemini_model(model_id: str) -> str:
        mid = (model_id or "").strip()
        # Common user mistake: gemini-3-flash (but docs use gemini-3-flash-preview)
        if mid.startswith("gemini-3-") and "preview" not in mid:
            if mid.endswith("-flash"):
                return "gemini-3-flash-preview"
            if mid.endswith("-pro"):
                return "gemini-3-pro-preview"
            if mid.endswith("-pro-image"):
                return "gemini-3-pro-image-preview"
            return mid + "-preview"
        return mid

    def _rebuild_gemini_url(u: str, version: str, model_id: str, method: str) -> str:
        uu = urlparse(u)
        new_path = f"/{version}/models/{model_id}:{method}"
        uu = uu._replace(path=new_path)
        return urlunparse(uu)

    def _list_models(version: str):
        if not api_key:
            return []
        list_url = f"https://generativelanguage.googleapis.com/{version}/models"
        list_url = _attach_key(list_url)
        hh = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        try:
            rr = requests.get(list_url, headers=hh, timeout=20)
            if rr.status_code >= 400:
                return []
            data = rr.json()
            return data.get("models", []) or []
        except Exception:
            return []

    def _choose_model(models: list, desired: str):
        desired = (desired or "").strip()
        desired_norm = desired.replace("models/", "")
        want_flash = "flash" in desired_norm
        want_pro = "-pro" in desired_norm and "pro-image" not in desired_norm
        want_lite = "lite" in desired_norm
        want_25 = desired_norm.startswith("gemini-2.5")
        want_3 = desired_norm.startswith("gemini-3")

        best = None
        best_score = -1
        for mobj in models:
            name = (mobj.get("name") or "")
            mid = name.replace("models/", "")
            if not mid:
                continue

            # Prefer models that support generateContent when info is available
            methods = mobj.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue

            score = 0
            if mid == desired_norm:
                score += 100
            if want_3 and mid.startswith("gemini-3"):
                score += 20
            if want_25 and mid.startswith("gemini-2.5"):
                score += 20
            if want_flash and "flash" in mid:
                score += 10
            if want_pro and "-pro" in mid:
                score += 10
            if want_lite and "lite" in mid:
                score += 6
            if "preview" in mid and want_3:
                score += 2

            # Token overlap heuristic
            for tok in [t for t in re.split(r"[-_.]", desired_norm) if t]:
                if tok in mid:
                    score += 1

            if score > best_score:
                best_score = score
                best = mid

        return best

    # Build request JSON
    req_json = payload
    url = api_url

    # Auth
    if api_key:
        if is_gemini:
            url = _attach_key(url)
            headers["x-goog-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"

    # Gemini payload conversion
    if is_gemini:
        parts = []
        ptxt = (payload or {}).get("prompt", "")
        if ptxt:
            parts.append(ptxt)
        ctx = (payload or {}).get("context", "")
        if ctx:
            parts.append("Context:\n" + ctx)

        tables = (payload or {}).get("tables", None)
        if tables is not None:
            try:
                tables_str = json.dumps(tables, ensure_ascii=False, indent=2)
            except Exception:
                tables_str = str(tables)
            parts.append("Tables (JSON):\n" + tables_str)

        hc = (payload or {}).get("handicap", None)
        if hc is not None:
            try:
                hc_str = json.dumps(hc, ensure_ascii=False, indent=2)
            except Exception:
                hc_str = str(hc)
            parts.append("Handicap summary (JSON):\n" + hc_str)

        market_snapshot = (payload or {}).get("market_snapshot")
        if market_snapshot is not None:
            try:
                market_snapshot_str = json.dumps(market_snapshot, ensure_ascii=False, indent=2)
            except Exception:
                market_snapshot_str = str(market_snapshot)
            parts.append("Market snapshot (JSON):\n" + market_snapshot_str)

        odds_snapshot = (payload or {}).get("odds_api_snapshot")
        if odds_snapshot is not None:
            try:
                odds_snapshot_str = json.dumps(odds_snapshot, ensure_ascii=False, indent=2)
            except Exception:
                odds_snapshot_str = str(odds_snapshot)
            parts.append("Odds API snapshot (JSON):\n" + odds_snapshot_str)

        record_bias = (payload or {}).get("historical_record_bias")
        if record_bias is not None:
            try:
                record_bias_str = json.dumps(record_bias, ensure_ascii=False, indent=2)
            except Exception:
                record_bias_str = str(record_bias)
            parts.append("Historical betting record bias (JSON):\n" + record_bias_str)

        text = "\n\n".join(parts).strip()
        req_json = {"contents": [{"role": "user", "parts": [{"text": text}]}]}

        # Normalize model id inside URL (gemini-3-* -> *-preview)
        v, mid, method = _parse_gemini(url)
        if v and mid:
            mid2 = _normalize_gemini_model(mid)
            if mid2 != mid:
                url = _rebuild_gemini_url(url, v, mid2, method)
                url = _attach_key(url)

    # POST with auto-repair for Gemini
    attempts = 3 if is_gemini else 1
    last_err = None

    for _ in range(attempts):
        try:
            r = requests.post(url, headers=headers, json=req_json, timeout=60)
            if r.status_code < 400:
                data = r.json()
                if is_gemini:
                    try:
                        txt = data["candidates"][0]["content"]["parts"][0].get("text", "")
                    except Exception:
                        txt = json.dumps(data, ensure_ascii=False, indent=2)
                    return {"ok": True, "text": txt, "raw": data}
                return {"ok": True, "raw": data}

            # Error
            last_err = {"status": r.status_code, "text": r.text}

            if not is_gemini or r.status_code not in (400, 404):
                break

            # Try to repair: list models and swap to closest match; also try v1 <-> v1beta
            v, desired_mid, method = _parse_gemini(url)
            if not v or not desired_mid:
                break

            desired_mid_norm = _normalize_gemini_model(desired_mid)

            # If URL still has un-normalized model, fix it first
            if desired_mid_norm != desired_mid:
                url = _rebuild_gemini_url(url, v, desired_mid_norm, method)
                url = _attach_key(url)
                continue

            # 1) Try models.list on current version
            models = _list_models(v)
            chosen = _choose_model(models, desired_mid_norm)

            # 2) If not found, try the other version
            if not chosen:
                alt_v = "v1" if v == "v1beta" else "v1beta"
                models2 = _list_models(alt_v)
                chosen = _choose_model(models2, desired_mid_norm)
                if chosen:
                    v = alt_v

            if chosen:
                url = _rebuild_gemini_url(url, v, chosen, method)
                url = _attach_key(url)
                continue

            # No repair possible
            break

        except Exception as e:
            last_err = {"error": str(e)}
            break

    return {"ok": False, **(last_err or {"error": "Unknown error"})}


def call_openai_api(model: str, api_key: str, payload: dict):
    """Call OpenAI Responses API (text-only) and return {'ok': bool, 'text': str, 'raw': dict}."""
    import json

    model = (model or "").strip()
    api_key = (api_key or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {"ok": False, "error": "Missing OPENAI_API_KEY (set env var OPENAI_API_KEY)."}

    try:
        from openai import OpenAI
    except Exception:
        return {"ok": False, "error": "Missing dependency: openai. Install with: python -m pip install -U openai"}

    parts = []
    ptxt = (payload or {}).get("prompt", "")
    if ptxt:
        parts.append(ptxt)

    ctx = (payload or {}).get("context", "")
    if ctx:
        parts.append("Context:\n" + str(ctx))

    tables = (payload or {}).get("tables", None)
    if tables is not None:
        try:
            tables_str = json.dumps(tables, ensure_ascii=False, indent=2)
        except Exception:
            tables_str = str(tables)
        parts.append("Tables (JSON):\n" + tables_str)

    hc = (payload or {}).get("handicap", None)
    if hc is not None:
        try:
            hc_str = json.dumps(hc, ensure_ascii=False, indent=2)
        except Exception:
            hc_str = str(hc)
        parts.append("Handicap summary (JSON):\n" + hc_str)

    market_snapshot = (payload or {}).get("market_snapshot")
    if market_snapshot is not None:
        try:
            market_str = json.dumps(market_snapshot, ensure_ascii=False, indent=2)
        except Exception:
            market_str = str(market_snapshot)
        parts.append("Market snapshot (JSON):\n" + market_str)

    odds_snapshot = (payload or {}).get("odds_api_snapshot")
    if odds_snapshot is not None:
        try:
            odds_str = json.dumps(odds_snapshot, ensure_ascii=False, indent=2)
        except Exception:
            odds_str = str(odds_snapshot)
        parts.append("Odds API snapshot (JSON):\n" + odds_str)

    record_bias = (payload or {}).get("historical_record_bias")
    if record_bias is not None:
        try:
            record_bias_str = json.dumps(record_bias, ensure_ascii=False, indent=2)
        except Exception:
            record_bias_str = str(record_bias)
        parts.append("Historical betting record bias (JSON):\n" + record_bias_str)

    raw_market_summary = (payload or {}).get("raw_market_summary")
    if raw_market_summary:
        parts.append("Raw market summary (text):\n" + str(raw_market_summary))

    input_text = "\n\n".join([p for p in parts if p]).strip()

    def _extract_text_from_raw(raw: dict) -> str:
        if not isinstance(raw, dict):
            return ""
        out = []
        for item in raw.get("output", []) or []:
            for c in (item.get("content", []) or []):
                ctype = c.get("type")
                if ctype in ("output_text", "text"):
                    t = c.get("text") or ""
                    if t:
                        out.append(t)
                elif ctype == "refusal":
                    t = c.get("refusal") or ""
                    if t:
                        out.append(t)
        return "\n".join([x for x in out if isinstance(x, str)]).strip()

    try:
        client = OpenAI(api_key=api_key)
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_text}],
                }
            ],
            max_output_tokens=int((payload or {}).get("max_output_tokens", 1500)),
            reasoning={"effort": (payload or {}).get("reasoning_effort", "low")},
            text={"format": {"type": "text"}, "verbosity": (payload or {}).get("text_verbosity", "low")},
        )

        out_text = (getattr(resp, "output_text", "") or "").strip()

        raw = None
        try:
            raw = resp.model_dump()
        except Exception:
            try:
                raw = resp.to_dict()
            except Exception:
                raw = None

        if not out_text and raw:
            out_text = _extract_text_from_raw(raw)

        if raw and raw.get("status") == "incomplete" and (raw.get("incomplete_details") or {}).get("reason") == "max_output_tokens":
            return {
                "ok": False,
                "error": "OpenAI returned no visible text because max_output_tokens was too low (GPT-5 uses this budget for hidden reasoning too). Increase max_output_tokens or set reasoning.effort='low'.",
                "text": out_text,
                "raw": raw,
            }

        return {"ok": True, "text": out_text, "raw": raw}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def fetch_odds_from_the_odds_api(
    sport_key: str,
    api_key: str,
    regions: str = "eu",
    markets: str = "h2h,totals",
    odds_format: str = "decimal",
    date_format: str = "unix",
) -> dict:
    """Fetch odds from TheOddsAPI (https://the-odds-api.com)."""
    import requests

    if not api_key:
        return {"ok": False, "error": "Missing TheOddsAPI key."}

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return {"ok": True, "raw": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e), "status_code": getattr(e, "response", None) and e.response.status_code}


def fetch_odds_api_io_events(
    api_key: str,
    sport: str = "football",
    league: str | None = None,
    bookmaker: str | None = None,
    days_ahead: int = 7,
) -> dict:
    import datetime
    import requests

    if not api_key:
        return {"ok": False, "error": "Missing odds-api.io API key."}

    now = datetime.datetime.utcnow()
    cutoff = now + datetime.timedelta(days=int(days_ahead))
    params = {
        "apiKey": api_key,
        "sport": sport,
        "status": "pending",
        "from": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if league:
        params["league"] = league
    if bookmaker:
        params["bookmaker"] = bookmaker

    try:
        resp = requests.get("https://api.odds-api.io/v3/events", params=params, timeout=30)
        resp.raise_for_status()
        return {"ok": True, "raw": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e), "status_code": getattr(e, "response", None) and e.response.status_code}


def fetch_odds_api_io_leagues(api_key: str, sport: str = "football", include_all: bool = True) -> dict:
    import requests

    if not api_key:
        return {"ok": False, "error": "Missing odds-api.io API key."}

    params = {
        "apiKey": api_key,
        "sport": sport,
    }
    if include_all:
        params["all"] = "true"

    try:
        resp = requests.get("https://api.odds-api.io/v3/leagues", params=params, timeout=30)
        resp.raise_for_status()
        return {"ok": True, "raw": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e), "status_code": getattr(e, "response", None) and e.response.status_code}


def fetch_odds_api_io_event_odds(api_key: str, event_id: int | str, bookmakers: str) -> dict:
    import requests

    if not api_key:
        return {"ok": False, "error": "Missing odds-api.io API key."}
    if not event_id:
        return {"ok": False, "error": "Missing event ID."}

    params = {
        "apiKey": api_key,
        "eventId": event_id,
        "bookmakers": bookmakers,
    }
    try:
        resp = requests.get("https://api.odds-api.io/v3/odds", params=params, timeout=30)
        resp.raise_for_status()
        return {"ok": True, "raw": resp.json()}
    except Exception as e:
        return {"ok": False, "error": str(e), "status_code": getattr(e, "response", None) and e.response.status_code}


def discover_record_workbook() -> Path | None:
    candidates = [
        Path("D:/banh bóng.xlsx"),
        Path("D:/banh bong.xlsx"),
    ]
    for p in candidates:
        if p.exists():
            return p

    matches = sorted(Path("D:/").glob("banh*.xlsx"))
    return matches[0] if matches else None


def build_google_sheet_xlsx_url(sheet_url: str) -> str | None:
    import re

    sheet_url = (sheet_url or "").strip()
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not m:
        return None
    return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=xlsx"


def build_google_sheet_csv_url(sheet_url: str, gid: str = "0") -> str | None:
    import re

    sheet_url = (sheet_url or "").strip()
    m = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not m:
        return None
    return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={gid}"


def normalize_record_result(x) -> str:
    s = str(x or "").strip().lower()
    mapping = {
        "win": "win",
        "w": "win",
        "halfwin": "halfwin",
        "w1/2": "halfwin",
        "draw": "draw",
        "push": "draw",
        "lose": "lose",
        "loss": "lose",
        "halflose": "halflose",
        "l1/2": "halflose",
    }
    return mapping.get(s, s)


def result_to_score(result: str) -> float | None:
    return {
        "win": 1.0,
        "halfwin": 0.5,
        "draw": 0.0,
        "lose": -1.0,
        "halflose": -0.5,
    }.get(normalize_record_result(result))


def infer_market_family(bet_text: str) -> str:
    s = str(bet_text or "").strip().lower()
    if not s:
        return "unknown"
    if "btts" in s:
        return "btts"
    if "thẻ" in s or "card" in s:
        return "cards"
    if "góc" in s or "corner" in s:
        return "corners"
    if "ht" in s or "1h" in s:
        return "1h_goals"
    if any(tok in s for tok in ["tài", "xỉu", "trên", "dưới", "over", "under", "o ", "u "]):
        return "goals_ou"
    if any(tok in s for tok in ["+", "-", " pk", " 0", "đồng banh"]):
        return "handicap"
    return "other"


def odds_band(rate) -> str:
    try:
        x = float(rate)
    except Exception:
        return "unknown"
    if x < 0.85:
        return "<0.85"
    if x < 1.0:
        return "0.85-0.99"
    if x < 1.15:
        return "1.00-1.14"
    return ">=1.15"


@st.cache_data(show_spinner=False)
def load_record_history(_: float = 0.0) -> dict:
    import io
    import requests

    path = discover_record_workbook()
    source_label = None
    workbook_mtime = None
    sheet_frames = []
    load_errors = []

    google_xlsx_url = build_google_sheet_xlsx_url(GOOGLE_RECORD_SHEET_URL)
    if google_xlsx_url:
        try:
            resp = requests.get(google_xlsx_url, timeout=30)
            resp.raise_for_status()
            workbook_mtime = float(len(resp.content))
            xl = pd.ExcelFile(io.BytesIO(resp.content))
            for sheet in xl.sheet_names:
                try:
                    sheet_frames.append((sheet, pd.read_excel(io.BytesIO(resp.content), sheet_name=sheet)))
                except Exception as e:
                    load_errors.append(f"Google Sheet tab '{sheet}' parse failed: {e}")
            source_label = f"google_sheet:{GOOGLE_RECORD_SHEET_URL}"
        except Exception as e:
            load_errors.append(f"Google Sheet load failed: {e}")

    if not sheet_frames:
        google_csv_url = build_google_sheet_csv_url(GOOGLE_RECORD_SHEET_URL, gid="0")
        if google_csv_url:
            try:
                resp = requests.get(google_csv_url, timeout=30)
                resp.raise_for_status()
                workbook_mtime = float(len(resp.content))
                csv_df = pd.read_csv(io.BytesIO(resp.content))
                sheet_frames.append(("GoogleSheet gid0", csv_df))
                source_label = f"google_sheet_csv:{GOOGLE_RECORD_SHEET_URL}"
                load_errors.append("Fell back to Google Sheet CSV because workbook parsing is unavailable in this environment.")
            except Exception as e:
                load_errors.append(f"Google Sheet CSV fallback failed: {e}")

    if not sheet_frames:
        if path is None or not path.exists():
            return {"ok": False, "error": "No betting record workbook found from Google Sheet or D:/", "details": load_errors}
        try:
            workbook_mtime = path.stat().st_mtime
            xl = pd.ExcelFile(path)
            for sheet in xl.sheet_names:
                try:
                    sheet_frames.append((sheet, pd.read_excel(path, sheet_name=sheet)))
                except Exception as e:
                    load_errors.append(f"Local workbook tab '{sheet}' parse failed: {e}")
            source_label = f"local_file:{path}"
        except Exception as e:
            return {"ok": False, "error": f"Record workbook exists but could not be parsed: {e}", "details": load_errors}

    frames = []
    monthly_model_rows = []

    for sheet, df in sheet_frames:
        if sheet.strip().lower() in {"form responses 1", "tính xiên"}:
            continue
        if df.empty:
            continue

        rename_map = {}
        for c in df.columns:
            cs = str(c).strip().lower()
            if cs in {"ngày", "ngay"}:
                rename_map[c] = "date"
            elif cs == "trận":
                rename_map[c] = "match"
            elif cs == "giải":
                rename_map[c] = "league"
            elif cs in {"kèo", "keo"}:
                rename_map[c] = "bet"
            elif cs == "rate":
                rename_map[c] = "rate"
            elif cs == "tỷ số":
                rename_map[c] = "score"
            elif cs == "kết quả":
                rename_map[c] = "result"
            elif cs == "tiền thắng":
                rename_map[c] = "profit"
            elif cs == "model":
                rename_map[c] = "model_pick"
        df = df.rename(columns=rename_map)

        required = {"match", "league", "bet", "rate", "result"}
        if required.issubset(df.columns):
            keep = [c for c in ["date", "match", "league", "bet", "rate", "score", "result", "profit", "model_pick"] if c in df.columns]
            sub = df[keep].copy()
            sub["sheet"] = sheet
            sub["result_norm"] = sub["result"].apply(normalize_record_result)
            sub["result_score"] = sub["result_norm"].apply(result_to_score)
            sub["market_family"] = sub["bet"].apply(infer_market_family)
            sub["odds_band"] = sub["rate"].apply(odds_band)
            sub["profit"] = pd.to_numeric(sub.get("profit"), errors="coerce")
            frames.append(sub)

        lower_cols = {str(c).strip().lower(): c for c in df.columns}
        summary_needed = {"model", "used", "win+halfwin", "rate", "p/l"}
        if summary_needed.issubset(lower_cols.keys()):
            mm = pd.DataFrame({
                "sheet": sheet,
                "model": df[lower_cols["model"]],
                "used": pd.to_numeric(df[lower_cols["used"]], errors="coerce"),
                "win_half": pd.to_numeric(df[lower_cols["win+halfwin"]], errors="coerce"),
                "hit_rate": pd.to_numeric(df[lower_cols["rate"]], errors="coerce"),
                "p_l": pd.to_numeric(df[lower_cols["p/l"]], errors="coerce"),
            }).dropna(subset=["model", "used"], how="any")
            mm["model"] = mm["model"].astype(str).str.strip()
            mm = mm[mm["model"].ne("") & mm["model"].ne("nan")]
            monthly_model_rows.append(mm)

    records = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    model_monthly = pd.concat(monthly_model_rows, ignore_index=True) if monthly_model_rows else pd.DataFrame()

    return {
        "ok": True,
        "path": str(path) if path else None,
        "source": source_label,
        "mtime": workbook_mtime,
        "records": records,
        "model_monthly": model_monthly,
        "details": load_errors,
    }


def build_record_bias_payload(record_data: dict, league: str, provider: str, model_name: str | None = None) -> dict | None:
    if not isinstance(record_data, dict) or not record_data.get("ok"):
        return None

    records = record_data.get("records")
    model_monthly = record_data.get("model_monthly")
    if not isinstance(records, pd.DataFrame) or records.empty:
        return {
            "record_path": record_data.get("path"),
            "matched": False,
            "message": "Record workbook loaded but no clean bet rows were parsed.",
        }

    clean = records[records["result_score"].notna()].copy()
    if clean.empty:
        return {
            "record_path": record_data.get("path"),
            "matched": False,
            "message": "Record workbook loaded but result normalization produced no usable rows.",
        }

    overall = {
        "bets": int(len(clean)),
        "avg_score": round(float(clean["result_score"].mean()), 4),
        "avg_profit": round(float(clean["profit"].dropna().mean()), 2) if clean["profit"].dropna().size else None,
    }

    league_df = clean[clean["league"].astype(str).str.strip().eq(str(league).strip())].copy()
    league_summary = None
    if not league_df.empty:
        league_summary = {
            "league": league,
            "bets": int(len(league_df)),
            "avg_score": round(float(league_df["result_score"].mean()), 4),
            "avg_profit": round(float(league_df["profit"].dropna().mean()), 2) if league_df["profit"].dropna().size else None,
            "markets": (
                league_df.groupby("market_family")
                .agg(bets=("result_score", "size"), avg_score=("result_score", "mean"))
                .reset_index()
                .sort_values(["bets", "avg_score"], ascending=[False, False])
                .head(6)
                .to_dict("records")
            ),
        }

    odds_summary = (
        clean.groupby("odds_band")
        .agg(bets=("result_score", "size"), avg_score=("result_score", "mean"))
        .reset_index()
        .sort_values("bets", ascending=False)
        .to_dict("records")
    )

    model_summary = None
    if isinstance(model_monthly, pd.DataFrame) and not model_monthly.empty and model_name:
        sub = model_monthly[model_monthly["model"].astype(str).str.strip().eq(str(model_name).strip())].copy()
        if not sub.empty:
            model_summary = {
                "model": model_name,
                "months": int(len(sub)),
                "used_total": int(sub["used"].fillna(0).sum()),
                "avg_hit_rate": round(float(sub["hit_rate"].replace("", pd.NA).dropna().astype(float).mean()), 4) if sub["hit_rate"].dropna().size else None,
                "total_p_l": round(float(sub["p_l"].fillna(0).sum()), 2),
                "by_sheet": sub[["sheet", "used", "hit_rate", "p_l"]].to_dict("records"),
            }

    return {
        "record_path": record_data.get("path"),
        "matched": True,
        "provider": provider,
        "model": model_name,
        "overall": overall,
        "league_summary": league_summary,
        "odds_bands": odds_summary,
        "model_summary": model_summary,
    }

def render_ai_result(result):
    """Pretty renderer for AI output. Accepts dict (preferred) or str."""
    import json

    data = result
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except Exception:
            data = {"text": result}

    if not isinstance(data, dict):
        st.write(data)
        return

    # Normalize common fields
    primary = data.get("primary_recommendation") or data.get("primary") or {}
    bet = None
    confidence = None
    if isinstance(primary, dict):
        bet = primary.get("bet") or primary.get("pick") or primary.get("recommendation")
        confidence = primary.get("confidence") or data.get("confidence")
    elif isinstance(primary, str):
        bet = primary
        confidence = data.get("confidence")

    if bet is None:
        bet = data.get("bet") or data.get("recommendation") or data.get("pick")
    if confidence is None:
        confidence = data.get("confidence") or data.get("conf")

    analysis_text = data.get("match_analysis") or data.get("analysis") or data.get("reasoning") or data.get("text")

    model_insights = data.get("model_insights") or data.get("insights") or []
    risk_factors = data.get("risk_factors") or data.get("risks") or []
    alternatives = data.get("alternative_bets") or data.get("alternatives") or []
    verdict = data.get("verdict") or data.get("conclusion")

    st.markdown("### 🎯 PRIMARY RECOMMENDATION")
    if bet:
        st.markdown(f"**Bet:** {bet}")
    if confidence:
        st.markdown(f"**Confidence:** {confidence}")

    if analysis_text:
        st.markdown("### 📊 MATCH ANALYSIS")
        st.write(analysis_text)

    if model_insights:
        st.markdown("### 📈 MODEL INSIGHTS")
        try:
            mi_df = pd.DataFrame(model_insights)
            st.dataframe(mi_df, use_container_width=True, hide_index=True)
        except Exception:
            st.write(model_insights)

    if risk_factors:
        st.markdown("### ⚠️ RISK FACTORS")
        if isinstance(risk_factors, (list, tuple)):
            for x in risk_factors:
                st.write(f"• {x}")
        else:
            st.write(risk_factors)

    if alternatives:
        st.markdown("### 🔁 ALTERNATIVE BETS")
        if isinstance(alternatives, (list, tuple)):
            for i, x in enumerate(alternatives, 1):
                st.write(f"{i}. {x}")
        else:
            st.write(alternatives)

    if verdict:
        st.markdown("### 💡 VERDICT")
        st.write(verdict)

    with st.expander("Raw AI JSON"):
        st.json(data)

def load_df(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Parse date (Football-Data is usually day-first)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

    # Derived: total goals / 1H goals / corners
    if {"FTHG", "FTAG"}.issubset(df.columns):
        df["TG"] = _num(df["FTHG"]) + _num(df["FTAG"])
    if {"HTHG", "HTAG"}.issubset(df.columns):
        df["H1G"] = _num(df["HTHG"]) + _num(df["HTAG"])
    if {"HC", "AC"}.issubset(df.columns):
        df["TC"] = _num(df["HC"]) + _num(df["AC"])

    # Cards: Yellow + 3*Red
    if {"HY", "AY"}.issubset(df.columns):
        df["TY"] = _num(df["HY"]) + _num(df["AY"])
    if {"HR", "AR"}.issubset(df.columns):
        df["TR"] = _num(df["HR"]) + _num(df["AR"])
    if {"TY", "TR"}.issubset(df.columns):
        df["TCards"] = df["TY"].fillna(0) + 3 * df["TR"].fillna(0)
    elif "TY" in df.columns:
        df["TCards"] = df["TY"]

    # BTTS (FT)
    if {"FTHG", "FTAG"}.issubset(df.columns):
        df["BTTS"] = (_num(df["FTHG"]) > 0) & (_num(df["FTAG"]) > 0)
    
    # --- Asian Handicap results (dựa trên AHh) ---
    if "AHh" in df.columns and {"FTHG","FTAG"}.issubset(df.columns):
        df["AHh"] = pd.to_numeric(df["AHh"], errors="coerce")

        # Kèo cho HOME (dùng AHh trực tiếp)
        df["AH_home_pts"] = df.apply(
            lambda r: ah_points(r["FTHG"], r["FTAG"], r["AHh"]), axis=1
        )
        # Kèo cho AWAY (line ngược lại)
        df["AH_away_pts"] = df.apply(
            lambda r: ah_points(r["FTAG"], r["FTHG"], -r["AHh"]), axis=1
        )

        df["AH_home_res"] = df["AH_home_pts"].apply(ah_label)
        df["AH_away_res"] = df["AH_away_pts"].apply(ah_label)

    # --- Corner Asian Handicap results (dựa trên AHCh, corners) ---
    # AHCh: line handicap theo số góc cho Home (âm = Home chấp góc).
    # Dựa trên HC/AC (Home/Away corners).
    if "AHCh" in df.columns and {"HC","AC"}.issubset(df.columns):
        df["AHCh"] = pd.to_numeric(df["AHCh"], errors="coerce")

        # Kèo cho HOME (dùng AHCh trực tiếp)
        df["CAH_home_pts"] = df.apply(
            lambda r: ah_points(r["HC"], r["AC"], r["AHCh"]), axis=1
        )
        # Kèo cho AWAY (line ngược lại)
        df["CAH_away_pts"] = df.apply(
            lambda r: ah_points(r["AC"], r["HC"], -r["AHCh"]), axis=1
        )

        df["CAH_home_res"] = df["CAH_home_pts"].apply(ah_label)
        df["CAH_away_res"] = df["CAH_away_pts"].apply(ah_label)


    return df

league = st.sidebar.selectbox("League", list(DATA_FILES.keys()), index=0)

# chỉ gộp 5 giải này thôi
ALL5_LEAGUES = ["EPL", "LaLiga", "Serie A", "Bundesliga", "Ligue 1", "Greece", "Netherlands", "Portugal", "Scotland", "Turkey", "Belgium"]

if league == "ALL (gộp 5 giải)":
    paths = [DATA_FILES[k] for k in ALL5_LEAGUES]
else:
    paths = [DATA_FILES[league]]

# check file tồn tại
missing = [p for p in paths if not os.path.exists(p)]
if missing:
    st.error("Không thấy file:\n" + "\n".join(missing))
    st.stop()

# load (cache theo từng file)
dfs = []
for p in paths:
    mtime = os.path.getmtime(p)
    dfi = load_df(p, mtime)
    dfi["League"] = league if league != "ALL (gộp 5 giải)" else os.path.splitext(os.path.basename(p))[0]
    dfs.append(dfi)

df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
record_history = load_record_history()


# ---------- Sidebar controls ----------
teams = sorted(pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]], ignore_index=True)))
with st.sidebar:
    st.subheader("Match selector")
    team_a = team_picker("Team A (HOME)", teams, "team_a", default_team=teams[0] if teams else None)
    away_options = [t for t in teams if t != team_a]
    default_away = away_options[0] if away_options else team_a
    team_b = team_picker("Team B (AWAY)", away_options or teams, "team_b", default_team=default_away)
    n = st.slider("Last N matches", 5, 30, 30)

    st.caption("Upcoming match context: **A HOME vs B AWAY**")

    venue_mode = st.checkbox("Venue mode (filter last-N by Home/Away)", value=True)
    h2h_mode_default = st.radio("H2H default", ["All venues", "Strict (A home vs B away)"], index=0, horizontal=True)

    if st.button("Reload data (clear cache)"):
        st.cache_data.clear()
        st.rerun()

# ---------- Data slices ----------

a_all = last_n(df, team_a, n)
b_all = last_n(df, team_b, n)
h2h_all = h2h(df, team_a, team_b, n)

a_venue = last_n_home(df, team_a, n)
b_venue = last_n_away(df, team_b, n)
h2h_strict_df = h2h_strict(df, team_a, team_b, n)
ctx_label = f"{team_a} (HOME) vs {team_b} (AWAY)"

a_last = a_venue if venue_mode else a_all
b_last = b_venue if venue_mode else b_all
h2h_df = h2h_strict_df if h2h_mode_default.startswith("Strict") else h2h_all

# ---------- Top: quick overview ----------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("A samples (last N)", len(a_last))
with c2:
    st.metric("B samples (last N)", len(b_last))
with c3:
    st.metric("H2H samples", len(h2h_df))
with c4:
    if "BTTS" in h2h_df.columns and len(h2h_df):
        st.metric("H2H BTTS%", round(h2h_df["BTTS"].mean() * 100, 1))
    else:
        st.metric("H2H BTTS%", "—")

tabs = st.tabs(["Goals", "Corners", "Cards", "H2H", "Handicap (AH)", "Summary", "Odds API.io"])

# ---------- Goals tab ----------
with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Last {n} — {team_a}")
        show_cols = [c for c in ["Date","HomeTeam","AwayTeam","FTHG","FTAG","HTHG","HTAG","TG","H1G","BTTS"] if c in a_last.columns]
        st.dataframe(a_last[show_cols], use_container_width=True)
    with col2:
        st.subheader(f"Last {n} — {team_b}")
        show_cols = [c for c in ["Date","HomeTeam","AwayTeam","FTHG","FTAG","HTHG","HTAG","TG","H1G","BTTS"] if c in b_last.columns]
        st.dataframe(b_last[show_cols], use_container_width=True)

    st.subheader(f"H2H (latest {n}) — {team_a} vs {team_b}")
    show_cols = [c for c in ["Date","HomeTeam","AwayTeam","FTHG","FTAG","HTHG","HTAG","TG","H1G","BTTS"] if c in h2h_df.columns]
    st.dataframe(h2h_df[show_cols], use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("### FT Goals O/U (H2H)")
        st.dataframe(hit_rate(h2h_df, "TG", GOAL_LINES), use_container_width=True)
    with g2:
        st.markdown("### 1H Goals O/U (H2H)")
        st.dataframe(hit_rate(h2h_df, "H1G", H1_LINES), use_container_width=True)


# ---------- Corners tab ----------
with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Last {n} — {team_a}")
        show_cols = [c for c in ["Date","HomeTeam","AwayTeam","HC","AC","TC"] if c in a_last.columns]
        st.dataframe(a_last[show_cols], use_container_width=True)
    with col2:
        st.subheader(f"Last {n} — {team_b}")
        show_cols = [c for c in ["Date","HomeTeam","AwayTeam","HC","AC","TC"] if c in b_last.columns]
        st.dataframe(b_last[show_cols], use_container_width=True)


    st.divider()
    st.subheader("Corner Handicap (AHCh)")

    if "AHCh" not in df.columns:
        st.info("File này không có cột Corner Asian Handicap (AHCh).")
        a_cah = pd.DataFrame()
        b_cah = pd.DataFrame()
    else:
        st.caption("AHCh là line chấp góc cho Home (âm = Home chấp). Win% tính theo (W + 0.5*HW)/N.")
        a_side = "H"
        b_side = "A"
        a_cah = team_cah_last(df, team_a, n, side_filter=a_side if venue_mode else None)
        b_cah = team_cah_last(df, team_b, n, side_filter=b_side if venue_mode else None)

    c1, c2 = st.columns(2)
    with c1:
        render_ah_panel(team_a, a_cah, show_table=True)
    with c2:
        render_ah_panel(team_b, b_cah, show_table=True)

    st.divider()
    st.subheader(f"H2H Corner Handicap snapshot — {team_a} vs {team_b}")
    cah_cols_pref = ["AHCh", "AvgCAHH", "AvgCAHA", "MaxCAHH", "MaxCAHA", "B365CAHH", "B365CAHA"]
    available_h2h = [c for c in cah_cols_pref if c in h2h_df.columns]
    show_cols = [c for c in (["Date", "HomeTeam", "AwayTeam"] + available_h2h) if c in h2h_df.columns]
    if len(show_cols) > 3:
        st.dataframe(h2h_df[show_cols], use_container_width=True)
    else:
        st.caption("H2H không có snapshot corner handicap odds/line (thiếu cột AHCh/odds).")


# ---------- Cards tab ----------
with tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Last {n} — {team_a}")
        show_cols = [c for c in ["Date","HomeTeam","AwayTeam","HY","AY","HR","AR","TCards"] if c in a_last.columns]
        st.dataframe(a_last[show_cols], use_container_width=True)
    with col2:
        st.subheader(f"Last {n} — {team_b}")
        show_cols = [c for c in ["Date","HomeTeam","AwayTeam","HY","AY","HR","AR","TCards"] if c in b_last.columns]
        st.dataframe(b_last[show_cols], use_container_width=True)


# ---------- Handicap tab ----------
with tabs[3]:
    st.header("H2H (Head-to-Head)")
    st.caption(f"Context: {ctx_label} | Venue mode: {'ON' if venue_mode else 'OFF'}")

    default_idx = 0 if h2h_mode_default == "All venues" else 1
    h2h_view = st.radio("H2H view", ["All venues", "Strict (A home vs B away)"], index=default_idx, horizontal=True)
    h2h_use = h2h_all if h2h_view == "All venues" else h2h_strict_df

    if h2h_use.empty:
        st.info("No H2H matches found for this filter.")
    else:
        show_cols = [c for c in ["Date","HomeTeam","AwayTeam","FTHG","FTAG","HTHG","HTAG","TG","H1G","TC","TCards","BTTS"] if c in h2h_use.columns]
        st.dataframe(h2h_use[show_cols], use_container_width=True)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("### FT Goals O/U (H2H)")
            st.dataframe(hit_rate(h2h_use, "TG", GOAL_LINES), use_container_width=True)
            sum_g, avg_g, meta_g = build_ou_one(h2h_use, "TG", [2, 2.5, 3, 3.5, 4])
            render_ou_cards("FT Goals — H2H Auto pick", sum_g, avg_g, {"samples": meta_g["n"], "weights": "H2H-only"})

        with g2:
            st.markdown("### 1H Goals O/U (H2H)")
            st.dataframe(hit_rate(h2h_use, "H1G", H1_LINES), use_container_width=True)
            sum_h1, avg_h1, meta_h1 = build_ou_one(h2h_use, "H1G", [0.75, 1.0, 1.25, 1.5])
            render_ou_cards("1H Goals — H2H Auto pick", sum_h1, avg_h1, {"samples": meta_h1["n"], "weights": "H2H-only"})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Corners O/U (H2H)")
            st.dataframe(hit_rate(h2h_use, "TC", COR_LINES), use_container_width=True)
            sum_c, avg_c, meta_c = build_ou_one(h2h_use, "TC", COR_LINES)
            render_ou_cards("Corners — H2H Auto pick", sum_c, avg_c, {"samples": meta_c["n"], "weights": "H2H-only"})

        with c2:
            st.markdown("### Cards O/U (H2H)")
            st.dataframe(hit_rate(h2h_use, "TCards", CARD_LINES), use_container_width=True)
            sum_cd, avg_cd, meta_cd = build_ou_one(h2h_use, "TCards", CARD_LINES)
            render_ou_cards("Cards — H2H Auto pick", sum_cd, avg_cd, {"samples": meta_cd["n"], "weights": "H2H-only"})

with tabs[4]:
    st.subheader("Handicap (AH)")

    if "AHh" not in df.columns:
        st.info("File này không có cột Asian Handicap (AHh).")
    else:
        st.subheader("Handicap (Last N) — Team view")
        a_side = "H"
        b_side = "A"
        a_ah = team_ah_last(df, team_a, n, side_filter=a_side if venue_mode else None)
        b_ah = team_ah_last(df, team_b, n, side_filter=b_side if venue_mode else None)

        c1, c2 = st.columns(2)
        with c1:
            render_ah_panel(team_a, a_ah, show_table=True)
        with c2:
            render_ah_panel(team_b, b_ah, show_table=True)

        st.divider()
        st.subheader(f"H2H Handicap snapshot — {team_a} vs {team_b}")
        ah_cols_pref = ["AHh", "AvgAHH", "AvgAHA", "MaxAHH", "MaxAHA", "B365AHH", "B365AHA"]
        available_h2h = [c for c in ah_cols_pref if c in h2h_df.columns]
        show_cols = [c for c in (["Date", "HomeTeam", "AwayTeam"] + available_h2h) if c in h2h_df.columns]
        if len(show_cols) > 3:
            st.dataframe(h2h_df[show_cols], use_container_width=True)
        else:
            st.caption("H2H không có snapshot odds/line (thiếu cột AHh/odds).")

        st.markdown("### Gợi ý đọc nhanh")
        st.write("- **AHh** là line handicap cho Home (âm = Home chấp).")
        if "AvgAHH" in df.columns and "AvgAHA" in df.columns:
            st.write("- **AvgAHH/AvgAHA** là odds trung bình cho Home/Away theo line đó (snapshot/closing-style).")
with tabs[5]:
    st.header("Summary")
    st.caption(
        f"Context: {ctx_label} | Last-N source: {'VENUE' if venue_mode else 'ALL'} | "
        f"H2H default: {h2h_mode_default}"
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        sum_g, avg_g, info_g = build_ou_summary(a_last, b_last, h2h_df, col="TG", lines=[2, 2.5, 3, 3.5, 4])
        sum_h1, avg_h1, info_h1 = build_ou_summary(a_last, b_last, h2h_df, col="H1G", lines=[0.75, 1.0, 1.25, 1.5])
        sum_c, avg_c, info_c = build_ou_summary(a_last, b_last, h2h_df, col="TC", lines=COR_LINES)
        sum_cd, avg_cd, info_cd = build_ou_summary(a_last, b_last, h2h_df, col="TCards", lines=CARD_LINES)
        summary_matched_odds = build_ai_odds_payload(st.session_state.get("latest_odds_api"), team_a, team_b)
        structured_odds = extract_structured_odds_lines(summary_matched_odds)

        best_goals = best_pick_from_summary(sum_g)
        best_h1 = best_pick_from_summary(sum_h1)
        best_corners = best_pick_from_summary(sum_c)
        best_cards = best_pick_from_summary(sum_cd)

        verdict_goals = verdict_from_pick(best_goals, structured_odds.get("goals", {}).get("line"), tolerance=0.5)
        verdict_h1 = verdict_from_pick(best_h1, None, tolerance=0.25)
        verdict_corners = verdict_from_pick(best_corners, None, tolerance=0.5)
        verdict_cards = verdict_from_pick(best_cards, None, tolerance=0.5)

        st.subheader("Quick Verdict")
        v1, v2 = st.columns(2)
        with v1:
            render_market_verdict("FT Goals", best_goals, verdict_goals)
            render_market_verdict("Corners", best_corners, verdict_corners)
        with v2:
            render_market_verdict("1H Goals", best_h1, verdict_h1)
            render_market_verdict("Cards", best_cards, verdict_cards)

        st.divider()
        st.subheader("Auto picks (A + B + H2H blended)")
        render_ou_cards("FT Goals — Auto pick", sum_g, avg_g, info_g)
        render_ou_cards("1H Goals — Auto pick", sum_h1, avg_h1, info_h1)
        render_ou_cards("Corners — Auto pick", sum_c, avg_c, info_c)
        render_ou_cards("Cards — Auto pick", sum_cd, avg_cd, info_cd)

        # BTTS (Both teams to score)
        btts = build_btts_summary(a_last, b_last, h2h_df)
        st.markdown("### BTTS (Both teams score) — Auto pick")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("A BTTS%", "—" if btts["A_BTTS%"] is None else f"{btts['A_BTTS%']}%")
        c2.metric("B BTTS%", "—" if btts["B_BTTS%"] is None else f"{btts['B_BTTS%']}%")
        c3.metric("H2H BTTS%", "—" if btts["H2H_BTTS%"] is None else f"{btts['H2H_BTTS%']}%")
        c4.metric("Blended", "—" if btts["Blended%"] is None else f"{btts['Blended%']}%")
        st.write(f"**Pick:** {btts['Pick']}")

        health_warnings = build_market_health_report(
            team_a=team_a,
            team_b=team_b,
            n=n,
            venue_mode=venue_mode,
            h2h_df=h2h_df,
            matched_odds=summary_matched_odds,
            ou_infos=[
                ("FT Goals", info_g),
                ("1H Goals", info_h1),
                ("Corners", info_c),
                ("Cards", info_cd),
            ],
            btts=btts,
            ah_a=a_ah,
            ah_b=b_ah,
        )
        st.divider()
        st.subheader("Risk Dashboard")
        if health_warnings:
            for msg in health_warnings:
                st.warning(msg)
        else:
            st.success("No major structural warning detected, but this still does not prove betting edge.")

        st.divider()
        st.subheader("Handicap Win% (Last N)")

        if "AHh" not in df.columns:
            st.info("File này chưa có cột AHh (Asian Handicap).")
            a_ah = pd.DataFrame()
            b_ah = pd.DataFrame()
        else:
            a_side = "H"
            b_side = "A"
            a_ah = team_ah_last(df, team_a, n, side_filter=a_side if venue_mode else None)
            b_ah = team_ah_last(df, team_b, n, side_filter=b_side if venue_mode else None)

            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"### {team_a}")
                sa = ah_winrate(a_ah)
                m1, m2, m3 = st.columns(3)
                m1.metric("AH Win%", f"{sa['Win%']}%")
                m2.metric("W-HW-P", f"{sa['W']}-{sa['HW']}-{sa['P']}")
                m3.metric("HL-L", f"{sa['HL']}-{sa['L']}")

            with cb:
                st.markdown(f"### {team_b}")
                sb = ah_winrate(b_ah)
                m1, m2, m3 = st.columns(3)
                m1.metric("AH Win%", f"{sb['Win%']}%")
                m2.metric("W-HW-P", f"{sb['W']}-{sb['HW']}-{sb['P']}")
                m3.metric("HL-L", f"{sb['HL']}-{sb['L']}")


    st.divider()
    st.subheader("Corner Handicap Win% (Last N)")

    if "AHCh" not in df.columns:
        st.info("File này chưa có cột AHCh (Corner Asian Handicap).")
        a_cah = pd.DataFrame()
        b_cah = pd.DataFrame()
    else:
        a_side = "H"
        b_side = "A"
        a_cah = team_cah_last(df, team_a, n, side_filter=a_side if venue_mode else None)
        b_cah = team_cah_last(df, team_b, n, side_filter=b_side if venue_mode else None)

    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"### {team_a}")
        sa = ah_winrate(a_cah)
        m1, m2, m3 = st.columns(3)
        m1.metric("Corner AH Win%", f"{sa['Win%']}%")
        m2.metric("W-HW-P", f"{sa['W']}-{sa['HW']}-{sa['P']}")
        m3.metric("HL-L", f"{sa['HL']}-{sa['L']}")
    with cb:
        st.markdown(f"### {team_b}")
        sb = ah_winrate(b_cah)
        m1, m2, m3 = st.columns(3)
        m1.metric("Corner AH Win%", f"{sb['Win%']}%")
        m2.metric("W-HW-P", f"{sb['W']}-{sb['HW']}-{sb['P']}")
        m3.metric("HL-L", f"{sb['HL']}-{sb['L']}")

    with st.expander("Corner AH (Last N) tables"):
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(a_cah, use_container_width=True, hide_index=True)
        with c2:
            st.dataframe(b_cah, use_container_width=True, hide_index=True)


    with right:
        st.subheader("AI Advisor (API)")
        st.caption("")

        provider = st.selectbox("Provider", ["Gemini", "OpenAI"], index=0, key="ai_provider")

        if provider == "Gemini":
            api_url = AI_API_URL
            gemini_key_choice = st.selectbox("Gemini key", ["Key 1", "Key 2"], index=0, key="gemini_key_choice")
            api_key = GEMINI_API_KEY if gemini_key_choice == "Key 1" else GEMINI_API_KEY_2
            openai_model = None
            active_model_name = "Gemini 3 Flash"
            st.caption(f"Using Gemini model: {GEMINI_MODEL} ({GEMINI_API_VERSION}) | {gemini_key_choice}")
            if not api_key:
                st.warning("Missing selected Gemini key. Paste it at the top of appbackup.py before running.")
        else:
            api_url = ""
            api_key = OPENAI_API_KEY
            openai_model = OPENAI_DEFAULT_MODEL
            active_model_name = "GPT5.4"
            st.caption(f"Using OpenAI model: {openai_model}")
            if not api_key:
                st.warning("Missing OPENAI_API_KEY. Set env var OPENAI_API_KEY before running.")

        latest_odds = st.session_state.get("latest_odds_api")

        matched_odds = build_ai_odds_payload(latest_odds, team_a, team_b)

        # EPL 9/10 ENGINE INJECTION
        try:
            import json
            from metrics_epl import extract_epl_metrics
            from poisson import predict_poisson_market_probs, filter_main_odds_lines
            from circuit_breaker import check_high_blowout_risk
            
            # Extract metrics for Poisson (fallback to Proxy xG)
            home_stats = extract_epl_metrics(df, team_a, lookback=10)
            away_stats = extract_epl_metrics(df, team_b, lookback=10)
            
            # Calculate probabilities
            poisson_ev_funcs = predict_poisson_market_probs(home_stats, away_stats)
            is_blowout = check_high_blowout_risk(home_stats, away_stats)
            
            # Find main line and filter EV
            matched_event_tmp = (matched_odds or {}).get("event") or {}
            normalized_books_tmp = _normalize_bookmakers_payload(matched_event_tmp) if matched_event_tmp else []
            tot_out, spr_out, btts_out, poisson_picks = filter_main_odds_lines(normalized_books_tmp, poisson_ev_funcs, is_blowout)
            
            if matched_odds is None:
                matched_odds = {}
            matched_odds["poisson_picks"] = poisson_picks
            matched_odds["poisson_picks"]["is_blowout_risk"] = is_blowout
            
        except Exception as e:
            st.error(f"Engine 9/10 Integration Error: {e}")

        input_lines = extract_input_lines_from_matched_odds(matched_odds)
        odds_prompt_block = build_prompt_odds_block(matched_odds, input_lines)
        raw_market_summary = build_raw_market_summary(matched_odds)
        matched_event = (matched_odds or {}).get("event") or {}
        normalized_books = _normalize_bookmakers_payload(matched_event) if matched_event else []
        if latest_odds and isinstance(latest_odds, dict):
            if latest_odds.get("provider") == "odds-api-io":
                if matched_odds and matched_odds.get("matched"):
                    event = matched_odds.get("event") or {}
                    st.caption(
                        f"Odds ready for AI: matched {event.get('home', '?')} vs {event.get('away', '?')} via odds-api.io"
                    )
                    if normalized_books:
                        market_count = sum(len(book.get("markets") or []) for book in normalized_books)
                        st.caption(f"Detected odds payload: {len(normalized_books)} bookmaker block(s), {market_count} market block(s).")
                    else:
                        st.warning("Matched odds snapshot exists, but no bookmaker/market blocks were detected in the raw event payload.")
                else:
                    st.caption(
                        f"odds-api.io fetched but no exact event match was stored for {team_a} vs {team_b}"
                    )
            else:
                odds_count = len(latest_odds.get("odds") or [])
                if matched_odds and matched_odds.get("matched"):
                    event = matched_odds.get("event") or {}
                    st.caption(
                        f"Odds ready for AI: matched {event.get('home_team', '?')} vs {event.get('away_team', '?')} from {odds_count} fetched matches"
                    )
                    filled = [k for k, v in input_lines.items() if v]
                    if filled:
                        st.caption(f"Auto-filled INPUT LINES from odds: {', '.join(filled)}")
                else:
                    st.caption(
                        f"Odds fetched ({latest_odds.get('sport_key', '')} | {odds_count} matches) but no exact match found for {team_a} vs {team_b}"
                    )
        else:
            st.caption("Odds fetch not loaded yet. GPT/Gemini will only use dataset + prompt until you fetch odds in the Odds API tab.")

        if False:
            overall = record_bias.get("overall") or {}
            league_summary = record_bias.get("league_summary") or {}
            model_summary = record_bias.get("model_summary") or {}
            st.caption(f"Record source: {record_history.get('source', record_bias.get('record_path', ''))}")
            c1, c2 = st.columns(2)
            c1.metric("Total Bets", overall.get("bets", 0))
            c2.metric("Overall Avg Score", overall.get("avg_score", "—"))
            if league_summary:
                st.caption(
                    f"League bias for {league_summary.get('league')}: bets={league_summary.get('bets', 0)} | avg_score={league_summary.get('avg_score', '—')}"
                )
            if model_summary:
                st.caption(
                    f"Model bias for {model_summary.get('model')}: used={model_summary.get('used_total', 0)} | total P/L={model_summary.get('total_p_l', '—')}"
                )
            top_markets = (league_summary.get("markets") or [])[:3]
            if top_markets:
                st.write("Top league market buckets:")
                st.dataframe(pd.DataFrame(top_markets), use_container_width=True, hide_index=True)
            if record_history.get("details"):
                with st.expander("Record load notes"):
                    for msg in record_history.get("details") or []:
                        st.write(msg)
        elif False:
            pass
        else:
            pass

        provider_extra_rules = ""
        if provider == "OpenAI":
            provider_extra_rules = (
                "- OPENAI UNLOCK: Treat the matched odds snapshot, RAW MARKET SUMMARY, and any visible bookmaker outcome prices in the raw event object as valid betting inputs even if INPUT LINES are incomplete.\n"
                "- OPENAI UNLOCK: If you can identify a clear bookmaker market with line and side prices from the raw matched event payload, use it directly instead of replying that odds are missing.\n"
                "- OPENAI UNLOCK: If the odds format is not explicitly labeled HK but the prices are clearly bookmaker side prices from the matched event, you may still use them for selection.\n"
                "- OPENAI UNLOCK: Prefer making one best-effort pick from visible matched-event bookmaker data rather than defaulting to NO BET for formatting reasons alone.\n"
            )

        default_prompt = (
            f"Match: {ctx_label}\\n"
            f"\\n"
            f"DATA AVAILABLE: dataset stats plus fetched odds snapshot from TheOddsAPI when available in session.\\n"
            f"\\n"
            f"{odds_prompt_block}\\n"
            f"\\n"
            f"{('RAW MARKET SUMMARY:\\n' + raw_market_summary + '\\n\\n') if raw_market_summary else ''}"
            f"INPUT LINES (fill in; use these lines first):\\n"
            f"- {(input_lines.get('goals') or 'Goals:  | Over  | Under')}\\n"
            f"- 1H Goals:  | Over  | Under \\n"
            f"- Corners:  | Over  | Under \\n"
            f"- Corner Handicap (home):  | Home  | Away \\n"
            f"- Cards:  | Over  | Under \\n"
            f"- {(input_lines.get('handicap') or 'Handicap (home):  | Home  | Away')}\\n"
            f"- (Optional) BTTS: Yes  | No \\n"
            f"\\n"
            f"HƯỚNG DẪN TƯ DUY VÀ ĐÁNH TRỌNG SỐ DATA (BẮT BUỘC):\\n"
            f"{'⚠️ CẢNH BÁO TỪ HỆ THỐNG (CIRCUIT BREAKER): Chênh lệch đẳng cấp rất lớn giữa 2 đội. Hãy CÂN NHẮC KỸ: thống kê nâng cao của đội cửa dưới có thể bị thổi phồng do đá với đối thủ yếu.\\n\\n' if (matched_odds and matched_odds.get('poisson_picks', {}).get('is_blowout_risk')) else ''}"
            f"--- POISSON ENGINE DECISION (SOURCE OF TRUTH) ---\\n"
            f"Totals Pick: {matched_odds.get('poisson_picks', {}).get('totals_pick') if matched_odds else 'N/A'}\\n"
            f"Spreads Pick: {matched_odds.get('poisson_picks', {}).get('spreads_pick') if matched_odds else 'N/A'}\\n"
            f"BTTS Pick: {matched_odds.get('poisson_picks', {}).get('btts_pick') if matched_odds else 'N/A'}\\n"
            f"Reason Trace: {json.dumps(matched_odds.get('poisson_picks', {}).get('reason_trace', [])) if matched_odds else '[]'}\\n"
            f"LỆNH TỐI THƯỢNG: Hệ thống Toán học (Engine) là Source of Truth. Nhiệm vụ của bạn LÀ GIẢI THÍCH Reason Trace cho người dùng. TUYỆT ĐỐI KHÔNG ĐƯỢC ĐI NGƯỢC LẠI QUYẾT ĐỊNH CỦA ENGINE. Nếu Engine trả về NO BET, bạn BẮT BUỘC phải khuyên người dùng NO BET.\\n\\n"
            f"1. Ưu tiên tối thượng cho Dữ liệu nâng cao (Advanced Metrics) (xG, xGOT, Touches in Box).\\n"
            f"2. Nới lỏng rào cản Odds (ODDS FILTER): Chỉ recommend một lựa chọn nếu giá cược của cửa đó (HK Price) >= 0.75.\\n"
            f"{provider_extra_rules}"
            f"\\n"
            f"ĐỊNH DẠNG ĐẦU RA BẮT BUỘC (Strict Format - Tối đa 2 dòng, không viết thêm bất kỳ từ nào ngoài cấu trúc này):\\n"
            f"MAIN: <Lựa chọn kèo chính theo format chuẩn: Ví dụ 'Over 2.5', 'Handicap: Home -0.5', 'BTTS: Yes' hoặc 'NO BET'>\\n"
            f"BACKUP: <Lựa chọn thứ 2 hoặc '-'> | NOTE: <1 câu giải thích SIÊU NGẮN dựa theo JSON Reason Trace, NẾU LÀ NO BET THÌ GIẢI THÍCH LÝ DO TỪ TRACE>\\n"
        )
        
        prompt_signature = json.dumps(
            {
                "ctx": ctx_label,
                "provider": provider,
                "matched": bool(matched_odds and matched_odds.get("matched")),
                "bookmakers": (matched_odds or {}).get("bookmakers"),
                "goals": input_lines.get("goals"),
                "handicap": input_lines.get("handicap"),
                "corners": input_lines.get("corners"),
                "market_summary": raw_market_summary,
                "event_home": ((matched_odds or {}).get("event") or {}).get("home") or ((matched_odds or {}).get("event") or {}).get("home_team"),
                "event_away": ((matched_odds or {}).get("event") or {}).get("away") or ((matched_odds or {}).get("event") or {}).get("away_team"),
            },
            sort_keys=True,
            default=str,
        )

        # Auto-update prompt when match or odds snapshot changes
        if st.session_state.get('_ai_prompt_sig') != prompt_signature:
            st.session_state['ai_prompt'] = default_prompt
            st.session_state['_ai_prompt_sig'] = prompt_signature
        prompt = st.text_area("Prompt", height=180, key="ai_prompt")

        if st.button("Run AI", key="ai_run"):
            market_snapshot = {}
            if not h2h_df.empty and "Date" in h2h_df.columns:
                latest = h2h_df.sort_values("Date", ascending=False).iloc[0]
                for k in [
                    "Date", "HomeTeam", "AwayTeam", "AHh", "AvgAHH", "B365AHH", "PAHH",
                    "Avg>2.5", "B365>2.5", "P>2.5", "Avg<2.5", "B365<2.5", "P<2.5",
                ]:
                    if k in h2h_df.columns:
                        market_snapshot[k] = latest.get(k)

            payload = {
                "prompt": prompt,
                "context": ctx_label,
                "last_n": n,
                "venue_mode": venue_mode,
                "h2h_mode": h2h_mode_default,
                "tables": {
                    "ft_goals": sum_g.to_dict(orient="records"),
                    "h1_goals": sum_h1.to_dict(orient="records"),
                    "corners": sum_c.to_dict(orient="records"),
                    "cards": sum_cd.to_dict(orient="records"),
                    "btts": btts,
                    "corner_ah_a_last": a_cah.to_dict(orient="records") if isinstance(a_cah, pd.DataFrame) else [],
                    "corner_ah_b_last": b_cah.to_dict(orient="records") if isinstance(b_cah, pd.DataFrame) else [],
                },
                "handicap": {
                    "team_a": ah_winrate(a_ah),
                    "team_b": ah_winrate(b_ah),
                    "corner_team_a": ah_winrate(a_cah),
                    "corner_team_b": ah_winrate(b_cah),
                },
                "market_snapshot": market_snapshot,
                "odds_api_snapshot": matched_odds,
                "raw_market_summary": raw_market_summary,
            }
            if provider == "Gemini":
                st.session_state["ai_result"] = call_ai_api(api_url, api_key, payload)
            else:
                payload["reasoning_effort"] = "medium"
                payload["text_verbosity"] = "medium"
                payload["max_output_tokens"] = 4000
                st.session_state["ai_result"] = call_openai_api(openai_model, api_key, payload)

        if "ai_result" in st.session_state and st.session_state["ai_result"]:
            st.markdown("### AI result")
            res = st.session_state["ai_result"]
            if isinstance(res, dict):
                if not res.get("ok", True):
                    status = res.get("status")
                    err = res.get("error")
                    raw_text = res.get("text") or ""

                    if err:
                        st.error(err)
                    elif status:
                        st.error(f"API error (HTTP {status})")
                    else:
                        st.error("API error")

                    # Try to extract a clean message from Gemini error JSON
                    msg = ""
                    try:
                        import json as _json
                        j = _json.loads(raw_text) if isinstance(raw_text, str) and raw_text.strip() else None
                        if isinstance(j, dict):
                            msg = (j.get("error") or {}).get("message") or ""
                    except Exception:
                        pass

                    if msg:
                        st.write(msg)

                    with st.expander("Error details"):
                        if raw_text:
                            st.code(raw_text[:6000])
                        else:
                            st.json(res)

                elif "text" in res and isinstance(res["text"], str):
                    st.markdown(res["text"])
                else:
                    st.write(res)
            else:
                st.write(res)

with tabs[6]:
    st.header("Odds API (odds-api.io)")
    st.markdown(
        "Use odds-api.io to fetch the current event odds for the match you selected. The matched event is kept only in this Streamlit session so AI can read it."
    )

    odds_api_key = st.text_input(
        "odds-api.io key",
        type="password",
        help="Get one at https://odds-api.io",
        key="odds_api_key",
    )
    sport_key = st.text_input(
        "Sport slug",
        "football",
        help="Example: football, basketball",
        key="odds_sport_key",
    )
    bookmaker_options = ["Bet365"]
    selected_bookmakers = st.multiselect(
        "Bookmakers",
        bookmaker_options,
        default=["Bet365"],
        help="Matched event odds will use Bet365.",
        key="odds_bookmakers_multi",
    )
    if not selected_bookmakers:
        selected_bookmakers = ["Bet365"]
    bookmaker_api_map = {
        "Bet365": "Bet365",
    }
    bookmaker_name = ",".join(bookmaker_api_map.get(x, x) for x in selected_bookmakers)
    st.caption(f"Bookmakers request: {', '.join(selected_bookmakers)}")
    curated_leagues = []
    if league == "ALL (gộp 5 giải)":
        for k in ALL5_LEAGUES:
            slug = ODDS_API_IO_LEAGUE_SLUGS.get(k)
            if slug:
                curated_leagues.append({"label": k, "slug": slug})
    else:
        slug = ODDS_API_IO_LEAGUE_SLUGS.get(league)
        if slug:
            curated_leagues.append({"label": league, "slug": slug})

    extra_leagues = [
        {"label": k, "slug": v}
        for k, v in ODDS_API_IO_LEAGUE_SLUGS.items()
        if {"label": k, "slug": v} not in curated_leagues
    ]
    league_options = curated_leagues + extra_leagues
    league_labels = ["All supported CSV leagues"] + [x["label"] for x in league_options]
    default_league_index = 0
    if league != "ALL (gộp 5 giải)":
        try:
            default_league_index = 1 + next(i for i, x in enumerate(league_options) if x["label"] == league)
        except StopIteration:
            default_league_index = 0
    selected_league_label = st.selectbox(
        "League",
        league_labels,
        index=default_league_index,
        key="odds_league_label",
        help="Only leagues mapped from your CSV dataset are shown here.",
    )
    if selected_league_label == "All supported CSV leagues":
        league_slug = ""
    else:
        selected = next((x for x in league_options if x["label"] == selected_league_label), None)
        league_slug = (selected or {}).get("slug", "")
    if league_slug:
        st.caption(f"League slug: {league_slug}")
    force_refresh = st.checkbox("Force refresh (ignore cache)", value=False, key="odds_force_refresh")
    days_ahead = st.slider("Search events within next days", 1, 30, 7, key="odds_days_ahead")

    fetch_scope_league = ""
    cache_key = f"{sport_key}|{fetch_scope_league}|{days_ahead}"
    if "odds_cache" not in st.session_state:
        st.session_state["odds_cache"] = {}
    if "odds_event_cache" not in st.session_state:
        st.session_state["odds_event_cache"] = {}

    current_league_marker = f"{sport_key}|{league_slug}|{days_ahead}"
    current_match_marker = f"{current_league_marker}|{bookmaker_name}|{team_a}|{team_b}"
    previous_league_marker = st.session_state.get("odds_active_league_marker")
    if previous_league_marker and previous_league_marker != current_league_marker:
        st.session_state.pop("latest_odds_api", None)
        st.session_state.pop("latest_odds_lock_marker", None)
    st.session_state["odds_active_league_marker"] = current_league_marker

    if st.button("Fetch odds", key="odds_fetch"):
        if not force_refresh and cache_key in st.session_state["odds_cache"]:
            res = st.session_state["odds_cache"][cache_key]
            st.info("Using cached odds data (no credit used).")
        else:
            res = fetch_odds_api_io_events(
                api_key=odds_api_key,
                sport=sport_key,
                league=None,
                days_ahead=days_ahead,
            )
            if res.get("ok"):
                st.session_state["odds_cache"][cache_key] = res

        if not res.get("ok"):
            st.session_state.pop("latest_odds_api", None)
            st.error(res.get("error") or "Unknown error")
        else:
            raw = res.get("raw")
            if league_slug:
                raw = filter_odds_api_io_events_by_league(
                    raw,
                    league_slug=league_slug,
                    league_label=selected_league_label,
                )
            if not raw:
                st.session_state.pop("latest_odds_api", None)
                st.warning(
                    f"No events found for league '{selected_league_label}' in the next {days_ahead} days."
                )
            else:
                matched_event = None
                team_a_norm = _normalize_team_name(team_a)
                team_b_norm = _normalize_team_name(team_b)
                for item in raw:
                    home_name = item.get("home", "")
                    away_name = item.get("away", "")
                    if _team_names_match(home_name, team_a_norm) and _team_names_match(away_name, team_b_norm):
                        matched_event = item
                        break

                if not matched_event:
                    st.session_state["latest_odds_api"] = {
                        "provider": "odds-api-io",
                        "sport_key": sport_key,
                        "league": league_slug,
                        "bookmakers": bookmaker_name,
                        "events": raw,
                        "event": None,
                        "match_error": f"No event match found for {team_a} vs {team_b}.",
                    }
                    st.warning(f"Fetched {len(raw)} events but could not match {team_a} vs {team_b}.")
                    st.json(raw[:10] if isinstance(raw, list) else raw)
                else:
                    event_id = matched_event.get("id")
                    odds_cache_key = f"{event_id}|{bookmaker_name}"
                    if not force_refresh and odds_cache_key in st.session_state["odds_event_cache"]:
                        odds_res = st.session_state["odds_event_cache"][odds_cache_key]
                        st.info("Using cached event odds (no extra credit used).")
                    else:
                        odds_res = fetch_odds_api_io_event_odds(odds_api_key, event_id, bookmaker_name)
                        if odds_res.get("ok"):
                            st.session_state["odds_event_cache"][odds_cache_key] = odds_res
                    if not odds_res.get("ok"):
                        st.session_state.pop("latest_odds_api", None)
                        st.error(odds_res.get("error") or "Failed to fetch event odds.")
                        st.info(
                            f"Matched event '{matched_event.get('home')} vs {matched_event.get('away')}', "
                            f"but {bookmaker_name} may not have odds for it right now."
                        )
                    else:
                        st.session_state["latest_odds_api"] = {
                            "provider": "odds-api-io",
                            "sport_key": sport_key,
                            "league": league_slug,
                            "bookmakers": bookmaker_name,
                            "events": raw,
                            "event": odds_res.get("raw"),
                            "requested_home": team_a,
                            "requested_away": team_b,
                            "event_id": event_id,
                            "snapshot_source": "fresh_fetch",
                        }
                        st.session_state["latest_odds_lock_marker"] = current_match_marker
                        st.success(f"Matched current selection and fetched odds-api.io event odds for event {event_id}.")
                        st.json(odds_res.get("raw"))

    if league_slug and cache_key in st.session_state.get("odds_cache", {}):
        cached_res = st.session_state["odds_cache"].get(cache_key) or {}
        cached_events = cached_res.get("raw") if cached_res.get("ok") else None
        cached_events = filter_odds_api_io_events_by_league(
            cached_events,
            league_slug=league_slug,
            league_label=selected_league_label,
        )
        auto_match = find_matching_odds_api_io_event(cached_events, team_a, team_b)
        current_snapshot = st.session_state.get("latest_odds_api")
        locked_match_marker = st.session_state.get("latest_odds_lock_marker")
        current_event = (current_snapshot or {}).get("event") if isinstance(current_snapshot, dict) else None
        current_home = (current_event or {}).get("home_team") or (current_event or {}).get("home") or ""
        current_away = (current_event or {}).get("away_team") or (current_event or {}).get("away") or ""
        current_is_same_match = bool(current_event) and _team_names_match(current_home, team_a) and _team_names_match(current_away, team_b)
        current_has_usable_markets = _event_has_usable_markets(current_event)

        if auto_match and not (locked_match_marker == current_match_marker and current_has_usable_markets) and not current_is_same_match:
            auto_event_id = auto_match.get("id")
            auto_odds_key = f"{auto_event_id}|{bookmaker_name}"
            cached_event_odds = st.session_state.get("odds_event_cache", {}).get(auto_odds_key)
            st.session_state["latest_odds_api"] = {
                "provider": "odds-api-io",
                "sport_key": sport_key,
                "league": league_slug,
                "bookmakers": bookmaker_name,
                "events": cached_events,
                "event": (cached_event_odds or {}).get("raw") if isinstance(cached_event_odds, dict) and cached_event_odds.get("ok") else None,
                "requested_home": team_a,
                "requested_away": team_b,
                "event_id": auto_event_id,
                "snapshot_source": "cache_rematch",
                "match_error": None if (isinstance(cached_event_odds, dict) and cached_event_odds.get("ok")) else f"Matched event {team_a} vs {team_b} from cached league events, but odds for {', '.join(selected_bookmakers)} have not been fetched yet.",
            }

    latest_odds = st.session_state.get("latest_odds_api")
    if latest_odds:

        matched_odds = build_ai_odds_payload(latest_odds, team_a, team_b)

        # EPL 9/10 ENGINE INJECTION
        try:
            import json
            from metrics_epl import extract_epl_metrics
            from poisson import predict_poisson_market_probs, filter_main_odds_lines
            from circuit_breaker import check_high_blowout_risk
            
            # Extract metrics for Poisson (fallback to Proxy xG)
            home_stats = extract_epl_metrics(df, team_a, lookback=10)
            away_stats = extract_epl_metrics(df, team_b, lookback=10)
            
            # Calculate probabilities
            poisson_ev_funcs = predict_poisson_market_probs(home_stats, away_stats)
            is_blowout = check_high_blowout_risk(home_stats, away_stats)
            
            # Find main line and filter EV
            matched_event_tmp = (matched_odds or {}).get("event") or {}
            normalized_books_tmp = _normalize_bookmakers_payload(matched_event_tmp) if matched_event_tmp else []
            tot_out, spr_out, btts_out, poisson_picks = filter_main_odds_lines(normalized_books_tmp, poisson_ev_funcs, is_blowout)
            
            if matched_odds is None:
                matched_odds = {}
            matched_odds["poisson_picks"] = poisson_picks
            matched_odds["poisson_picks"]["is_blowout_risk"] = is_blowout
            
        except Exception as e:
            st.error(f"Engine 9/10 Integration Error: {e}")

        if latest_odds.get("provider") == "odds-api-io":
            st.caption(
                f"Current AI odds snapshot: odds-api.io | sport={latest_odds.get('sport_key', '')} | bookmaker(s)={latest_odds.get('bookmakers', '')} | source={latest_odds.get('snapshot_source', 'unknown')}"
            )
        else:
            st.caption(
                f"Current AI odds snapshot: {latest_odds.get('sport_key', '')} | {len(latest_odds.get('odds') or [])} matches"
            )
        if matched_odds and matched_odds.get("matched"):
            event = matched_odds.get("event") or {}
            st.success(f"Matched current selection: {event.get('home_team', event.get('home', '?'))} vs {event.get('away_team', event.get('away', '?'))}")
            with st.expander("Matched odds event for current match"):
                st.json(event)
            with st.expander("Parsed market summary for current match"):
                if raw_market_summary:
                    st.code(raw_market_summary)
                else:
                    st.write("No parsed market summary could be built from the matched raw event payload.")
        elif matched_odds:
            st.warning(matched_odds.get("match_error") or "No matching odds event found for the current match.")
