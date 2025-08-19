#!/usr/bin/env python3
"""
Combined Steam export + HLTB enrichment for modeling (NO Hours_Played in outputs).

Outputs:
- steam_library_for_model.csv      (raw Steam metadata; type == game only; no playtime)
- steam_library_with_hltb.csv      (enriched with HLTB Hours_Main_Story + metadata backfill)

Final columns:
- Name
- Hours_Main_Story
- Genre1
- Genre2
- Year Released
- Developer
- Publisher
- AppID

CLI:
    python steam_export_and_enrich.py --hltb hltb.csv [--overrides overrides.csv] [--cutoff 80] [--include-non-games]

Requirements:
    pip install requests python-dateutil pandas rapidfuzz python-dotenv

Env (can be provided via .env or flags):
    STEAM_API_KEY, STEAM_ID64
"""

import os
import re
import csv
import time
import math
import argparse
import requests
import pandas as pd
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dateutil import parser as dateparser
from rapidfuzz import process, fuzz

# --- Steam API endpoints ---
OWNED_GAMES_URL      = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
APPDETAILS_URL       = "https://store.steampowered.com/api/appdetails"
RESOLVE_VANITY_URL   = "https://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/"

# --- Output filenames ---
RAW_OUTPUT   = "steam_library_for_model.csv"
FINAL_OUTPUT = "steam_library_with_hltb.csv"

# --- Networking tunables ---
REQUEST_TIMEOUT   = 20          # seconds per HTTP request
RETRY_BACKOFF     = [1, 2, 4]   # backoff schedule for retries
RATE_LIMIT_SLEEP  = 0.35        # seconds between appdetails calls (politeness / throttle safety)

# Load .env (if present) so users can avoid passing flags every time
load_dotenv()
STEAM_API_KEY = os.getenv("STEAM_API_KEY")
STEAM_ID64    = os.getenv("STEAM_ID64")

# ---------------------------
# Helpers: SteamID / Vanity
# ---------------------------
def looks_like_steamid64(value: str) -> bool:
    """Return True if the string is exactly a 17-digit SteamID64."""
    return bool(re.fullmatch(r"\d{17}", value.strip()))

def resolve_vanity_to_steamid64(api_key: str, vanity: str) -> str:
    """
    Convert a vanity name (or full vanity URL) to SteamID64 via ResolveVanityURL.

    Accepts:
      - "CheesyRatPan"
      - "https://steamcommunity.com/id/CheesyRatPan/"
      - "https://steamcommunity.com/profiles/7656119XXXXXXXXXX" (returns same 64-bit ID)

    Raises ValueError if resolution fails.
    """
    # Extract only the vanity part if a full URL was provided
    m = re.search(r"steamcommunity\.com/(?:id|profiles)/([^/]+)/?", vanity, re.IGNORECASE)
    if m:
        vanity = m.group(1)

    params = {"key": api_key, "vanityurl": vanity}
    r = requests.get(RESOLVE_VANITY_URL, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json().get("response", {})
    if data.get("success") == 1 and "steamid" in data:
        return data["steamid"]
    raise ValueError(f"Could not resolve vanity '{vanity}' to SteamID64 (response: {data})")

# ---------------------------
# Helpers: Steam export
# ---------------------------
def get_owned_games(api_key: str, steamid: str) -> List[Dict[str, Any]]:
    """
    Retrieve the list of owned apps (games + others) for a given SteamID64.

    Notes:
      - Requires 'Game details' privacy to be set to Public on the user's Steam profile.
      - Returns dictionaries with keys like: appid, name, playtime_forever, etc.
      - We only *keep* 'game' types later when enriching with appdetails.
    """
    params = {
        "key": api_key,
        "steamid": steamid,
        "include_appinfo": 1,
        "include_played_free_games": 1,
        "format": "json",
    }
    for attempt, backoff in enumerate([0] + RETRY_BACKOFF):
        try:
            if backoff:
                time.sleep(backoff)
            r = requests.get(OWNED_GAMES_URL, params=params, timeout=REQUEST_TIMEOUT)
            try:
                r.raise_for_status()
            except requests.HTTPError as err:
                # Common 400 causes to help users recover quickly:
                detail = (
                    f"HTTP {r.status_code} from GetOwnedGames. Check that:\n"
                    f"  • SteamID is 17-digit (SteamID64), not a vanity/URL\n"
                    f"  • Steam profile → Privacy → 'Game details' is Public\n"
                    f"  • API key is valid: https://steamcommunity.com/dev/apikey"
                )
                raise requests.HTTPError(detail) from err

            data = r.json()
            return data.get("response", {}).get("games", []) or []
        except Exception:
            if attempt == len(RETRY_BACKOFF):
                raise
    return []

def get_app_details(appid: int) -> Optional[Dict[str, Any]]:
    """
    Query Steam Store 'appdetails' for a single app ID.

    We need this to:
      - confirm 'type' == 'game' (otherwise skip)
      - get Year (from release_date)
      - get Developer/Publisher arrays
      - get Genres
    """
    params = {"appids": appid}
    for attempt, backoff in enumerate([0] + RETRY_BACKOFF):
        try:
            if backoff:
                time.sleep(backoff)
            r = requests.get(APPDETAILS_URL, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            payload = r.json()
            entry = payload.get(str(appid))
            if not entry or not entry.get("success"):
                return None
            return entry.get("data")
        except Exception:
            if attempt == len(RETRY_BACKOFF):
                return None
    return None

def parse_year(release_date: Dict[str, Any]) -> Optional[int]:
    """Extract a 4-digit year from the Steam release_date blob; robust to odd formats."""
    if not release_date:
        return None
    date_str = release_date.get("date")
    if not date_str:
        return None
    try:
        dt = dateparser.parse(date_str, default=datetime(1900,1,1))
        return dt.year
    except Exception:
        m = re.search(r"(19|20)\d{2}", date_str)
        return int(m.group(0)) if m else None

def first_two_genres(genres: Any) -> Tuple[str, str]:
    """Return first two distinct genre descriptions from Steam appdetails."""
    if isinstance(genres, list) and genres:
        desc = [g.get("description") for g in genres if isinstance(g, dict) and g.get("description")]
        seen, out = set(), []
        for d in desc:
            if d not in seen:
                out.append(d)
                seen.add(d)
        g1 = out[0] if len(out) >= 1 else ""
        g2 = out[1] if len(out) >= 2 else ""
        return g1, g2
    return "", ""

def is_game_type(appdata: Dict[str, Any]) -> bool:
    """True iff appdetails says this record's 'type' is 'game' (not DLC/demo/tool/soundtrack/etc.)."""
    return (appdata or {}).get("type", "") == "game"

# ---------------------------
# Helpers: HLTB enrichment
# ---------------------------
EDITION_WORDS = [
    r"\bdefinitive edition\b", r"\bremastered\b", r"\bremaster\b", r"\bgame of the year\b",
    r"\bgoty\b", r"\bcomplete\b", r"\bultimate\b", r"\bhd\b", r"\bcollection\b",
    r"\banniversary\b", r"\bredux\b", r"\bre-elected\b", r"\bthe definitive edition\b",
]
EDITION_RE = re.compile("|".join(EDITION_WORDS), re.IGNORECASE)

def normalize_title(s: str) -> str:
    """
    Normalize game titles to improve fuzzy match reliability:
      - remove edition words and bracketed clauses
      - strip punctuation
      - normalize whitespace
      - lightly map roman numerals (iii->3, ii->2, iv->4, vi->6)
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = EDITION_RE.sub("", s)
    s = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", s)
    s = re.sub(r"[^a-z0-9\s:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\biii\b", "3", s)
    s = re.sub(r"\bii\b",  "2", s)
    s = re.sub(r"\biv\b",  "4", s)
    s = re.sub(r"\bvi\b",  "6", s)
    s = s.replace(" & ", " and ")
    return s

def coerce_hours(val) -> Optional[float]:
    """
    Convert HLTB-like hour fields to a float.
    Accepts "12", "12.5", "12-14", "12–14", "12 hours", etc. → returns first numeric.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    txt = str(val).strip().lower().replace("–","-")
    if not txt:
        return None
    m = re.search(r"(\d+(\.\d+)?)", txt)
    return float(m.group(1)) if m else None

def load_overrides_csv(path: Optional[str]) -> Dict[str, str]:
    """
    Load manual title overrides for perfect matches when fuzzy is tricky.
    CSV columns must be: Steam_Name, HLTB_Name
    (We normalize both sides so spelling/case/edition text won't matter.)
    """
    mapping: Dict[str, str] = {}
    if not path:
        return mapping
    if not os.path.exists(path):
        raise SystemExit(f"Overrides file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "Steam_Name" not in reader.fieldnames or "HLTB_Name" not in reader.fieldnames:
            raise SystemExit("Overrides CSV must have headers: Steam_Name, HLTB_Name")
        for row in reader:
            left, right = row.get("Steam_Name",""), row.get("HLTB_Name","")
            if left and right:
                mapping[normalize_title(left)] = normalize_title(right)
    return mapping

# ---------------------------
# Export Steam library
# ---------------------------
def export_steam(api_key: str, steamid: str, include_non_games: bool=False) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """
    Export only 'game'-type apps with metadata sufficient for modeling.

    Returns:
      - DataFrame with columns [Name, Hours_Main_Story, Genre1, Genre2, Year Released, Developer, Publisher, AppID]
      - stats dict summarizing filtering (how many skipped and why)
    """
    owned = get_owned_games(api_key, steamid)
    stats = {
        "owned_total": len(owned),       # total records returned by GetOwnedGames (games+others)
        "skipped_missing_appdetails": 0, # appdetails failed (can't classify / no metadata)
        "skipped_non_game_type": 0,      # DLC/demo/tool/soundtrack/etc.
        "dedup_removed": 0,              # duplicates removed by AppID
        "kept": 0,                       # final records kept
    }

    rows: List[Dict[str, Any]] = []

    for g in owned:
        appid = g.get("appid")
        name  = g.get("name") or ""

        # Polite delay to avoid hammering Steam Store API (throttling common).
        time.sleep(RATE_LIMIT_SLEEP)

        appdata = get_app_details(appid)
        if not appdata:
            stats["skipped_missing_appdetails"] += 1
            continue
        if not include_non_games and not is_game_type(appdata):
            stats["skipped_non_game_type"] += 1
            continue

        year = parse_year(appdata.get("release_date"))
        devs = appdata.get("developers") or []
        pubs = appdata.get("publishers") or []
        developer = "; ".join(devs) if isinstance(devs, list) else (devs or "")
        publisher = "; ".join(pubs) if isinstance(pubs, list) else (pubs or "")
        g1, g2 = first_two_genres(appdata.get("genres"))

        rows.append({
            "Name": name,
            "Hours_Main_Story": "",          # will be filled from HLTB if matched
            "Genre1": g1 or "",
            "Genre2": g2 or "",
            "Year Released": year if year else "",
            "Developer": developer,
            "Publisher": publisher,
            "AppID": appid
        })

    # Deduplicate by AppID (defensive)
    before = len(rows)
    df = pd.DataFrame(rows, columns=["Name","Hours_Main_Story","Genre1","Genre2","Year Released","Developer","Publisher","AppID"])
    df = df.drop_duplicates(subset=["AppID"])
    stats["dedup_removed"] = before - len(df)
    stats["kept"] = len(df)

    # Persist the raw modeling CSV (games only)
    df.to_csv(RAW_OUTPUT, index=False, encoding="utf-8")
    return df, stats

# ---------------------------
# Enrich with HLTB
# ---------------------------
def enrich_with_hltb(df: pd.DataFrame, hltb_path: str, overrides_path: Optional[str], score_cutoff: int=80, show_matches: bool=True) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """
    Fuzzy-match Steam titles to HLTB rows to fill Hours_Main_Story.
    Also backfills Genre1/2, Year, Developer, Publisher *only if* Steam fields are blank.
    """
    if not os.path.exists(hltb_path):
        raise SystemExit(f"HLTB file not found: {hltb_path}")

    hltb = pd.read_csv(hltb_path)

    # Ensure both frames have the expected columns (create blanks where missing)
    for c in ["Name","Hours_Main_Story","Genre1","Genre2","Year Released","Developer","Publisher"]:
        if c not in df.columns:
            df[c] = ""
        if c not in hltb.columns:
            hltb[c] = "" if c != "Hours_Main_Story" else ""

    # Keys for matching
    df["_key"]   = df["Name"].fillna("").map(normalize_title)
    hltb["_key"] = hltb["Name"].fillna("").map(normalize_title)

    # Manual overrides (normalized left->right)
    override_map = load_overrides_csv(overrides_path)

    # Build a quick index for HLTB
    hltb_keys = hltb["_key"].fillna("").tolist()
    key_to_idx = {k:i for i,k in enumerate(hltb_keys)}

    stats = {
        "override_matches": 0,     # matched via overrides CSV
        "fuzzy_matches": 0,        # matched via fuzzy token_set_ratio
        "no_match": 0,             # didn’t match anything
        "hours_filled": 0,         # Hours_Main_Story populated from HLTB
        "backfill_genre1": 0,      # times Genre1 filled from HLTB because Steam blank
        "backfill_genre2": 0,
        "backfill_year": 0,
        "backfill_dev": 0,
        "backfill_pub": 0,
    }

    # Perform matching
    matches: List[Tuple[int, Optional[int], float, str]] = []
    for i, row in df.iterrows():
        lk = row["_key"]

        # Use manual overrides first (deterministic, "100% match")
        if lk in override_map:
            rk = override_map[lk]
            idx = key_to_idx.get(rk)
            if idx is not None:
                matches.append((i, idx, 100.0, "override"))
                stats["override_matches"] += 1
                continue

        # If no key, record a no-match
        if not lk:
            matches.append((i, None, 0.0, "none"))
            stats["no_match"] += 1
            continue

        # Fuzzy match with token_set_ratio (order-invariant)
        best = process.extractOne(
            lk, hltb_keys, scorer=fuzz.token_set_ratio, score_cutoff=score_cutoff
        )
        if best:
            matched_key, score, _ = best
            matches.append((i, key_to_idx.get(matched_key), float(score), "fuzzy"))
            stats["fuzzy_matches"] += 1
        else:
            matches.append((i, None, 0.0, "no_match"))
            stats["no_match"] += 1

    out = df.copy()
    h_hours = hltb["Hours_Main_Story"].map(coerce_hours)

    def prefer(a, b):
        """Prefer 'a' unless it is blank/NaN/0 → then use 'b' if present."""
        if a is None or (isinstance(a, float) and math.isnan(a)):
            return b
        if str(a).strip() in ("", "0", "0.0"):
            return b
        return a

    new_hours, new_g1, new_g2, new_year, new_dev, new_pub = [], [], [], [], [], []

    for (i, idx, score, mode) in matches:
        h  = out.loc[i, "Hours_Main_Story"]
        g1 = out.loc[i, "Genre1"]; g2 = out.loc[i, "Genre2"]
        yr = out.loc[i, "Year Released"]
        dv = out.loc[i, "Developer"]; pb = out.loc[i, "Publisher"]

        if idx is not None:
            # Hours from HLTB (only if Steam side is empty)
            current   = coerce_hours(h)
            candidate = h_hours.iloc[idx]
            filled    = prefer(current, candidate)
            if filled != current and candidate is not None:
                stats["hours_filled"] += 1
            h = filled

            # Backfill metadata only when Steam fields are blank
            if str(g1).strip() == "":
                g1_hltb = hltb.loc[idx, "Genre1"]
                if str(g1_hltb).strip():
                    g1 = g1_hltb
                    stats["backfill_genre1"] += 1
            if str(g2).strip() == "":
                g2_hltb = hltb.loc[idx, "Genre2"]
                if str(g2_hltb).strip():
                    g2 = g2_hltb
                    stats["backfill_genre2"] += 1
            if str(yr).strip() == "":
                y_hltb = hltb.loc[idx, "Year Released"]
                if str(y_hltb).strip():
                    yr = y_hltb
                    stats["backfill_year"] += 1
            if str(dv).strip() == "":
                d_hltb = hltb.loc[idx, "Developer"]
                if str(d_hltb).strip():
                    dv = d_hltb
                    stats["backfill_dev"] += 1
            if str(pb).strip() == "":
                p_hltb = hltb.loc[idx, "Publisher"]
                if str(p_hltb).strip():
                    pb = p_hltb
                    stats["backfill_pub"] += 1

        new_hours.append(h if h is None else h)
        new_g1.append(g1); new_g2.append(g2)
        new_year.append(yr); new_dev.append(dv); new_pub.append(pb)

    out["Hours_Main_Story"] = new_hours
    out["Genre1"] = new_g1
    out["Genre2"] = new_g2
    out["Year Released"] = new_year
    out["Developer"] = new_dev
    out["Publisher"] = new_pub

    out = out.drop(columns=[c for c in ["_key"] if c in out.columns])
    out.to_csv(FINAL_OUTPUT, index=False, encoding="utf-8")
    return out, stats

# ---------------------------
# CLI / main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Export Steam library and enrich with HLTB (no Hours_Played).")
    parser.add_argument("--api-key", default=os.getenv("STEAM_API_KEY"), help="Steam Web API key")
    parser.add_argument("--steamid",  default=os.getenv("STEAM_ID64"), help="SteamID64 or vanity name/URL")
    parser.add_argument("--hltb",     required=True, help="Path to HLTB CSV (Name, Hours_Main_Story; optional Genre1/2, Year Released, Developer, Publisher)")
    parser.add_argument("--overrides", help="Optional CSV with Steam_Name, HLTB_Name for difficult matches")
    parser.add_argument("--cutoff",   type=int, default=80, help="Fuzzy match score cutoff (0-100; default 80)")
    parser.add_argument("--include-non-games", action="store_true", help="Include DLC/tools/demos (default False)")
    args = parser.parse_args()

    api_key_arg = (args.api_key or "").strip()
    steamid_arg = (args.steamid or "").strip()

    if not api_key_arg:
        raise SystemExit("Missing Steam API key. Set STEAM_API_KEY in .env or pass --api-key.")
    if not steamid_arg:
        raise SystemExit("Missing SteamID. Set STEAM_ID64 in .env or pass --steamid (vanity or 64-bit).")

    # Accept vanity names or full profile URLs, auto-resolve to 64-bit SteamID
    if not looks_like_steamid64(steamid_arg):
        try:
            print(f"Resolving vanity '{steamid_arg}' to SteamID64…")
            steamid_arg = resolve_vanity_to_steamid64(api_key_arg, steamid_arg)
            print(f"Resolved SteamID64: {steamid_arg}")
        except Exception as e:
            raise SystemExit(f"Failed to resolve vanity '{steamid_arg}': {e}")

    # ---- Export (Steam → games only) ----
    print("Exporting Steam library…")
    df_steam, export_stats = export_steam(api_key_arg, steamid_arg, include_non_games=args.include_non_games)
    print(f"Wrote raw Steam CSV → {RAW_OUTPUT} (rows kept: {len(df_steam)})")

    # ---- Enrich (HLTB) ----
    print("Enriching with HLTB (fuzzy match)…")
    df_final, hltb_stats = enrich_with_hltb(df_steam, args.hltb, overrides_path=args.overrides, score_cutoff=args.cutoff, show_matches=True)
    print(f"Wrote enriched CSV → {FINAL_OUTPUT} (rows: {len(df_final)})")

    # ---- Filtering & matching summary ----
    print("\n================= FILTERING SUMMARY =================")
    print("Export phase:")
    print(f"  Owned (GetOwnedGames):           {export_stats['owned_total']}")
    print(f"  Skipped: missing appdetails:     {export_stats['skipped_missing_appdetails']}")
    print(f"  Skipped: non-game type:          {export_stats['skipped_non_game_type']}")
    print(f"  Deduplicates removed (by AppID): {export_stats['dedup_removed']}")
    print(f"  Kept (games only):               {export_stats['kept']}")

    print("\nHLTB enrichment (cutoff = %d):" % args.cutoff)
    print(f"  Matches via overrides:           {hltb_stats['override_matches']}")
    print(f"  Matches via fuzzy:               {hltb_stats['fuzzy_matches']}")
    print(f"  No match:                        {hltb_stats['no_match']}")
    print(f"  Hours filled from HLTB:          {hltb_stats['hours_filled']}")
    print(f"  Backfilled Genre1:               {hltb_stats['backfill_genre1']}")
    print(f"  Backfilled Genre2:               {hltb_stats['backfill_genre2']}")
    print(f"  Backfilled Year Released:        {hltb_stats['backfill_year']}")
    print(f"  Backfilled Developer:            {hltb_stats['backfill_dev']}")
    print(f"  Backfilled Publisher:            {hltb_stats['backfill_pub']}")
    print("=====================================================\n")

    print("Done. Feed steam_library_with_hltb.csv into your modeling pipeline.")

if __name__ == "__main__":
    main()
