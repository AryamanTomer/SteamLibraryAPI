#!/usr/bin/env python3
"""
Simple, offline political-lean estimator for Steam games (two-party share).
- Uses only your CSV columns (no web calls).
- No franchise reliance; includes light name-token features.
- Soft priors + small ML model (logistic regression with calibration).
- Outputs two CSVs in the current folder (SteamLibraryAPI):
    - steam_lean_refined.csv     (per-game two-party Dem/Rep % + confidence)
    - steam_lean_summary.csv     (one-row library-level summary)
Run:
    py -3 simple_political_model.py --csv ".\steam_library_with_hltb.csv" --anchor-prior --two-party-prior 0.50
"""

import argparse, math, re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# -------------------------
# Small, conservative priors
# -------------------------
GENRE_PRIOR = {
    "Narrative": +0.20, "Visual Novel": +0.20, "Puzzle": +0.20, "RPG": +0.10,
    "Action-Adventure": +0.05, "Open World": +0.05, "Adventure": +0.05,
    "FPS": -0.15, "Military": -0.20, "Tactical": -0.10, "Shooter": -0.05,
    "Stealth": +0.10, "Comedy": +0.10, "Indie": +0.25, "Survival": +0.05, "Sci-Fi": +0.05,
}

# Light brand hints (dev/publisher text contains…)
BRAND_HINTS = {
    "valve": +0.15, "rockstar": +0.10, "arkane": +0.10, "irrational": +0.10, "remedy": +0.10,
    "bioware": +0.10, "quantic dream": +0.15, "telltale": +0.10, "volition": +0.05, "4a games": +0.05,
    "infinity ward": -0.10, "treyarch": -0.10, "sledgehammer": -0.10, "activision": -0.05,
    "gearbox": -0.02, "bungie": -0.02, "343 industries": -0.02,
}

# -------------------------
# Helpers
# -------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def safe_float(x) -> Optional[float]:
    try:
        f = float(x)
        if math.isnan(f): return None
        return f
    except Exception:
        return None

def year_effect(y) -> float:
    try:
        yr = int(float(y))
    except Exception:
        return 0.0
    # tiny positive tilt for newer titles
    return ((yr - 2000) / 10.0) * 0.05

def hours_effect(h) -> float:
    h = safe_float(h)
    if h is None: return 0.0
    # small nudge: longer AAA ~ slightly less Dem, short indies/puzzles ~ slightly more Dem
    return -0.03 * ((h - 10.0) / 10.0)

def lower(s: Any) -> str:
    return str(s).lower() if isinstance(s, str) else ""

def np_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def np_inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

# Safe CSV write: if file is locked, write a timestamped fallback in the same folder
def safe_to_csv(df: pd.DataFrame, filename: str) -> str:
    p = Path(filename)
    try:
        df.to_csv(p, index=False, encoding="utf-8")
        return str(p)
    except PermissionError:
        fallback = p.with_name(f"{p.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{p.suffix or '.csv'}")
        df.to_csv(fallback, index=False, encoding="utf-8")
        print(f"[warn] '{p.name}' locked; wrote fallback: {fallback.name}")
        return str(fallback)

# -------------------------
# Name tokens (no priors)
# -------------------------
NAME_STOP = set("""
the a an and or of for to in on at from edition remastered definitive complete goty hd
re-elected redux anniversary collection enhanced ultimate game games trilogy saga
""".split())

ROMAN = {"iii":"3","ii":"2","iv":"4","vi":"6"}

def normalize_title_tokens(name: str) -> List[str]:
    s = re.sub(r"[^a-zA-Z0-9\s\-:']", " ", name or "")
    s = re.sub(r"\s+", " ", s).strip().lower()
    toks = [ROMAN.get(t, t) for t in s.split() if t not in NAME_STOP and len(t) > 2]
    return toks

def discover_name_tokens(names: List[str], min_count: int=3) -> List[str]:
    from collections import Counter
    c = Counter()
    for nm in names:
        for t in normalize_title_tokens(nm):
            c[t]+=1
    return [t for t,n in c.items() if n >= min_count]

# -------------------------
# Prior scoring (soft)
# -------------------------
def prior_score(row: pd.Series, prior_scale: float) -> float:
    s = 0.0
    for g in [row.get("Genre1",""), row.get("Genre2","")]:
        s += GENRE_PRIOR.get(g, 0.0)
    for field in [row.get("Developer",""), row.get("Publisher","")]:
        t = lower(field)
        for key, w in BRAND_HINTS.items():
            if key in t:
                s += w
    s += year_effect(row.get("Year Released",""))
    s += hours_effect(row.get("Hours_Main_Story",""))
    return s * prior_scale

# -------------------------
# Featureization (no external data)
# -------------------------
def build_features(df: pd.DataFrame,
                   min_genre: int, min_dev: int, min_pub: int, min_name_token: int) -> (np.ndarray, List[str]):
    # Significant genres by frequency
    from collections import Counter
    g_list = [g for g in pd.concat([df["Genre1"], df["Genre2"]], axis=0).fillna("").tolist() if g]
    g_counts = Counter(g_list)
    sig_genres = {g for g,c in g_counts.items() if c >= min_genre}

    # Frequent devs/pubs
    dev_counts = Counter([d for d in df["Developer"].fillna("").tolist() if d])
    pub_counts = Counter([p for p in df["Publisher"].fillna("").tolist() if p])
    sig_devs = {d for d,c in dev_counts.items() if c >= min_dev}
    sig_pubs = {p for p,c in pub_counts.items() if c >= min_pub}

    # Name tokens
    sig_name_tokens = set(discover_name_tokens(df["Name"].fillna("").tolist(), min_count=min_name_token))

    def feat_row(r) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        for g in [r.get("Genre1",""), r.get("Genre2","")]:
            if g in sig_genres: feats[f"genre::{g}"] = 1.0
        dev = r.get("Developer",""); pub = r.get("Publisher","")
        for d in sig_devs:
            if d and d.lower() in lower(dev): feats[f"dev::{d}"] = 1.0
        for p in sig_pubs:
            if p and p.lower() in lower(pub): feats[f"pub::{p}"] = 1.0
        for tok in normalize_title_tokens(r.get("Name","")):
            if tok in sig_name_tokens: feats[f"name::{tok}"] = 1.0
        # numeric (simple scaling)
        yr = safe_float(r.get("Year Released",""))
        if yr is not None: feats["num::year_centered"] = (yr - 2000.0) / 10.0
        hr = safe_float(r.get("Hours_Main_Story",""))
        if hr is not None: feats["num::hours_centered"] = (hr - 10.0) / 10.0
        return feats

    feat_dicts = [feat_row(r) for _, r in df.iterrows()]
    all_keys = sorted({k for d in feat_dicts for k in d.keys()})
    X = np.zeros((len(feat_dicts), len(all_keys)), dtype=float)
    for i, d in enumerate(feat_dicts):
        for k, v in d.items():
            X[i, all_keys.index(k)] = v
    return X, all_keys

def feature_coverage(feat_keys: List[str]) -> float:
    groups = {
        "genre": any(k.startswith("genre::") for k in feat_keys),
        "brand": any(k.startswith("dev::") or k.startswith("pub::") for k in feat_keys),
        "name": any(k.startswith("name::") for k in feat_keys),
        "numeric": any(k.startswith("num::") for k in feat_keys),
    }
    return sum(groups.values())/len(groups)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Input CSV. Defaults to steam_library_with_hltb.csv else steam_library_for_model.csv in current folder.")
    ap.add_argument("--min-genre", type=int, default=3)
    ap.add_argument("--min-developer", type=int, default=3)
    ap.add_argument("--min-publisher", type=int, default=3)
    ap.add_argument("--min-name-token", type=int, default=3)
    ap.add_argument("--prior-scale", type=float, default=0.6)
    ap.add_argument("--C", type=float, default=0.5, help="LogReg inverse regularization (lower => stronger reg).")
    ap.add_argument("--temp", type=float, default=0.85, help="Smoothing: p' = 0.5 + (p-0.5)*temp")
    ap.add_argument("--coverage-weight", type=float, default=0.2)
    ap.add_argument("--two-party-prior", type=float, default=0.50)
    ap.add_argument("--anchor-prior", action="store_true")
    ap.add_argument("--out-csv", default="steam_lean_refined.csv")
    ap.add_argument("--summary-csv", default="steam_lean_summary.csv")
    args = ap.parse_args()

    # pick input CSV inside current dir
    in_csv = args.csv or ("steam_library_with_hltb.csv" if Path("steam_library_with_hltb.csv").exists() else "steam_library_for_model.csv")
    if not Path(in_csv).exists():
        raise SystemExit(f"Input CSV not found: {in_csv}")

    # load
    df = pd.read_csv(in_csv, low_memory=False)
    for need in ["Name","Genre1","Genre2","Year Released","Developer","Publisher"]:
        if need not in df.columns:
            df[need] = ""
    if "Hours_Main_Story" not in df.columns:
        df["Hours_Main_Story"] = ""

    # soft pseudo-labels from priors
    df["prior_dem_share"] = df.apply(lambda r: sigmoid(prior_score(r, args.prior_scale)), axis=1)
    y_pseudo = (df["prior_dem_share"] >= 0.5).astype(int)

    # features
    X, keys = build_features(df, args.min_genre, args.min_developer, args.min_publisher, args.min_name_token)

    # fit small LR + calibration
    clf = LogisticRegression(penalty="l2", C=args.C, solver="liblinear", max_iter=300, class_weight="balanced")
    cal = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
    cal.fit(X, y_pseudo)
    proba_dem = cal.predict_proba(X)[:,1]

    # optional anchoring to two-party prior
    if args.anchor_prior:
        target = float(args.two_party_prior)
        logits = np_logit(proba_dem)
        shift = np_logit(np.array([target]))[0] - logits.mean()
        proba_dem = np_inv_logit(logits + shift)

    # temperature smoothing and caps
    proba_dem = 0.5 + (proba_dem - 0.5) * float(args.temp)
    proba_dem = np.clip(proba_dem, 0.01, 0.99)

    # final two-party composition
    dem_2p = proba_dem
    rep_2p = 1.0 - dem_2p

    # confidence: blend certainty and coverage
    feat_keys_per_row = []
    for i in range(X.shape[0]):
        present = [k for j,k in enumerate(keys) if X[i,j] != 0.0]
        feat_keys_per_row.append(present)
    coverage = np.array([feature_coverage(lst) for lst in feat_keys_per_row])
    certainty = np.maximum(dem_2p, rep_2p)  # distance from 0.5
    cw = max(0.0, min(1.0, float(args.coverage_weight)))
    conf_score = (1.0 - cw)*certainty + cw*coverage

    def bucket(c: float) -> str:
        if c >= 0.75: return "high"
        if c >= 0.60: return "medium"
        return "low"
    conf = np.array([bucket(c) for c in conf_score])

    # output frames
    out = df.copy()
    out["Dem_Share_2P"] = np.round(dem_2p, 4)
    out["Rep_Share_2P"] = np.round(rep_2p, 4)
    out["Dem_%_2P"] = (dem_2p * 100).round(1)
    out["Rep_%_2P"] = (rep_2p * 100).round(1)
    out["Lean"] = np.where(dem_2p >= 0.5, "Dem-lean", "Rep-lean")
    out["ConfidenceScore"] = np.round(conf_score, 3)
    out["Confidence"] = conf

    keep_cols = [
        "Name","Hours_Main_Story","Genre1","Genre2","Year Released","Developer","Publisher"
    ]
    if "AppID" in out.columns:
        keep_cols.append("AppID")
    keep_cols += ["Dem_Share_2P","Rep_Share_2P","Dem_%_2P","Rep_%_2P","Lean","Confidence","ConfidenceScore"]

    wrote_refined = safe_to_csv(out[keep_cols], args.out_csv)

    summary = pd.DataFrame([{
        "Run_Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Input_CSV": in_csv,
        "Total_Games": int(len(out)),
        "Avg_Dem_Share_2P": float(out["Dem_Share_2P"].mean()),
        "Avg_Rep_Share_2P": float(out["Rep_Share_2P"].mean()),
        "Avg_Dem_%_2P": float(out["Dem_%_2P"].mean()),
        "Avg_Rep_%_2P": float(out["Rep_%_2P"].mean()),
        "Dem_Lead_Count": int((out["Dem_Share_2P"] >= 0.5).sum()),
        "Rep_Lead_Count": int((out["Dem_Share_2P"] < 0.5).sum()),
        "Confidence_High": int((out["Confidence"] == "high").sum()),
        "Confidence_Medium": int((out["Confidence"] == "medium").sum()),
        "Confidence_Low": int((out["Confidence"] == "low").sum()),
        "PriorScale": float(args.prior_scale),
        "C": float(args.C),
        "Temp": float(args.temp),
        "CoverageWeight": float(args.coverage_weight),
        "AnchorPrior": bool(args.anchor_prior),
        "TwoPartyPrior": float(args.two_party_prior),
        "Model_Feature_Count": int(len(keys)),
    }])
    wrote_summary = safe_to_csv(summary, args.summary_csv)

    # console report
    print("\n=== Simple Political Model (two-party) ===")
    print(f"Rows: {len(out)} | Avg Dem_%_2P: {out['Dem_%_2P'].mean():.1f}")
    print(f"Confidence → high:{(out['Confidence']=='high').sum()}  "
          f"medium:{(out['Confidence']=='medium').sum()}  "
          f"low:{(out['Confidence']=='low').sum()}")
    print(f"Wrote: {Path(wrote_refined).name}")
    print(f"Wrote: {Path(wrote_summary).name}")

if __name__ == "__main__":
    main()
