#!/usr/bin/env python3
"""
FootyDex — Football Transfer Intelligence Dashboard (Moneyball Edition)
Moneyball Score Script: Merges Transfermarkt profiles with real FBref performance statistics,
engineers advanced Moneyball features (including xG per 90), calculates z-score valuations,
labels players into strategic categories, and outputs data/moneyball_players.csv.
"""

import os
import re
import json
import logging
import unicodedata
import pandas as pd
import numpy as np
from scipy.stats import zscore
from rapidfuzz import process, fuzz

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FootyDex.Moneyball")

def normalize_name(name):
    """Standardizes player names by mapping special characters and stripping diacritics for reliable cross-source merging."""
    if not isinstance(name, str) or not name:
        return ""
    replacements = {
        'ø': 'o', 'Ø': 'o', 'æ': 'ae', 'Æ': 'ae', 'ß': 'ss',
        'đ': 'd', 'Đ': 'd', 'ł': 'l', 'Ł': 'l', 'œ': 'oe', 'Œ': 'oe',
        'ä': 'a', 'ö': 'o', 'ü': 'u', 'Ä': 'a', 'Ö': 'o', 'Ü': 'u'
    }
    for k, v in replacements.items():
        name = name.replace(k, v)
    n = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    n = re.sub(r"[^a-zA-Z0-9\s]", "", n).lower().strip()
    return re.sub(r"\s+", " ", n)

def engineer_features(df_players, df_transfers, df_fbref):
    logger.info("Merging Transfermarkt profiles with real FBref performance statistics...")
    
    # 1. Merge FBref stats onto Transfermarkt players using normalized player names
    df_players = df_players.drop(columns=["goals", "assists", "minutes_played", "xg", "xg_per_90"], errors="ignore")
    df_players["name_norm"] = df_players["player_name"].apply(normalize_name)
    
    if not df_fbref.empty:
        df_fbref["name_norm"] = df_fbref["player_name"].apply(normalize_name)
        # Deduplicate FBref by normalized name in case of multi-club season records
        fbref_dedup = df_fbref.drop_duplicates(subset=["name_norm"], keep="first")
        fbref_cols = ["name_norm", "goals", "assists", "minutes_played", "xg"]
        df = pd.merge(df_players, fbref_dedup[fbref_cols], on="name_norm", how="left")
        
        # Calculate exact match statistics
        exact_matches_cnt = df["minutes_played"].notna().sum()
        unmatched_mask = df["minutes_played"].isna()
        unmatched_players_cnt = unmatched_mask.sum()
        
        # Fuzzy matching using rapidfuzz for unmatched players (threshold ~85)
        logger.info(f"Exact name matches: {exact_matches_cnt}. Performing rapidfuzz fuzzy matching for {unmatched_players_cnt} unmatched players...")
        fbref_names = fbref_dedup["name_norm"].tolist()
        fbref_stats_map = fbref_dedup.set_index("name_norm")[["goals", "assists", "minutes_played", "xg"]].to_dict("index")
        
        fuzzy_matches_cnt = 0
        for idx in df[unmatched_mask].index:
            p_name = df.loc[idx, "name_norm"]
            match = process.extractOne(p_name, fbref_names, scorer=fuzz.WRatio)
            if match and match[1] >= 85:
                fuzzy_matches_cnt += 1
                matched_stats = fbref_stats_map[match[0]]
                df.loc[idx, "goals"] = matched_stats["goals"]
                df.loc[idx, "assists"] = matched_stats["assists"]
                df.loc[idx, "minutes_played"] = matched_stats["minutes_played"]
                df.loc[idx, "xg"] = matched_stats["xg"]
                
        total_matched_cnt = df["minutes_played"].notna().sum()
        still_unmatched_cnt = len(df) - total_matched_cnt
        logger.info(f"\n=== MERGE STATISTICS ===")
        logger.info(f"Total Transfermarkt Players: {len(df)}")
        logger.info(f"Exact Matches: {exact_matches_cnt} ({exact_matches_cnt/len(df)*100:.1f}%)")
        logger.info(f"Fuzzy Matches (>=85 WRatio): {fuzzy_matches_cnt} ({fuzzy_matches_cnt/len(df)*100:.1f}%)")
        logger.info(f"Total Combined Matches: {total_matched_cnt} ({total_matched_cnt/len(df)*100:.1f}%)")
        logger.info(f"Unmatched Players: {still_unmatched_cnt} ({still_unmatched_cnt/len(df)*100:.1f}%)")
        
        # Print samples of unmatched names from both sides
        sample_unmatched_tm = df[df["minutes_played"].isna()]["player_name"].head(10).tolist()
        matched_fbref_norms = set(df[df["minutes_played"].notna()]["name_norm"])
        sample_unmatched_fbref = fbref_dedup[~fbref_dedup["name_norm"].isin(matched_fbref_norms)]["player_name"].head(10).tolist()
        logger.info(f"Sample Unmatched Transfermarkt Players: {sample_unmatched_tm}")
        logger.info(f"Sample Unmatched FBref Players: {sample_unmatched_fbref}")
        logger.info("========================\n")
    else:
        logger.warning("FBref dataset is empty! Defaulting stats to 0.")
        df = df_players.copy()
        df["goals"] = 0
        df["assists"] = 0
        df["minutes_played"] = 0
        df["xg"] = 0.0
        
    df.drop(columns=["name_norm"], inplace=True, errors="ignore")
    
    # Fill missing FBref stats with 0 for players without appearances
    df["goals"] = pd.to_numeric(df["goals"], errors="coerce").fillna(0).astype(int)
    df["assists"] = pd.to_numeric(df["assists"], errors="coerce").fillna(0).astype(int)
    df["minutes_played"] = pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0).astype(int)
    df["xg"] = pd.to_numeric(df["xg"], errors="coerce").fillna(0.0)

    # 2. Merge latest transfer fee information for each player
    if not df_transfers.empty and "player_id" in df_transfers.columns:
        df_transfers_sorted = df_transfers.sort_values(by=["season", "transfer_date"], ascending=[False, False])
        latest_transfers = df_transfers_sorted.drop_duplicates(subset=["player_id"], keep="first")
        df = pd.merge(df, latest_transfers[["player_id", "transfer_fee", "market_value_at_transfer", "season", "transfer_date"]], on="player_id", how="left")
    else:
        df["transfer_fee"] = 0.0
        df["market_value_at_transfer"] = df["market_value"]
        df["season"] = "2024/25"
        df["transfer_date"] = ""

    # Fill missing values
    df["transfer_fee"] = df["transfer_fee"].fillna(0.0)
    df["market_value_at_transfer"] = df["market_value_at_transfer"].fillna(df["market_value"])
    
    # Deduplicate before median imputation using player_id if available, else player_name
    dedup_key = "player_id" if "player_id" in df.columns else "player_name"
    df["market_value"] = pd.to_numeric(df["market_value"], errors="coerce")
    df = df.sort_values(by="market_value", ascending=False, kind="mergesort").drop_duplicates(subset=[dedup_key], keep="first")
    df["market_value"] = df["market_value"].replace(0, np.nan).fillna(df["market_value"].median() or 10_000_000.0)
    df["is_ooc_pro"] = (df["minutes_played"].fillna(0) == 0) & (df["market_value"].fillna(0) >= 1_500_000)
    
    # 3. fee_to_value_ratio = transfer fee / market value (>1 means overpriced, <1 means bargain)
    df["fee_to_value_ratio"] = np.where(
        df["transfer_fee"] > 0,
        df["transfer_fee"] / df["market_value"],
        np.where(df["market_value_at_transfer"] > 0, 0.6, 1.0)
    )
    
    # 4. age_at_transfer
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(25)
    df["age_at_transfer"] = df["age"]
    
    # 5. Per 90 metrics using real FBref stats
    minutes_clamped = np.maximum(df["minutes_played"], 90.0)
    df["goals_per_90"] = df["goals"] / (minutes_clamped / 90.0)
    df["assists_per_90"] = df["assists"] / (minutes_clamped / 90.0)
    df["goal_contributions_per_90"] = df["goals_per_90"] + df["assists_per_90"]
    df["xg_per_90"] = df["xg"] / (minutes_clamped / 90.0)
    
    # Drop is_estimated_stats if present
    df.drop(columns=["is_estimated_stats"], inplace=True, errors="ignore")
    
    return df

def calculate_moneyball_score(df):
    logger.info("Calculating Moneyball Scores using normalized z-scores on real FBref performance and league difficulty...")
    if len(df) < 2:
        df["moneyball_score"] = 50.0
        return df
        
    def safe_zscore(series):
        std = series.std()
        if std == 0 or np.isnan(std):
            return np.zeros(len(series))
        return (series - series.mean()) / std

    # Load league strength multipliers
    league_map = {}
    if os.path.exists("config/league_strength.json"):
        try:
            with open("config/league_strength.json", "r") as f:
                league_map = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load league_strength.json: {e}")
            
    def get_league_mult(comp):
        comp_str = str(comp)
        for k, v in league_map.items():
            if k == "default": continue
            if k.lower() in comp_str.lower():
                return v
        return league_map.get("default", 0.75)
        
    l_mults = df["competition_name"].apply(get_league_mult) if "competition_name" in df.columns else np.ones(len(df))
    df["league_strength_mult"] = l_mults
    
    # Weight goal contributions by league difficulty
    weighted_gc = df["goal_contributions_per_90"] * l_mults
    
    # Weight: low fee-to-value (30%), league-weighted goal contributions/90 (35%), league strength (15%), optimal age 21-26 (10%), minutes played (10%)
    z_low_fee = safe_zscore(-np.log1p(df["fee_to_value_ratio"]))
    z_gc = safe_zscore(weighted_gc)
    z_league = safe_zscore(l_mults)
    
    age_optimality = np.exp(-((df["age"] - 23.5) ** 2) / 18.0)
    z_age = safe_zscore(age_optimality)
    z_min = safe_zscore(np.log1p(df["minutes_played"]))
    
    raw_score = 0.30 * z_low_fee + 0.35 * z_gc + 0.15 * z_league + 0.10 * z_age + 0.10 * z_min
    
    min_val, max_val = raw_score.min(), raw_score.max()
    if max_val > min_val:
        scaled = ((raw_score - min_val) / (max_val - min_val)) * 85.0 + 10.0
    else:
        scaled = np.full(len(df), 50.0)
        
    # Minimum 500 minutes threshold — players below this get a score penalty of 20 points (exempting out-of-coverage senior pros)
    is_ooc_pro = df["is_ooc_pro"] if "is_ooc_pro" in df.columns else False
    penalty = np.where((df["minutes_played"] < 500) & ~is_ooc_pro, 20.0, 0.0)
    scaled = scaled - penalty
    
    # Minimum market value floor of €500K — players below this are capped/filtered out of top rankings entirely
    scaled = np.where(df["market_value"] < 500_000, np.minimum(scaled, 45.0), scaled)
        
    df["moneyball_score"] = np.round(np.clip(scaled, 0.0, 100.0), 1)
    return df

def label_players(df):
    logger.info("Assigning strategic Moneyball labels based on real valuation metrics and sample size guardrails...")
    labels = []
    for _, row in df.iterrows():
        score = row["moneyball_score"]
        ftv = row["fee_to_value_ratio"]
        mv = row["market_value"]
        age = row["age"]
        fee = row["transfer_fee"]
        mins = row.get("minutes_played", 0)
        l_mult = row.get("league_strength_mult", 0.75)
        is_ooc_pro = row.get("is_ooc_pro", False)
        
        # Guardrails: Low sample size or low market value players cannot be in top categories
        if mins < 500 and not is_ooc_pro:
            label = "Sample Size Risk"
        elif mv < 500_000:
            label = "Low Valuation / Unverified"
        elif age > 29 and (fee > mv or ftv > 1.1):
            label = "High Risk"
        elif mv < 15_000_000 and score > 72 and l_mult >= 0.85 and (mins >= 600 or is_ooc_pro):
            label = "Hidden Gem"
        elif score > 75 and ftv < 0.8 and l_mult >= 0.85 and (mins >= 600 or is_ooc_pro):
            label = "Bargain"
        elif score < 40 and ftv > 1.3:
            label = "Overpriced"
        elif 40 <= score <= 75:
            label = "Fair Value"
        else:
            if score > 75 and l_mult >= 0.80:
                label = "Bargain"
            elif score < 40:
                label = "Overpriced"
            else:
                label = "Fair Value"
        labels.append(label)
        
    df["moneyball_label"] = labels
    return df

def main():
    if not os.path.exists("data/players.csv"):
        logger.error("data/players.csv not found! Please run collect_data.py first.")
        return

    df_players = pd.read_csv("data/players.csv")
    df_transfers = pd.read_csv("data/transfers.csv") if os.path.exists("data/transfers.csv") else pd.DataFrame()
    df_fbref = pd.read_csv("data/fbref_stats.csv") if os.path.exists("data/fbref_stats.csv") else pd.DataFrame()
    
    logger.info(f"Loaded {len(df_players)} Transfermarkt profiles, {len(df_fbref)} FBref stat records, and {len(df_transfers)} transfers.")
    
    df_enriched = engineer_features(df_players, df_transfers, df_fbref)
    df_scored = calculate_moneyball_score(df_enriched)
    df_labeled = label_players(df_scored)
    
    os.makedirs("data", exist_ok=True)
    out_path = "data/moneyball_players.csv"
    df_labeled.to_csv(out_path, index=False)
    
    logger.info(f"\nMoneyball Score Engineering Complete! Saved {len(df_labeled)} enriched players with real FBref stats to {out_path}.")
    logger.info("\nLabel Distribution:\n" + str(df_labeled["moneyball_label"].value_counts()))

if __name__ == "__main__":
    main()
