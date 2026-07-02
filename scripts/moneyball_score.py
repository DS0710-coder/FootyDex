#!/usr/bin/env python3
"""
FootyDex — Football Transfer Intelligence Dashboard (Moneyball Edition)
Moneyball Score Script: Engineers features, calculates the Moneyball score using z-score normalization,
labels players into strategic categories, and outputs data/moneyball_players.csv.
"""

import os
import logging
import pandas as pd
import numpy as np
from scipy.stats import zscore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FootyDex.Moneyball")

def engineer_features(df_players, df_transfers):
    logger.info("Engineering Moneyball features...")
    
    # Merge latest transfer information for each player
    if not df_transfers.empty and "player_id" in df_transfers.columns:
        # Sort transfers by date or season descending to get latest transfer
        df_transfers_sorted = df_transfers.sort_values(by=["season", "transfer_date"], ascending=[False, False])
        latest_transfers = df_transfers_sorted.drop_duplicates(subset=["player_id"], keep="first")
        
        df = pd.merge(df_players, latest_transfers[["player_id", "transfer_fee", "market_value_at_transfer", "season", "transfer_date"]], on="player_id", how="left")
    else:
        df = df_players.copy()
        df["transfer_fee"] = 0.0
        df["market_value_at_transfer"] = df["market_value"]
        df["season"] = "2024/25"
        df["transfer_date"] = ""

    # Fill missing values
    df["transfer_fee"] = df["transfer_fee"].fillna(0.0)
    df["market_value_at_transfer"] = df["market_value_at_transfer"].fillna(df["market_value"])
    df["market_value"] = df["market_value"].replace(0, np.nan).fillna(df["market_value"].median() or 10_000_000.0)
    
    # 1. fee_to_value_ratio = transfer fee / market value (>1 means overpriced, <1 means bargain)
    # For players with 0 transfer fee (academy/free), we set ratio to 0.5 (bargain baseline) or actual ratio
    df["fee_to_value_ratio"] = np.where(
        df["transfer_fee"] > 0,
        df["transfer_fee"] / df["market_value"],
        np.where(df["market_value_at_transfer"] > 0, 0.6, 1.0)
    )
    
    # 2. age_at_transfer = age when transferred (approximate from current age if needed)
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(25)
    df["age_at_transfer"] = df["age"]  # Default
    
    # 3. Per 90 metrics: goals_per_90, assists_per_90, goal_contributions_per_90
    df["minutes_played"] = pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0)
    df["goals"] = pd.to_numeric(df["goals"], errors="coerce").fillna(0)
    df["assists"] = pd.to_numeric(df["assists"], errors="coerce").fillna(0)
    
    # Avoid division by zero by clamping minutes to at least 90 for active calculation
    minutes_clamped = np.maximum(df["minutes_played"], 90.0)
    df["goals_per_90"] = df["goals"] / (minutes_clamped / 90.0)
    df["assists_per_90"] = df["assists"] / (minutes_clamped / 90.0)
    df["goal_contributions_per_90"] = df["goals_per_90"] + df["assists_per_90"]
    
    return df

def calculate_moneyball_score(df):
    logger.info("Calculating Moneyball Scores using normalized z-scores...")
    if len(df) < 2:
        df["moneyball_score"] = 50.0
        return df
        
    # Z-score normalization with NaN protection
    def safe_zscore(series):
        std = series.std()
        if std == 0 or np.isnan(std):
            return np.zeros(len(series))
        return (series - series.mean()) / std

    # Weight: low fee-to-value (40%), high goal contributions/90 (30%), optimal age 21-26 (20%), minutes played (10%)
    # 1. Low fee-to-value -> invert so lower ratio is higher score
    z_low_fee = safe_zscore(-np.log1p(df["fee_to_value_ratio"]))
    
    # 2. High goal contributions per 90
    z_gc = safe_zscore(df["goal_contributions_per_90"])
    
    # 3. Optimal age 21-26 -> gaussian reward around age 23.5
    age_optimality = np.exp(-((df["age"] - 23.5) ** 2) / 18.0)
    z_age = safe_zscore(age_optimality)
    
    # 4. Minutes played
    z_min = safe_zscore(np.log1p(df["minutes_played"]))
    
    raw_score = 0.40 * z_low_fee + 0.30 * z_gc + 0.20 * z_age + 0.10 * z_min
    
    # Scale score to 0-100 robustly
    min_val, max_val = raw_score.min(), raw_score.max()
    if max_val > min_val:
        scaled = ((raw_score - min_val) / (max_val - min_val)) * 85.0 + 10.0
    else:
        scaled = np.full(len(df), 50.0)
        
    df["moneyball_score"] = np.round(np.clip(scaled, 0.0, 100.0), 1)
    return df

def label_players(df):
    logger.info("Assigning strategic Moneyball labels...")
    labels = []
    for _, row in df.iterrows():
        score = row["moneyball_score"]
        ftv = row["fee_to_value_ratio"]
        mv = row["market_value"]
        age = row["age"]
        fee = row["transfer_fee"]
        
        if age > 29 and (fee > mv or ftv > 1.1):
            label = "High Risk"
        elif mv < 15_000_000 and score > 70:
            label = "Hidden Gem"
        elif score > 75 and ftv < 0.8:
            label = "Bargain"
        elif score < 40 and ftv > 1.3:
            label = "Overpriced"
        elif 40 <= score <= 75:
            label = "Fair Value"
        else:
            # Fallback based on score
            if score > 75:
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
    
    logger.info(f"Loaded {len(df_players)} players and {len(df_transfers)} transfers.")
    
    df_enriched = engineer_features(df_players, df_transfers)
    df_scored = calculate_moneyball_score(df_enriched)
    df_labeled = label_players(df_scored)
    
    os.makedirs("data", exist_ok=True)
    out_path = "data/moneyball_players.csv"
    df_labeled.to_csv(out_path, index=False)
    
    logger.info(f"\nMoneyball Score Engineering Complete! Saved {len(df_labeled)} enriched players to {out_path}.")
    logger.info("\nLabel Distribution:\n" + str(df_labeled["moneyball_label"].value_counts()))

if __name__ == "__main__":
    main()
