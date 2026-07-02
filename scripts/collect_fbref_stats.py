#!/usr/bin/env python3
"""
FootyDex — Football Transfer Intelligence Dashboard (Moneyball Edition)
FBref Stats Collector: Scrapes real standard player performance stats (goals, assists, minutes, xG)
from FBref across the 5 major European leagues for the 2024-25 season using soccerdata.
"""

import os
import argparse
import logging
import pandas as pd
import soccerdata as sd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FootyDex.FBrefCollector")

TARGET_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

TARGET_SEASON = "2425"

def clean_fbref_data(df):
    """Resets MultiIndex and standardizes column names into a clean tabular structure."""
    logger.info("Cleaning and standardizing FBref performance data...")
    
    # Reset MultiIndex (league, season, team, player) into standard columns
    df_reset = df.reset_index()
    
    # Flatten MultiIndex columns so row indexing returns clean scalar strings, not Series
    flat_cols = []
    for col in df_reset.columns:
        if isinstance(col, tuple):
            parts = [str(p).strip() for p in col if str(p).strip() != ""]
            flat_cols.append("_".join(parts) if parts else "unknown")
        else:
            flat_cols.append(str(col).strip())
    df_reset.columns = flat_cols
    
    clean_records = []
    
    for _, row in df_reset.iterrows():
        # 1. Extract Index / Basic Info
        player = row.get("player", row.get("Player", "Unknown"))
        team = row.get("team", row.get("Team", row.get("Squad", "Unknown")))
        league = row.get("league", row.get("League", "Unknown"))
        season = row.get("season", row.get("Season", "2024-25"))
        
        # Helper to find stat values across flat column strings
        def find_stat(target_keys):
            for col in df_reset.columns:
                col_name = str(col).strip().lower()
                for t in target_keys:
                    if col_name == t or col_name.endswith("_" + t) or col_name.endswith(" " + t):
                        val = row[col]
                        try:
                            return float(val) if pd.notnull(val) and val != "" else 0.0
                        except (ValueError, TypeError):
                            continue
            return 0.0
            
        minutes = find_stat(["min", "minutes", "minutes_played", "mins"])
        goals = find_stat(["gls", 'goals', 'g'])
        assists = find_stat(["ast", 'assists', 'a'])
        xg = find_stat(["xg", 'expected_goals', 'exp_g', 'xgo'])
        
        clean_records.append({
            "player_name": str(player).strip(),
            "club": str(team).strip(),
            "league": str(league).strip(),
            "season": str(season).strip(),
            "goals": int(goals),
            "assists": int(assists),
            "minutes_played": int(minutes),
            "xg": float(xg),
        })
        
    df_clean = pd.DataFrame(clean_records)
    # Drop duplicates if a player played for two clubs or appeared twice
    df_clean = df_clean.drop_duplicates(subset=["player_name", "club", "league"], keep="first")
    return df_clean

def collect_fbref_stats(leagues=None, season=TARGET_SEASON):
    os.makedirs("data", exist_ok=True)
    if leagues is None or leagues == TARGET_LEAGUES:
        leagues_to_scrape = ["Big 5 European Leagues Combined"]
        logger.info(f"Initializing soccerdata FBref scraper for season {season} using Big 5 Combined optimization...")
    else:
        leagues_to_scrape = leagues
        logger.info(f"Initializing soccerdata FBref scraper for season {season} across {len(leagues_to_scrape)} leagues...")
    fbref = sd.FBref(leagues=leagues_to_scrape, seasons=season)
    
    logger.info("Reading standard player season statistics from FBref...")
    df_raw = fbref.read_player_season_stats(stat_type="standard")
    
    if df_raw is None or df_raw.empty:
        logger.error("Failed to retrieve data from FBref.")
        return
        
    df_clean = clean_fbref_data(df_raw)
    
    out_path = "data/fbref_stats.csv"
    df_clean.to_csv(out_path, index=False)
    logger.info(f"Successfully saved {len(df_clean)} player performance records to {out_path}.")
    logger.info(f"\nSample data:\n{df_clean.head(5)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape FBref standard player season stats.")
    parser.add_argument("--league", type=str, default=None, help="Specific league to scrape (for testing)")
    args = parser.parse_args()
    
    leagues_arg = [args.league] if args.league else None
    collect_fbref_stats(leagues=leagues_arg)
