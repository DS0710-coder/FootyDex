#!/usr/bin/env python3
"""
FootyDex — Football Transfer Intelligence Dashboard (v2.0)
FBref Stats Collector: Scrapes and merges multi-table player performance stats
(standard, passing, defense, possession, gca, misc, shooting) across major leagues.
"""

import os
import argparse
import logging
import pandas as pd
import soccerdata as sd
from lxml import etree, html
from soccerdata.fbref import FBREF_API, _parse_table, _fix_nation_col, _concat, TEAMNAME_REPLACEMENTS, standardize_colnames

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

def get_fbref_table(fb, stat_type):
    """Custom extractor to pull any FBref table by bypassing default stat_type restrictions."""
    page = stat_type
    seasons = fb.read_seasons()
    filemask = "players_{}_{}_{}.html"
    players = []
    for (lkey, skey), season in seasons.iterrows():
        big_five = lkey == "Big 5 European Leagues Combined"
        filepath = fb.data_dir / filemask.format(lkey, skey, stat_type)
        url = (FBREF_API + "/".join(season.url.split("/")[:-1]) + f"/{page}" + ("/players/" if big_five else "/") + season.url.split("/")[-1])
        reader = fb.get(url, filepath)
        tree = html.parse(reader)
        for elem in tree.xpath('//td[@data-stat="comp_level"]//span'):
            elem.getparent().remove(elem)
        try:
            (el,) = tree.xpath(f'//comment()[contains(., "div_stats_{stat_type}")]')
            parser = etree.HTMLParser(recover=True)
            (html_table,) = etree.fromstring(el.text, parser).xpath(f'//table[contains(@id, "stats_{stat_type}")]')
            df_table = _parse_table(html_table)
            df_table[("Unnamed: league", "league")] = lkey
            df_table[("Unnamed: season", "season")] = skey
            df_table = _fix_nation_col(df_table)
            players.append(df_table)
        except Exception as e:
            logger.warning(f"Failed to parse table {stat_type} for {lkey} {skey}: {e}")
            continue
    if not players:
        return pd.DataFrame()
    df = _concat(players, key=["league", "season"])
    df = df[df.Player != "Player"]
    return df.drop("Matches", axis=1, level=0, errors="ignore").drop("Rk", axis=1, level=0, errors="ignore").rename(columns={"Squad": "team"}).replace({"team": TEAMNAME_REPLACEMENTS}).pipe(standardize_colnames, cols=["Player", "Nation", "Pos", "Age", "Born"]).set_index(["league", "season", "team", "player"]).sort_index()

def flatten_columns(df, prefix=""):
    flat_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(p).strip() for p in col if str(p).strip() != "" and not str(p).startswith("Unnamed")]
            col_name = "_".join(parts)
        else:
            col_name = str(col).strip()
        flat_cols.append(f"{prefix}{col_name}".lower())
    df.columns = flat_cols
    return df

def collect_fbref_stats(leagues=None, season=TARGET_SEASON):
    os.makedirs("data", exist_ok=True)
    leagues_to_scrape = leagues if leagues else ["Big 5 European Leagues Combined"]
    logger.info(f"Initializing soccerdata FBref scraper for season {season} across {leagues_to_scrape}...")
    fb = sd.FBref(leagues=leagues_to_scrape, seasons=season)
    
    tables_to_pull = ["standard", "passing", "defense", "possession", "gca", "misc", "shooting"]
    merged_df = None
    
    for t_name in tables_to_pull:
        logger.info(f"Extracting FBref table: {t_name}...")
        try:
            if t_name == "standard":
                df_t = fb.read_player_season_stats(stat_type="standard")
            else:
                df_t = get_fbref_table(fb, t_name)
                
            if df_t is None or df_t.empty:
                logger.warning(f"Table {t_name} is empty, skipping...")
                continue
                
            df_t = flatten_columns(df_t, prefix=f"{t_name}_" if t_name != "standard" else "")
            
            if merged_df is None:
                merged_df = df_t
            else:
                # Merge on index [league, season, team, player]
                cols_to_use = [c for c in df_t.columns if c not in merged_df.columns]
                merged_df = merged_df.join(df_t[cols_to_use], how="left")
        except Exception as e:
            logger.error(f"Error processing table {t_name}: {e}")
            
    if merged_df is None or merged_df.empty:
        logger.error("No data extracted from FBref.")
        return
        
    df_clean = merged_df.reset_index()
    
    # Clean up standard column names for downstream engines
    rename_map = {
        "player": "player_name",
        "team": "club",
        "playing time_min": "minutes_played",
        "performance_gls": "goals",
        "performance_ast": "assists",
        "expected_xg": "xg",
        "expected_xag": "xag",
        "expected_npxg": "npxg",
        "progression_prgc": "prg_carries",
        "progression_prgp": "prg_passes",
        "progression_prgr": "prg_passes_received",
        "passing_total_cmp%": "pass_cmp_pct",
        "passing_total_prgdist": "prg_pass_dist",
        "passing_long_cmp%": "long_pass_cmp_pct",
        "passing_1/3": "final_third_passes",
        "passing_kp": "key_passes",
        "passing_tb": "through_balls",
        "passing_sw": "switches",
        "defense_tackles_tklw": "tackles_won",
        "defense_challenges_tkl%": "def_duels_won_pct",
        "defense_blocks_blocks": "blocks",
        "defense_int": "interceptions",
        "defense_clr": "clearances",
        "defense_err": "errors_to_shot",
        "possession_touches_att pen": "touches_in_box",
        "possession_take-ons_succ": "successful_dribbles",
        "possession_take-ons_succ%": "dribble_success_pct",
        "possession_carries_1/3": "carries_final_third",
        "possession_carries_cpa": "carries_into_box",
        "possession_mis": "miscontrols",
        "possession_dis": "dispossessed",
        "gca_sca_sca": "sca",
        "gca_gca_gca": "gca",
        "misc_aerial duels_won%": "aerial_won_pct",
        "misc_recov": "recoveries",
        "shooting_standard_sot%": "shots_on_target_pct",
        "shooting_standard_g/sh": "shot_conversion_pct"
    }
    
    df_clean.rename(columns=rename_map, inplace=True)
            
    # Ensure numerical formatting
    num_cols = ["minutes_played", "goals", "assists", "xg", "xag", "npxg", "prg_carries", "prg_passes", 
                "pass_cmp_pct", "prg_pass_dist", "long_pass_cmp_pct", "final_third_passes", "key_passes", 
                "through_balls", "tackles_won", "def_duels_won_pct", "blocks", "interceptions", "clearances", 
                "errors_to_shot", "touches_in_box", "successful_dribbles", "dribble_success_pct", "sca", "gca", 
                "aerial_won_pct", "recoveries", "shots_on_target_pct", "shot_conversion_pct"]
                
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0.0)
        else:
            df_clean[col] = 0.0
            
    out_path = "data/fbref_stats.csv"
    write_fbref = True
    if os.path.exists(out_path) and not df_clean.empty:
        try:
            df_existing = pd.read_csv(out_path)
            combined = pd.concat([df_existing, df_clean], ignore_index=True)
            # Use season+league alongside player+club so records from different
            # seasons or competitions are preserved rather than overwritten.
            dedup_cols = [c for c in ["league", "season", "player_name", "club"] if c in combined.columns]
            df_clean = combined.drop_duplicates(subset=dedup_cols, keep="last")
        except Exception as e:
            logger.error(f"Could not merge with existing {out_path}: {e}. Aborting write to protect historical data.")
            write_fbref = False
            
    if write_fbref:
        df_clean.to_csv(out_path, index=False)
        logger.info(f"Successfully saved {len(df_clean)} enriched multi-table player records to {out_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape FBref multi-table player stats.")
    parser.add_argument("--league", type=str, default=None, help="Specific league to scrape")
    args = parser.parse_args()
    leagues = [args.league] if args.league else None
    collect_fbref_stats(leagues=leagues)
