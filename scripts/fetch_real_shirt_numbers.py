#!/usr/bin/env python3
"""
FootyDex - Authentic Club Squad Number Fetcher
Scrapes real club shirt numbers from local Transfermarkt API across all players in the dataset
and caches them in data/player_shirt_numbers.json.
"""

import os
import json
import time
import logging
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FootyDex.ShirtNumberFetcher")

API_BASE = "http://localhost:8000"
CACHE_FILE = "data/player_shirt_numbers.json"

def fetch_player_number(pid):
    pid_str = str(pid).strip()
    if not pid_str or pid_str == "nan":
        return pid_str, None
        
    try:
        # 1. Try profile endpoint
        r = requests.get(f"{API_BASE}/players/{pid_str}/profile", timeout=8).json()
        num_str = str(r.get("shirtNumber", "")).replace("#", "").strip()
        if num_str.isdigit() and int(num_str) > 0:
            return pid_str, int(num_str)
            
        # 2. Try jersey_numbers endpoint
        r2 = requests.get(f"{API_BASE}/players/{pid_str}/jersey_numbers", timeout=8).json()
        jlist = r2.get("jerseyNumbers", [])
        if isinstance(jlist, list):
            for j in jlist:
                if isinstance(j, dict):
                    num = j.get("jerseyNumber")
                    if isinstance(num, (int, float)) and num > 0:
                        return pid_str, int(num)
        return pid_str, None
    except Exception:
        return pid_str, None

def main():
    os.makedirs("data", exist_ok=True)
    
    # Load existing cache if available
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info(f"Loaded {len(cache)} existing shirt numbers from cache.")
        except Exception as e:
            logger.warning(f"Could not read cache: {e}")
            
    # Gather all unique player_ids from data CSVs
    all_pids = set()
    for fn in ["recruitment_index.csv", "players.csv", "moneyball_players.csv"]:
        path = os.path.join("data", fn)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "player_id" in df.columns:
                    pids = df["player_id"].dropna().astype(str).unique()
                    all_pids.update(pids)
            except Exception as e:
                logger.warning(f"Error reading {fn}: {e}")
                
    # Filter out already cached valid numbers
    to_fetch = [pid for pid in all_pids if pid not in cache or cache[pid] is None]
    logger.info(f"Total unique players: {len(all_pids)}. Need to fetch: {len(to_fetch)}.")
    
    if to_fetch:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=30) as ex:
            futures = {ex.submit(fetch_player_number, pid): pid for pid in to_fetch}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching Authentic Shirt Numbers"):
                pid, num = fut.result()
                if num is not None:
                    cache[pid] = num
                    
        elapsed = time.time() - t0
        logger.info(f"Fetched {len(to_fetch)} players in {elapsed:.1f}s. Valid numbers in cache: {sum(1 for v in cache.values() if v is not None)}.")
        
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        logger.info(f"Saved authentic shirt numbers to {CACHE_FILE}.")
    
    # Now update the CSV files directly so they contain the real numbers
    for fn in ["recruitment_index.csv", "players.csv", "moneyball_players.csv"]:
        path = os.path.join("data", fn)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if "player_id" in df.columns:
                    df["player_id_str"] = df["player_id"].astype(str).str.strip()
                    # Apply cached shirt numbers
                    mapped_nums = df["player_id_str"].map(cache)
                    if "shirt_number" in df.columns:
                        df["shirt_number"] = mapped_nums.combine_first(df["shirt_number"])
                    else:
                        df["shirt_number"] = mapped_nums
                    df.drop(columns=["player_id_str"], inplace=True)
                    df.to_csv(path, index=False)
                    logger.info(f"Updated {fn} with authentic shirt numbers.")
            except Exception as e:
                logger.warning(f"Could not update {fn}: {e}")

if __name__ == "__main__":
    main()
