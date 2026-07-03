#!/usr/bin/env python3
"""
FootyDex — Football Transfer Intelligence Dashboard (Moneyball Edition)
Data Collection Script: Collects club, player profile, market value, and transfer fee data
from the local Transfermarkt API running at http://localhost:8000.
"""

import os
import time
import argparse
import logging
import re
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FootyDex.Collector")

BASE_URL = "http://localhost:8000"
RATE_LIMIT_DELAY = 0.5  # Seconds between requests

TARGET_COMPETITIONS = [
    {"name": "Premier League", "id": "GB1"},
    {"name": "La Liga", "id": "ES1"},
    {"name": "Bundesliga", "id": "L1"},
    {"name": "Serie A", "id": "IT1"},
    {"name": "Ligue 1", "id": "FR1"},
]

def make_request(url, params=None, retries=2):
    """Makes a rate-limited GET request to the local API with retry logic."""
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            response = requests.get(url, params=params, timeout=8)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            else:
                logger.warning(f"Request failed: {url} (Status {response.status_code})")
        except Exception as e:
            logger.warning(f"Error requesting {url} on attempt {attempt+1}: {e}")
            time.sleep(1)
    return None

def parse_currency_to_float(val):
    """Converts Transfermarkt currency strings (e.g. '€10.00m', '€500k', '€200', 50000000) to numeric float in Euros."""
    if val is None or val == "" or val == "-" or val == "?":
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).replace("€", "").replace(",", "").strip()
    if not val_str:
        return 0.0
        
    mult = 1.0
    if val_str.lower().endswith("m") or val_str.lower().endswith("mio"):
        mult = 1_000_000.0
        val_str = re.sub(r"[a-zA-Z]", "", val_str)
    elif val_str.lower().endswith("k") or val_str.lower().endswith("tsd"):
        mult = 1_000.0
        val_str = re.sub(r"[a-zA-Z]", "", val_str)
    elif val_str.lower().endswith("bn") or val_str.lower().endswith("mrd"):
        mult = 1_000_000_000.0
        val_str = re.sub(r"[a-zA-Z]", "", val_str)
    else:
        val_str = re.sub(r"[^\d.]", "", val_str)
        
    try:
        return float(val_str) * mult if val_str else 0.0
    except ValueError:
        return 0.0

def collect_data(limit_per_club=None, max_clubs_per_league=None, target_league=None):
    os.makedirs("data", exist_ok=True)
    players_data = []
    transfers_data = []
    
    comps_to_process = TARGET_COMPETITIONS
    if target_league:
        comps_to_process = [c for c in TARGET_COMPETITIONS if target_league.lower() in c["name"].lower() or target_league.lower() == c["id"].lower()]
        
    logger.info(f"Starting FootyDex Data Collection across {len(comps_to_process)} competitions...")
    
    for comp in comps_to_process:
        comp_name = comp["name"]
        comp_id = comp["id"]
        logger.info(f"\n--- Processing Competition: {comp_name} ({comp_id}) ---")
        
        clubs_resp = make_request(f"{BASE_URL}/competitions/{comp_id}/clubs")
        if not clubs_resp:
            logger.warning(f"Could not retrieve clubs for {comp_name}")
            continue
            
        clubs = clubs_resp.get("clubs", clubs_resp if isinstance(clubs_resp, list) else [])
        if max_clubs_per_league:
            clubs = clubs[:max_clubs_per_league]
            
        logger.info(f"Found {len(clubs)} clubs in {comp_name}.")
        
        for club in tqdm(clubs, desc=f"{comp_name} Clubs"):
            club_id = str(club.get("id", ""))
            club_name = club.get("name", "Unknown Club")
            if not club_id:
                continue
                
            players_resp = make_request(f"{BASE_URL}/clubs/{club_id}/players")
            if not players_resp:
                continue
                
            players = players_resp.get("players", players_resp if isinstance(players_resp, list) else [])
            if limit_per_club:
                players = players[:limit_per_club]
                
            for p in players:
                player_id = str(p.get("id", ""))
                player_name = p.get("name", "Unknown Player")
                if not player_id:
                    continue
                    
                # 1. Profile
                profile_resp = make_request(f"{BASE_URL}/players/{player_id}/profile") or {}
                age = profile_resp.get("age", p.get("age", 25))
                try:
                    age = int(age) if age is not None else 25
                except ValueError:
                    age = 25
                    
                pos_obj = profile_resp.get("position", p.get("position", {}))
                if isinstance(pos_obj, dict):
                    position = pos_obj.get("main", "Midfielder")
                else:
                    position = str(pos_obj) if pos_obj else "Midfielder"
                    
                citizenship = profile_resp.get("citizenship", p.get("nationality", ["Unknown"]))
                nationality = citizenship[0] if isinstance(citizenship, list) and citizenship else str(citizenship)
                
                mv_raw = profile_resp.get("marketValue", p.get("marketValue", 0))
                market_value = parse_currency_to_float(mv_raw)
                
                height = profile_resp.get("height", p.get("height", 180))
                try:
                    height = float(height) if height else 180.0
                except ValueError:
                    height = 180.0
                foot = profile_resp.get("foot", p.get("foot", "right"))
                
                club_obj = profile_resp.get("club", {})
                contract_expires = club_obj.get("contractExpires", "") if isinstance(club_obj, dict) else ""
                
                # Injury history
                injuries_resp = make_request(f"{BASE_URL}/players/{player_id}/injuries") or {}
                injuries_list = injuries_resp.get("injuries", [])
                total_days_injured = 0
                total_games_missed = 0
                for inj in injuries_list:
                    if isinstance(inj, dict):
                        try:
                            total_days_injured += int(inj.get("days", 0) or 0)
                            total_games_missed += int(inj.get("gamesMissed", 0) or 0)
                        except ValueError:
                            pass
                    
                players_data.append({
                    "player_id": player_id,
                    "player_name": player_name,
                    "competition_id": comp_id,
                    "competition_name": comp_name,
                    "club_id": club_id,
                    "club_name": club_name,
                    "age": age,
                    "position": position,
                    "nationality": nationality,
                    "market_value": market_value,
                    "height": height,
                    "foot": foot,
                    "contract_expires": contract_expires,
                    "total_days_injured": total_days_injured,
                    "total_games_missed": total_games_missed,
                })
                
                # 2. Transfers
                transfers_resp = make_request(f"{BASE_URL}/players/{player_id}/transfers") or {}
                transfers_list = transfers_resp.get("transfers", [])
                for tr in transfers_list:
                    t_id = str(tr.get("id", "")) or f"{player_id}_{tr.get('season', '')}_{tr.get('date', '')}"
                    club_from = tr.get("clubFrom", {})
                    club_to = tr.get("clubTo", {})
                    
                    mv_at_t = parse_currency_to_float(tr.get("marketValue", 0))
                    fee_cleaned = parse_currency_to_float(tr.get("fee", 0))
                    
                    transfers_data.append({
                        "transfer_id": t_id,
                        "player_id": player_id,
                        "player_name": player_name,
                        "club_from_id": str(club_from.get("id", "")),
                        "club_from_name": club_from.get("name", "Unknown Club"),
                        "club_to_id": str(club_to.get("id", "")),
                        "club_to_name": club_to.get("name", "Unknown Club"),
                        "transfer_date": tr.get("date", ""),
                        "season": tr.get("season", ""),
                        "market_value_at_transfer": mv_at_t,
                        "transfer_fee": fee_cleaned,
                    })

    # Save CSVs
    df_players = pd.DataFrame(players_data)
    df_transfers = pd.DataFrame(transfers_data)
    
    if os.path.exists("data/players.csv") and not df_players.empty:
        try:
            df_existing_p = pd.read_csv("data/players.csv")
            df_players = pd.concat([df_existing_p, df_players], ignore_index=True).drop_duplicates(subset=["player_id"], keep="last")
        except Exception as e:
            logger.warning(f"Could not merge with existing players.csv: {e}")
            
    if os.path.exists("data/transfers.csv") and not df_transfers.empty:
        try:
            df_existing_t = pd.read_csv("data/transfers.csv")
            df_transfers = pd.concat([df_existing_t, df_transfers], ignore_index=True).drop_duplicates(subset=["transfer_id"], keep="last")
        except Exception as e:
            logger.warning(f"Could not merge with existing transfers.csv: {e}")
            
    df_players.to_csv("data/players.csv", index=False)
    df_transfers.to_csv("data/transfers.csv", index=False)
    
    logger.info(f"\nData Collection Complete! Saved {len(df_players)} players to data/players.csv and {len(df_transfers)} transfers to data/transfers.csv.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect FootyDex football player profile and transfer data.")
    parser.add_argument("--limit-per-club", type=int, default=None, help="Limit number of players per club (for rapid testing)")
    parser.add_argument("--max-clubs", type=int, default=None, help="Limit number of clubs per competition (for rapid testing)")
    parser.add_argument("--league", type=str, default=None, help="Filter by competition name or ID (e.g. 'Premier League' or 'GB1')")
    args = parser.parse_args()
    
    collect_data(limit_per_club=args.limit_per_club, max_clubs_per_league=args.max_clubs, target_league=args.league)
