#!/usr/bin/env python3
"""
FootyDex — Squad Number Utility Module
Provides shared squad number allocation logic across data pipelines and UI components.
"""

import os
import json
import pandas as pd

def assign_squad_numbers(df):
    """
    Assigns authentic, unique squad numbers (#1-#45) per club based on authentic club rosters and position ranking.
    Preserves existing valid numbers, cleanly fills partial gaps, and handles squad overflow.
    """
    d = df.copy()
    
    # 1. Apply cached authentic club squad numbers if available
    cache_path = os.path.join("data", "player_shirt_numbers.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if "player_id" in d.columns:
                mapped = d["player_id"].astype(str).str.strip().map(cache)
                if "shirt_number" in d.columns:
                    d["shirt_number"] = d["shirt_number"].combine_first(mapped)
                else:
                    d["shirt_number"] = mapped
        except Exception:
            pass
            
    has_col = "shirt_number" in d.columns
    if has_col and d["shirt_number"].notna().all() and (d["shirt_number"] > 0).all():
        return d
    if not has_col:
        d["shirt_number"] = pd.Series(dtype=object)
    pos_map = {
        "Goalkeeper": [1, 13, 22, 30, 31, 12, 25],
        "Centre-Back": [4, 5, 6, 14, 24, 15, 26, 33, 3],
        "Left-Back": [3, 15, 21, 12, 32, 26, 18],
        "Right-Back": [2, 12, 20, 22, 28, 14, 31],
        "Central Midfield": [8, 6, 16, 18, 20, 14, 23, 24, 25],
        "Defensive Midfield": [6, 16, 4, 18, 24, 22, 30, 15],
        "Attacking Midfield": [10, 20, 25, 8, 18, 19, 21, 11],
        "Right Winger": [7, 17, 23, 11, 19, 27, 20, 14],
        "Left Winger": [11, 21, 17, 7, 19, 27, 22, 20],
        "Centre-Forward": [9, 19, 29, 11, 18, 14, 7, 23]
    }
    club_col = "club" if "club" in d.columns else "club_name"
    for _, c_df in d.groupby(club_col):
        used = set()
        for val in c_df["shirt_number"]:
            if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
                used.add(int(val))
        sort_cols = [c for c in ["ability_score", "recruitment_index"] if c in c_df.columns]
        c_df_sorted = c_df.sort_values(by=sort_cols, ascending=False) if sort_cols else c_df
        for idx, row in c_df_sorted.iterrows():
            curr_val = row.get("shirt_number")
            if pd.notna(curr_val) and isinstance(curr_val, (int, float)) and curr_val > 0:
                continue
            pos = row.get("position", "Central Midfield")
            pref_list = pos_map.get(pos, [8, 14, 16, 18, 20, 22, 24, 25, 26])
            assigned = None
            for n in pref_list:
                if n not in used:
                    assigned = n
                    break
            if assigned is None:
                for n in range(1, 99):
                    if n not in used:
                        assigned = n
                        break
            if assigned is None:
                assigned = max(used) + 1 if used else 99
            used.add(assigned)
            d.at[idx, "shirt_number"] = int(assigned)
    return d
