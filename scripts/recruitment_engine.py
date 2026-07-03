#!/usr/bin/env python3
"""
FootyDex — Football Transfer Intelligence Dashboard (v2.0)
Recruitment Intelligence Engine:
Implements Feature Engineering, Feature Store, the 3 Evaluation Pillars,
K-Means Tactical Clustering, Cosine Similarity for Cheaper Alternatives,
Replacement Value, Confidence, and Percentile-Backed Explainability.
"""

import os
import json
import math
import logging
import argparse
import datetime
import unicodedata
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FootyDex.RecruitmentEngine")

def load_configs():
    config_dir = "config"
    def load_json(name, default):
        path = os.path.join(config_dir, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default
        
    league_strength = load_json("league_strength.json", {"Premier League": 1.0, "default": 0.75})
    pos_weights = load_json("position_weights.json", {})
    market_weights = load_json("market_weights.json", {})
    return league_strength, pos_weights, market_weights

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8")
    return n.lower().strip()

def map_broad_position(pos_str):
    pos = str(pos_str).lower()
    if "goal" in pos or "gk" in pos:
        return "Goalkeeper"
    elif "centre-back" in pos or "center back" in pos or "cb" in pos:
        return "Centre-Back"
    elif "back" in pos or "fb" in pos or "wb" in pos or "left-back" in pos or "right-back" in pos:
        return "Full-Back"
    elif "defensive midfield" in pos or "cdm" in pos or "dm" in pos:
        return "Defensive Midfield"
    elif "attacking midfield" in pos or "cam" in pos or "am" in pos:
        return "Attacking Midfield"
    elif "central midfield" in pos or "midfield" in pos or "cm" in pos:
        return "Central Midfield"
    elif "wing" in pos or "left winger" in pos or "right winger" in pos:
        return "Winger"
    elif "forward" in pos or "striker" in pos or "st" in pos or "cf" in pos:
        return "Striker"
    return "Central Midfield"

def calculate_contract_years(contract_str):
    if not contract_str or not isinstance(contract_str, str):
        return 2.5
    try:
        parts = contract_str.split("-")
        if len(parts) >= 1 and len(parts[0]) == 4:
            exp_year = int(parts[0])
            exp_month = int(parts[1]) if len(parts) >= 2 else 6
            now = datetime.datetime.now()
            exp_date = datetime.datetime(exp_year, exp_month, min(28, 30))
            diff_years = (exp_date - now).days / 365.25
            return max(0.1, round(diff_years, 2))
    except Exception:
        pass
    return 2.5

def run_recruitment_engine():
    logger.info("Starting FootyDex v2.0 Recruitment Intelligence Engine...")
    league_strength, pos_weights, market_weights = load_configs()
    
    # Load Transfermarkt Players
    tm_path = "data/players.csv"
    if not os.path.exists(tm_path):
        tm_path = "data/moneyball_players.csv"
    if not os.path.exists(tm_path):
        logger.error("No Transfermarkt players dataset found in data/.")
        return
        
    df_tm = pd.read_csv(tm_path)
    logger.info(f"Loaded {len(df_tm)} players from {tm_path}.")
    
    # Load FBref Stats
    fb_path = "data/fbref_stats.csv"
    if os.path.exists(fb_path):
        df_fb = pd.read_csv(fb_path)
        logger.info(f"Loaded {len(df_fb)} multi-table FBref records.")
    else:
        logger.warning("No fbref_stats.csv found. Operating on Transfermarkt biometrics only.")
        df_fb = pd.DataFrame()
        
    # Merge on normalized player name
    df_tm["norm_name"] = df_tm["player_name"].apply(normalize_name)
    if not df_fb.empty and "player_name" in df_fb.columns:
        df_fb["norm_name"] = df_fb["player_name"].apply(normalize_name)
        # Drop duplicate names in fbref
        df_fb_uniq = df_fb.drop_duplicates(subset=["norm_name"], keep="first")
        df_merged = pd.merge(df_tm, df_fb_uniq, on="norm_name", how="left", suffixes=("", "_fb"))
    else:
        df_merged = df_tm.copy()
        
    if "club" not in df_merged.columns:
        df_merged["club"] = df_merged.get("club_name", "Unknown Club")
    else:
        df_merged["club"] = df_merged["club"].fillna(df_merged.get("club_name", "Unknown Club"))
        
    # Fill defaults for missing numerical columns
    feature_cols = ["minutes_played", "goals", "assists", "xg", "xag", "npxg", "prg_carries", "prg_passes", 
                    "pass_cmp_pct", "prg_pass_dist", "long_pass_cmp_pct", "final_third_passes", "key_passes", 
                    "through_balls", "tackles_won", "def_duels_won_pct", "blocks", "interceptions", "clearances", 
                    "errors_to_shot", "touches_in_box", "successful_dribbles", "dribble_success_pct", "sca", "gca", 
                    "aerial_won_pct", "recoveries", "shots_on_target_pct", "shot_conversion_pct"]
                    
    for col in feature_cols:
        if col not in df_merged.columns:
            df_merged[col] = 0.0
        else:
            df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce").fillna(0.0)
            
    df_merged["broad_pos"] = df_merged["position"].apply(map_broad_position)
    df_merged["contract_years"] = df_merged["contract_expires"].apply(calculate_contract_years) if "contract_expires" in df_merged.columns else 2.5
    if "total_days_injured" not in df_merged.columns:
        df_merged["total_days_injured"] = 0.0
    else:
        df_merged["total_days_injured"] = pd.to_numeric(df_merged["total_days_injured"], errors="coerce").fillna(0.0)
    if "market_value" not in df_merged.columns:
        df_merged["market_value"] = 1000000.0
    else:
        df_merged["market_value"] = pd.to_numeric(df_merged["market_value"], errors="coerce").fillna(1000000.0)
    
    # ---------------------------------------------------------
    # 1. FEATURE STORE NORMALIZATION (Z-Scores by Position)
    # ---------------------------------------------------------
    logger.info("Computing Feature Store Z-Scores and Percentiles by Position...")
    z_cols = []
    pct_cols = {}
    
    for pos_group, group in df_merged.groupby("broad_pos"):
        idx = group.index
        for col in feature_cols:
            vals = group[col].values
            std = np.std(vals)
            mean = np.mean(vals)
            z_vals = (vals - mean) / (std if std > 1e-5 else 1.0)
            z_col_name = f"z_{col}"
            if z_col_name not in df_merged.columns:
                df_merged[z_col_name] = 0.0
            df_merged.loc[idx, z_col_name] = z_vals
            
            # Percentiles
            ranks = pd.Series(vals).rank(pct=True).values * 100.0
            pct_col_name = f"pct_{col}"
            if pct_col_name not in df_merged.columns:
                df_merged[pct_col_name] = 50.0
            df_merged.loc[idx, pct_col_name] = np.round(ranks, 1)
            
    df_merged = df_merged.copy()
    
    # ---------------------------------------------------------
    # 2. PILLAR 1: FOOTBALLING ABILITY SCORE (0–100)
    # ---------------------------------------------------------
    logger.info("Executing Pillar 1: Footballing Ability & Execution Score...")
    ability_scores = []
    for _, row in df_merged.iterrows():
        b_pos = row["broad_pos"]
        weights = pos_weights.get(b_pos, pos_weights.get("Central Midfield", {}))
        
        score = 50.0
        # Primary additions
        for p_col in weights.get("primary", []):
            score += row.get(f"z_{p_col}", 0.0) * 8.5
        # Secondary additions
        for s_col in weights.get("secondary", []):
            score += row.get(f"z_{s_col}", 0.0) * 4.0
        # Negative penalties
        for n_col in weights.get("negative", []):
            score -= abs(row.get(f"z_{n_col}", 0.0)) * 6.0
            
        # Add form / xG overperformance bonus
        xg_diff = row["goals"] - row["xg"]
        score += np.clip(xg_diff * 2.0, -5.0, 8.0)
        
        ability_scores.append(np.clip(score, 15.0, 99.0))
    df_merged["ability_score"] = np.round(ability_scores, 1)
    
    # ---------------------------------------------------------
    # 3. PILLAR 2: CONTEXT SCORE (0–100)
    # ---------------------------------------------------------
    logger.info("Executing Pillar 2: Context & Environment Score...")
    context_scores = []
    for _, row in df_merged.iterrows():
        comp_name = str(row.get("competition_name", row.get("league", "")))
        l_mult = league_strength.get("default", 0.75)
        for k, v in league_strength.items():
            if k == "default":
                continue
            if k.lower() in comp_name.lower():
                l_mult = v
                break
        
        # Base context from league multiplier and minutes played reliability (with Big Club rotation cushion)
        mins = row.get("minutes_played", 0)
        is_elite_club = any(ec in str(row.get("club", "")).lower() or ec in str(row.get("club_name", "")).lower() for ec in ["barcelona", "madrid", "bayern", "psg", "paris", "city", "arsenal", "liverpool", "chelsea", "united", "inter", "milan", "juve", "dortmund", "atletico", "leverkusen"])
        min_divisor = 900.0 if is_elite_club else 1800.0
        min_factor = min(1.0, mins / min_divisor) if mins > 0 else (0.75 if is_elite_club else 0.5)
        c_score = (l_mult * 75.0) + (min_factor * 20.0)
        context_scores.append(np.clip(c_score, 20.0, 98.0))
    df_merged["context_score"] = np.round(context_scores, 1)
    
    # ---------------------------------------------------------
    # 4. PILLAR 3: MARKET & LEVERAGE SCORE (0–100)
    # ---------------------------------------------------------
    logger.info("Executing Pillar 3: Market & Leverage Score...")
    market_scores = []
    fair_val_lows = []
    fair_val_highs = []
    
    for _, row in df_merged.iterrows():
        # Contract leverage
        c_yrs = row["contract_years"]
        c_mult = 1.10
        for k_range, mult in market_weights.get("contract_leverage_curve", {}).items():
            low_r, high_r = [float(x) for x in k_range.split("_")]
            if low_r <= c_yrs < high_r:
                c_mult = mult
                break
            
        # Age curve
        age = row["age"]
        age_mult = 1.00
        for k_range, val_dict in market_weights.get("age_potential_curve", {}).items():
            low_a, high_a = [int(x) for x in k_range.split("_")]
            if low_a <= age <= high_a:
                age_mult = val_dict.get("multiplier", 1.00)
                break
            
        # Injury penalty
        inj_days = row["total_days_injured"]
        inj_config = market_weights.get("injury_penalties", {})
        inj_pen = min(25.0, inj_days * inj_config.get("penalty_factor_per_day", 0.0015) * 100.0)
        
        # Selling club leverage
        club_name = str(row["club"])
        sell_config = market_weights.get("selling_club_leverage", {})
        sell_mult = sell_config.get("tier_1_multiplier", 1.15) if club_name in sell_config.get("tier_1_extractors", []) else sell_config.get("default_multiplier", 1.00)
        
        m_score = (70.0 * c_mult * age_mult * (2.0 - sell_mult)) - inj_pen
        market_scores.append(np.clip(m_score, 15.0, 99.0))
        
        # Fair Market Valuation Range (€)
        # Baseline fair val derived from Ability, Context, Age, and Contract
        mv = row["market_value"]
        is_elite = any(ec in str(row.get("club", "")).lower() or ec in str(row.get("club_name", "")).lower() for ec in ["barcelona", "madrid", "bayern", "psg", "paris", "city", "arsenal", "liverpool", "chelsea", "united", "inter", "milan", "juve", "dortmund", "atletico", "leverkusen"])
        elite_mult = 1.25 if is_elite else 1.0
        ability_factor = ((row["ability_score"] / 60.0) ** 1.1) * elite_mult
        est_fair = mv * ability_factor * (c_mult ** 0.5) * (age_mult ** 0.5)
        
        low_val = round((est_fair * 0.90) / 1e6, 1)
        high_val = round((est_fair * 1.15) / 1e6, 1)
        fair_val_lows.append(max(0.5, low_val))
        fair_val_highs.append(max(0.8, high_val))
        
    df_merged["market_score"] = np.round(market_scores, 1)
    df_merged["fair_val_low"] = fair_val_lows
    df_merged["fair_val_high"] = fair_val_highs
    
    # ---------------------------------------------------------
    # 5. RECRUITMENT INDEX (RI) & REPLACEMENT VALUE
    # ---------------------------------------------------------
    logger.info("Calculating Recruitment Index (RI) and Replacement Values...")
    ri_vals = (df_merged["ability_score"] * 0.45) + (df_merged["context_score"] * 0.25) + (df_merged["market_score"] * 0.30)
    df_merged["recruitment_index"] = np.round(np.clip(ri_vals, 10.0, 99.4), 1)
    
    # Replacement Value (+X above position average)
    rep_vals = []
    for pos_group, group in df_merged.groupby("broad_pos"):
        avg_ri = group["recruitment_index"].mean()
        for idx in group.index:
            delta = df_merged.loc[idx, "recruitment_index"] - avg_ri
            rep_vals.append((idx, round(delta, 1), round(avg_ri, 1)))
    rep_df = pd.DataFrame(rep_vals, columns=["index", "replacement_value", "pos_avg_ri"]).set_index("index")
    df_merged["replacement_value"] = rep_df["replacement_value"]
    df_merged["pos_avg_ri"] = rep_df["pos_avg_ri"]
    
    # ---------------------------------------------------------
    # 6. TACTICAL CLUSTERING ENGINE (Unsupervised KMeans)
    # ---------------------------------------------------------
    logger.info("Executing Tactical Profile Discovery Engine (K-Means Clustering)...")
    cluster_features = ["z_prg_passes", "z_prg_carries", "z_tackles_won", "z_interceptions", "z_sca", "z_xg", "z_pass_cmp_pct", "z_aerial_won_pct"]
    X_cluster = df_merged[[c for c in cluster_features if c in df_merged.columns]].fillna(0.0).values
    
    n_clusters = min(13, max(3, len(df_merged) // 5))
    if len(df_merged) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_cluster)
        df_merged["cluster_id"] = kmeans.labels_
    else:
        df_merged["cluster_id"] = 0
        
    # Map clusters to tactical archetype labels based on high z-scores
    cluster_labels = {}
    for c_id, group in df_merged.groupby("cluster_id"):
        mean_prg_p = group["z_prg_passes"].mean()
        mean_prg_c = group["z_prg_carries"].mean()
        mean_def = (group["z_tackles_won"].mean() + group["z_interceptions"].mean()) / 2.0
        mean_att = (group["z_sca"].mean() + group["z_xg"].mean()) / 2.0
        mean_aer = group["z_aerial_won_pct"].mean()
        
        dom_pos = group["broad_pos"].mode()[0] if not group["broad_pos"].empty else "Central Midfield"
        
        if dom_pos == "Goalkeeper":
            label = "Sweeper-Keeper / Modern Goalie"
        elif dom_pos == "Centre-Back":
            label = "Ball-Playing Centre-Back" if mean_prg_p > 0.3 or mean_prg_c > 0.3 else "Traditional Stopper CB / Aerial Dominator"
        elif dom_pos == "Full-Back":
            label = "Inverted Full-Back / Midfield Support" if mean_prg_p > 0.2 else "Attacking Wing-Back / Touchline Flyer"
        elif dom_pos == "Defensive Midfield":
            label = "Deep-Lying Playmaker / Regista" if mean_prg_p > 0.4 else "Ball-Winning Destroyer / Anchor"
        elif dom_pos in ["Central Midfield", "Attacking Midfield"]:
            label = "Advanced Playmaker / Creator" if mean_att > 0.4 else "Box-to-Box Engine / Transition Dynamo"
        elif dom_pos == "Winger":
            label = "Inverted Scoring Winger / Inside Forward" if mean_att > 0.3 else "Wide Touchline Creator / Dribbler"
        else:
            label = "Complete Striker / False 9" if mean_prg_p > 0.1 else "Poacher / Box Predator"
        cluster_labels[c_id] = f"{label} (Cluster {c_id})"
        
    df_merged["tactical_profile"] = df_merged["cluster_id"].map(cluster_labels)
    
    # ---------------------------------------------------------
    # 7. SIMILARITY ENGINE & CHEAPER ALTERNATIVES
    # ---------------------------------------------------------
    logger.info("Executing Cosine Similarity Engine for Similar Profiles & Budget Alternatives...")
    sim_features = ["z_minutes_played", "z_goals", "z_assists", "z_xg", "z_xag", "z_prg_carries", "z_prg_passes", 
                    "z_pass_cmp_pct", "z_final_third_passes", "z_tackles_won", "z_interceptions", "z_sca", "z_recoveries"]
    X_sim = df_merged[[c for c in sim_features if c in df_merged.columns]].fillna(0.0).values
    
    sim_matrix = cosine_similarity(X_sim)
    
    most_similar_list = []
    cheaper_alt_list = []
    
    records = df_merged[["player_name", "club", "broad_pos", "market_value"]].to_dict('records')
    for idx in range(len(records)):
        row = records[idx]
        target_pos = row["broad_pos"]
        target_mv = row["market_value"]
        
        scores = sim_matrix[idx]
        # Sort indices by similarity descending, excluding self
        sorted_indices = np.argsort(scores)[::-1]
        
        sims = []
        alts = []
        for s_idx in sorted_indices:
            if s_idx == idx:
                continue
            cand = records[s_idx]
            if cand["broad_pos"] != target_pos:
                continue
                
            match_pct = round(scores[s_idx] * 100.0, 1)
            cand_mv_m = round(cand["market_value"] / 1e6, 1)
            cand_str = f"{cand['player_name']} ({cand['club']} | €{cand_mv_m}M | {match_pct}% Match)"
            
            if len(sims) < 3:
                sims.append(cand_str)
                
            # Check cheaper alternative condition: at least €15M cheaper or < 60% valuation
            if (target_mv - cand["market_value"]) >= 15000000 or (target_mv > 15000000 and cand["market_value"] <= target_mv * 0.65):
                savings_m = round((target_mv - cand["market_value"]) / 1e6, 1)
                alt_str = f"{cand['player_name']} ({cand['club']} | €{cand_mv_m}M | {match_pct}% Match | 👉 €{savings_m}M Savings)"
                if len(alts) < 2:
                    alts.append(alt_str)
                    
            if len(sims) >= 3 and len(alts) >= 2:
                break
                    
        most_similar_list.append(" • ".join(sims) if sims else "No direct positional matches found")
        cheaper_alt_list.append(" • ".join(alts) if alts else "No significant budget alternatives found")
        
    df_merged["similar_players"] = most_similar_list
    df_merged["cheaper_alternatives"] = cheaper_alt_list
    
    # ---------------------------------------------------------
    # 8. CONFIDENCE ENGINE, DATA QUALITY & RECOMMENDATION
    # ---------------------------------------------------------
    logger.info("Executing Confidence Engine, System Fit, and Two-Axis Recommendations...")
    conf_scores = []
    data_quality = []
    recommendations = []
    risk_profiles = []
    system_fits = []
    explainability_notes = []
    
    for _, row in df_merged.iterrows():
        # Confidence
        mins = row.get("minutes_played", 0)
        c_pct = 95 if mins >= 2200 else (88 if mins >= 1400 else (74 if mins >= 800 else 54))
        conf_scores.append(f"{c_pct}%")
        
        # Data Quality
        has_fb = row.get("minutes_played", 0) > 0
        has_inj = pd.notna(row.get("total_days_injured")) and row.get("total_days_injured", -1) >= 0
        has_cont = row.get("contract_years", 0) != 2.5
        dq = 98 if (has_fb and has_inj and has_cont) else (92 if has_fb else 75)
        
        fb_mark = "✓" if has_fb else "✗"
        cont_mark = "✓" if has_cont else "✗"
        inj_mark = "✓" if has_inj else "✗"
        data_quality.append(f"{dq}% Complete (FBref {fb_mark} | Contract {cont_mark} | Injuries {inj_mark})")
        
        # Risk Axis
        inj_d = row.get("total_days_injured", 0)
        c_y = row.get("contract_years", 2.5)
        if inj_d > 60 or c_y < 1.0 or c_pct < 60:
            risk = "🔴 HIGH RISK"
        elif inj_d > 25 or c_y < 1.8:
            risk = "🟡 MEDIUM RISK"
        else:
            risk = "🟢 LOW RISK"
        risk_profiles.append(risk)
        
        # Purchase Value Recommendation
        ri = row["recruitment_index"]
        mv_m = row["market_value"] / 1e6
        fair_h = row["fair_val_high"]
        
        is_elite = any(ec in str(row.get("club", "")).lower() or ec in str(row.get("club_name", "")).lower() for ec in ["barcelona", "madrid", "bayern", "psg", "paris", "city", "arsenal", "liverpool", "chelsea", "united", "inter", "milan", "juve", "dortmund", "atletico", "leverkusen"])
        if (ri >= 82.0 and fair_h >= mv_m * 0.95) or (is_elite and ri >= 72.0 and fair_h >= mv_m * 0.9):
            rec = "🟢 ELITE TARGET"
        elif (ri >= 72.0 and fair_h >= mv_m * 0.85) or (is_elite and ri >= 64.0 and fair_h >= mv_m * 0.8):
            rec = "🟢 GOOD VALUE"
        elif ri >= 58.0 or fair_h >= mv_m * 0.75:
            rec = "🟡 FAIR VALUE"
        else:
            rec = "🔴 OVERPRICED / AVOID"
        recommendations.append(rec)
        
        # System Fit Stars
        prg_p_pct = row.get("pct_prg_passes", 50.0)
        press_r = row.get("pct_pass_cmp_pct", 50.0)
        tkl_pct = row.get("pct_tackles_won", 50.0)
        
        stars_poss = "★★★★★" if prg_p_pct > 80 else ("★★★★☆" if prg_p_pct > 60 else "★★★☆☆")
        stars_press = "★★★★★" if tkl_pct > 80 else ("★★★★☆" if tkl_pct > 60 else "★★★☆☆")
        system_fits.append(f"Possession Build-Up: {stars_poss} | High Pressing: {stars_press}")
        
        # Percentile-Backed Explainability (+) and (-)
        pos_notes = []
        neg_notes = []
        
        if row.get("pct_prg_passes", 0) >= 85:
            pos_notes.append(f"[+] Elite ball progression ({row['pct_prg_passes']}th percentile Progressive Passes)")
        if row.get("pct_tackles_won", 0) >= 85 or row.get("pct_interceptions", 0) >= 85:
            pos_notes.append(f"[+] Outstanding defensive work rate ({max(row.get('pct_tackles_won',0), row.get('pct_interceptions',0))}th percentile Interceptions/Tackles)")
        if row.get("pct_sca", 0) >= 85:
            pos_notes.append(f"[+] World-class shot creation ({row['pct_sca']}th percentile Shot-Creating Actions)")
        if row["replacement_value"] >= 10.0:
            pos_notes.append(f"[+] High replacement value (+{row['replacement_value']} RI above average starter)")
        if row["contract_years"] >= 3.0:
            pos_notes.append(f"[+] Strong contractual leverage ({row['contract_years']} years remaining)")
            
        if not pos_notes:
            pos_notes.append(f"[+] Reliable squad contributor ({row.get('pct_minutes_played', 50)}th percentile Minutes Played)")
            
        if row["contract_years"] < 1.2:
            neg_notes.append(f"[-] Expiring contract volatility ({row['contract_years']} years remaining — loss of resale value)")
        if row.get("total_days_injured", 0) >= 30:
            neg_notes.append(f"[-] Availability deduction (Sidelined {int(row['total_days_injured'])} days in recent seasons)")
        if row.get("pct_errors_to_shot", 0) >= 75:
            neg_notes.append(f"[-] Prone to defensive errors ({row['pct_errors_to_shot']}th percentile Errors Leading to Shot)")
        if not neg_notes:
            neg_notes.append("[-] No critical physical or contractual flags identified")
            
        full_note = "\n".join(pos_notes[:3] + neg_notes[:2])
        explainability_notes.append(full_note)
        
    df_merged["confidence_score"] = conf_scores
    df_merged["data_quality"] = data_quality
    df_merged["recommendation"] = recommendations
    df_merged["risk_profile"] = risk_profiles
    df_merged["system_fit"] = system_fits
    df_merged["explainability"] = explainability_notes
    
    # Save Enriched Output
    out_file = "data/recruitment_index.csv"
    df_merged.to_csv(out_file, index=False)
    logger.info(f"Successfully generated FootyDex v2.0 Recruitment Index dataset with {len(df_merged)} players -> {out_file}!")

if __name__ == "__main__":
    run_recruitment_engine()
