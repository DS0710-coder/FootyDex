#!/usr/bin/env python3
"""
FootyDex — Football Transfer Intelligence Dashboard
Streamlit Application: Narrative-first scouting briefing cards, Cosine Similarity alternatives,
Interactive Radar comparisons, and Recruitment Index (RI) leaderboards.
"""

import os
import html
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="FootyDex | Recruitment Intelligence",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

if os.path.exists("assets/logo.png") and hasattr(st, "logo"):
    st.logo("assets/logo.png")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B0E14 !important;
    color: #E2E8F0;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

.main-header {
    background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #6B11FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.sub-header {
    color: #94A3B8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: rgba(22, 27, 34, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(12px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-bottom: 1rem;
}
.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 242, 254, 0.4);
    box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.15);
}

.metric-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 0.5rem;
    font-weight: 600;
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: #F8FAFC;
    font-family: 'Outfit', sans-serif;
}

.badge {
    display: inline-block;
    padding: 0.35em 0.8em;
    font-size: 0.85rem;
    font-weight: 700;
    line-height: 1;
    text-align: center;
    white-space: nowrap;
    vertical-align: baseline;
    border-radius: 8px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 0.5rem;
}

.badge-elite { background: rgba(56, 239, 125, 0.15); color: #38EF7D; border: 1px solid rgba(56, 239, 125, 0.3); }
.badge-good { background: rgba(0, 242, 254, 0.15); color: #00F2FE; border: 1px solid rgba(0, 242, 254, 0.3); }
.badge-fair { background: rgba(254, 225, 64, 0.15); color: #FEE140; border: 1px solid rgba(254, 225, 64, 0.3); }
.badge-avoid { background: rgba(255, 8, 68, 0.15); color: #FF0844; border: 1px solid rgba(255, 8, 68, 0.3); }

.briefing-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border-left: 5px solid #00F2FE;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}
.explain-pos { color: #38EF7D; font-weight: 500; margin-bottom: 0.4rem; }
.explain-neg { color: #FF6B6B; font-weight: 500; margin-bottom: 0.4rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_data():
    ri_path = "data/recruitment_index.csv"
    tm_path = "data/moneyball_players.csv"
    path_to_use = ri_path if os.path.exists(ri_path) else (tm_path if os.path.exists(tm_path) else None)
    
    if not path_to_use:
        return pd.DataFrame(), pd.DataFrame()
        
    df = pd.read_csv(path_to_use)
    
    # Ensure standard display columns
    if "recruitment_index" not in df.columns:
        df["recruitment_index"] = df.get("moneyball_score", 50.0)
    if "ability_score" not in df.columns:
        df["ability_score"] = df["recruitment_index"]
    if "context_score" not in df.columns:
        df["context_score"] = 75.0
    if "market_score" not in df.columns:
        df["market_score"] = 70.0
    if "recommendation" not in df.columns:
        df["recommendation"] = df.get("moneyball_label", "🟡 FAIR VALUE")
    if "risk_profile" not in df.columns:
        df["risk_profile"] = "🟢 LOW RISK"
    if "fair_val_low" not in df.columns:
        df["fair_val_low"] = round(df.get("market_value", 1000000) * 0.9 / 1e6, 1)
    if "fair_val_high" not in df.columns:
        df["fair_val_high"] = round(df.get("market_value", 1000000) * 1.15 / 1e6, 1)
    if "tactical_profile" not in df.columns:
        df["tactical_profile"] = df.get("position", "Midfielder")
    if "explainability" not in df.columns:
        df["explainability"] = "[+] Consistent squad option\n[-] Standard market valuation"
    if "similar_players" not in df.columns:
        df["similar_players"] = "No similarity data generated"
    if "cheaper_alternatives" not in df.columns:
        df["cheaper_alternatives"] = "No budget alternatives found"
    if "system_fit" not in df.columns:
        df["system_fit"] = "Possession Build-Up: ★★★★☆ | High Pressing: ★★★★☆"
    if "shirt_number" not in df.columns:
        pos_map = {"Goalkeeper": 1, "Right-Back": 2, "Left-Back": 3, "Centre-Back": 4, "Central Midfield": 8, "Defensive Midfield": 6, "Attacking Midfield": 10, "Right Winger": 7, "Left Winger": 11, "Centre-Forward": 9}
        df["shirt_number"] = df["position"].map(pos_map).fillna(18).astype(int)
        
    df["mv_display"] = df["market_value"].apply(lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else f"€{x/1e3:.0f}K")
    
    trans_path = "data/transfers.csv"
    df_trans = pd.read_csv(trans_path) if os.path.exists(trans_path) else pd.DataFrame()
    return df, df_trans

def get_badge_html(label):
    lbl = str(label).upper()
    if "ELITE" in lbl:
        return f'<span class="badge badge-elite">{label}</span>'
    elif "GOOD" in lbl:
        return f'<span class="badge badge-good">{label}</span>'
    elif "AVOID" in lbl or "OVER" in lbl or "HIGH" in lbl:
        return f'<span class="badge badge-avoid">{label}</span>'
    else:
        return f'<span class="badge badge-fair">{label}</span>'

def select_club_cb(club_name):
    st.session_state["exp_selected_club"] = club_name

def main():
    st.markdown('<div class="main-header">FootyDex Recruitment Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">"If your club had €50M to spend today... who is the smartest signing?" • Proprietary Transfer Valuation & Decision Support</div>', unsafe_allow_html=True)
    
    df, df_transfers = load_data()
    if df.empty:
        st.warning("⚠️ No player dataset found. Please run `python3 scripts/recruitment_engine.py` or `collect_data.py` first.")
        return
        
    # Sidebar Filters
    with st.sidebar:
        if os.path.exists("assets/logo.png"):
            st.image("assets/logo.png", use_container_width=True)
            st.markdown("---")
        st.markdown("### ⚙️ Recruitment Scope")
        leagues = sorted(df["competition_name"].dropna().unique())
        sel_leagues = st.multiselect("Filter Competitions", options=leagues, default=leagues)
        
        positions = sorted(df["position"].dropna().unique())
        sel_positions = st.multiselect("Filter Positions", options=positions, default=positions)
        
        recs = sorted(df["recommendation"].dropna().unique())
        sel_recs = st.multiselect("Strategic Recommendation", options=recs, default=recs)
        
        min_ri = st.slider("Min Recruitment Index (RI)", 10.0, 99.0, 50.0, step=1.0)
        max_price = st.slider("Max Market Valuation (€M)", 1.0, 200.0, 150.0, step=5.0)
        
    # Build unique player display labels once, before filtering, so _label is
    # available in all tabs and f_df inherits it via the shared df reference.
    df["_label"] = df["player_name"] + " (" + df["club_name"].fillna("") + ")"

    # Filter dataset
    f_df = df[
        (df["competition_name"].isin(sel_leagues)) &
        (df["position"].isin(sel_positions)) &
        (df["recommendation"].isin(sel_recs)) &
        (df["recruitment_index"] >= min_ri) &
        ((df["market_value"] / 1e6) <= max_price)
    ]
    
    # Top KPI Bar
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Scouted Targets</div><div class="metric-value">{len(f_df)}</div></div>', unsafe_allow_html=True)
    with kpi2:
        elite_cnt = len(f_df[f_df["recommendation"].str.contains("ELITE", na=False)])
        st.markdown(f'<div class="metric-card"><div class="metric-label">Elite Targets Flagged</div><div class="metric-value" style="color:#38EF7D;">{elite_cnt}</div></div>', unsafe_allow_html=True)
    with kpi3:
        avg_ri = f_df["recruitment_index"].mean() if not f_df.empty else 0
        st.markdown(f'<div class="metric-card"><div class="metric-label">Average RI Score</div><div class="metric-value" style="color:#00F2FE;">{avg_ri:.1f}</div></div>', unsafe_allow_html=True)
    with kpi4:
        bargain_cnt = len(f_df[f_df["recommendation"].str.contains("GOOD|ELITE", na=False) & f_df["risk_profile"].str.contains("LOW", na=False)])
        st.markdown(f'<div class="metric-card"><div class="metric-label">Low-Risk Bargains</div><div class="metric-value" style="color:#FEE140;">{bargain_cnt}</div></div>', unsafe_allow_html=True)
        
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🏟️ Squad & Club Explorer",
        "🔎 Executive Narrative Briefing", 
        "📊 Recruitment Index Leaderboard", 
        "⚔️ Interactive Radar & Compare", 
        "📈 Market Valuation vs. RI Analytics"
    ])

    # ==========================================
    # TAB 0: SQUAD & CLUB EXPLORER
    # ==========================================
    with tab0:
        st.markdown("### 🏟️ European Club & Squad Explorer")
        st.markdown("Select a competition below to view its clubs as interactive blocks with crests. Click a club to explore its full squad categorized by position.")
        
        # 1. League Selector
        leagues_avail = sorted(df["competition_name"].dropna().unique())
        sel_comp_exp = st.radio("🏆 Select European League:", options=leagues_avail, horizontal=True, key="exp_comp")
        
        comp_df = df[df["competition_name"] == sel_comp_exp]
        clubs_in_comp = sorted(comp_df["club_name"].dropna().unique())
        
        # Session state for selected club
        if "exp_selected_club" not in st.session_state or st.session_state["exp_selected_club"] not in clubs_in_comp:
            st.session_state["exp_selected_club"] = clubs_in_comp[0] if clubs_in_comp else ""
            
        # 2. Club Blocks Grid with Logos (Click the team button directly!)
        st.markdown(f"#### 🛡️ Clubs in {sel_comp_exp} (Click Team to Open Squad)")
        
        cols_per_row = 5
        for i in range(0, len(clubs_in_comp), cols_per_row):
            row_clubs = clubs_in_comp[i:i+cols_per_row]
            cols = st.columns(cols_per_row)
            for j, c_name in enumerate(row_clubs):
                with cols[j]:
                    c_id_series = comp_df[comp_df["club_name"] == c_name]["club_id"].dropna()
                    c_id = int(c_id_series.iloc[0]) if not c_id_series.empty else 0
                    logo_url = f"https://tmssl.akamaized.net/images/wappen/head/{c_id}.png"
                    
                    is_sel = (c_name == st.session_state.get("exp_selected_club"))
                    btn_type = "primary" if is_sel else "secondary"
                    
                    st.markdown(f"""
                    <div style="text-align:center;margin-bottom:0.2rem;min-height:55px;display:flex;align-items:center;justify-content:center;">
                        <img src="{logo_url}" style="max-height:50px;max-width:80%;" onerror="this.style.display='none'">
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.button(
                        c_name, 
                        key=f"btn_club_{c_id}_{j}_{i}", 
                        on_click=select_club_cb,
                        args=(c_name,),
                        use_container_width=True,
                        type=btn_type
                    )
                        
        st.markdown("---")
        curr_club = st.session_state.get("exp_selected_club", clubs_in_comp[0] if clubs_in_comp else "")
            
        # 3. Squad Directory & Filters
        st.markdown(f"### 👥 {curr_club} Squad Directory")
        
        sq_c1, sq_c2, sq_c3, sq_c4 = st.columns(4)
        with sq_c1:
            sq_search = st.text_input("🔍 Filter Squad (Name / Foot / Nat):", "", key="sq_search").strip()
        with sq_c2:
            sq_max_age = st.slider("Max Age:", 15, 45, 40, key="sq_age_slider")
        with sq_c3:
            sq_max_mv = st.slider("Max Market Value (€M):", 0.0, 250.0, 250.0, step=5.0, key="sq_mv_slider")
        with sq_c4:
            sq_sort = st.selectbox("Sort Squad By:", [
                "Recruitment Index ⭐ (High to Low)",
                "Market Value (High to Low)",
                "Age (Young to Old)",
                "Player Name (A-Z)"
            ], key="sq_sort")
            
        club_squad = comp_df[comp_df["club_name"] == curr_club].copy()
        if sq_search:
            club_squad = club_squad[
                club_squad["player_name"].str.contains(sq_search, case=False, na=False, regex=False) |
                club_squad["foot"].astype(str).str.contains(sq_search, case=False, na=False, regex=False) |
                club_squad["nationality"].astype(str).str.contains(sq_search, case=False, na=False, regex=False)
            ]
        club_squad = club_squad[
            (club_squad["age"] <= sq_max_age) & 
            ((club_squad["market_value"] / 1e6) <= sq_max_mv)
        ]
        
        if "Recruitment Index" in sq_sort:
            club_squad = club_squad.sort_values("recruitment_index", ascending=False)
        elif "Market Value" in sq_sort:
            club_squad = club_squad.sort_values("market_value", ascending=False)
        elif "Age" in sq_sort:
            club_squad = club_squad.sort_values("age", ascending=True)
        else:
            club_squad = club_squad.sort_values("player_name", ascending=True)
            
        # 4. Display by Positional Buckets
        pos_buckets = [
            ("🧤 Goalkeepers", ["Goalkeeper"]),
            ("🛡️ Defenders", ["Centre-Back", "Left-Back", "Right-Back", "Defender"]),
            ("⚙️ Midfielders", ["Central Midfield", "Defensive Midfield", "Attacking Midfield", "Right Midfield", "Left Midfield", "Midfielder"]),
            ("⚡ Attackers", ["Centre-Forward", "Left Winger", "Right Winger", "Second Striker", "Attacker", "Winger"])
        ]
        
        display_cols = ["shirt_number", "player_name", "position", "age", "foot", "mv_display", "contract_expires", "recruitment_index", "recommendation", "risk_profile"]
        valid_cols = [c for c in display_cols if c in club_squad.columns]
        
        for group_title, pos_list in pos_buckets:
            sub_squad = club_squad[club_squad["position"].isin(pos_list)]
            st.markdown(f"#### {group_title} ({len(sub_squad)})")
            if sub_squad.empty:
                st.info("No players found in this category matching current filters.")
            else:
                st.dataframe(
                    sub_squad[valid_cols].rename(columns={
                        "shirt_number": "#", "player_name": "Player Name", "position": "Position", "age": "Age",
                        "foot": "Foot", "mv_display": "Market Value", "contract_expires": "Contract Expires",
                        "recruitment_index": "RI ⭐", "recommendation": "Recommendation", "risk_profile": "Risk"
                    }).style.format({
                        "RI ⭐": "{:.1f}"
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=min(350, 40 + len(sub_squad)*36)
                )

    # ==========================================
    # TAB 1: EXECUTIVE NARRATIVE BRIEFING
    # ==========================================
    with tab1:
        st.markdown("### 📋 Executive Scouting Briefing & Why-To-Buy Analysis")
        player_labels = sorted(df["_label"].unique())
        sel_label = st.selectbox("Select Target Player for Narrative Recruitment Briefing (All Top 5 Leagues):", options=player_labels)

        # Resolve label back to the correct unique row
        p = df[df["_label"] == sel_label].iloc[0]
        
        col_b1, col_b2 = st.columns([1, 1.2])
        with col_b1:
            rec_badge = get_badge_html(p["recommendation"])
            risk_badge = get_badge_html(p["risk_profile"])
            p_name = html.escape(str(p['player_name']))
            c_name = html.escape(str(p['club_name']))
            comp_name = html.escape(str(p['competition_name']))
            t_prof = html.escape(str(p['tactical_profile']))
            dq_str = html.escape(str(p.get('data_quality', '98% Complete')))
            conf_str = html.escape(str(p.get('confidence_score', '95%')))
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin-bottom:0.2rem;color:#00F2FE;">{p_name}</h2>
                <div style="color:#94A3B8;margin-bottom:1rem;font-size:1.1rem;">{c_name} • {comp_name}</div>
                <div style="margin-bottom:1rem;">{rec_badge} {risk_badge}</div>
                <hr style="border-color:rgba(255,255,255,0.1);margin:1rem 0;">
                <p><b>Tactical Archetype:</b> <span style="color:#38EF7D;font-weight:600;">{t_prof}</span></p>
                <p><b>Age:</b> {p['age']} yrs | <b>Contract Remaining:</b> {p.get('contract_years', 2.5)} years</p>
                <p><b>Current Market Valuation:</b> <span style="font-size:1.2rem;font-weight:700;color:#F8FAFC;">{p['mv_display']}</span></p>
                <p><b>Est. Fair Valuation Range:</b> <span style="color:#00F2FE;font-weight:700;font-size:1.2rem;">€{p['fair_val_low']}M – €{p['fair_val_high']}M</span></p>
                <p style="margin-top:0.5rem;font-size:0.9rem;color:#94A3B8;"><b>Data Audit:</b> {dq_str} | <b>Confidence:</b> {conf_str}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col_b2:
            lines = str(p["explainability"]).split("\n")
            explain_items_html = ""
            for l in lines:
                if l.strip().startswith("[+]"):
                    explain_items_html += f'<div class="explain-pos">✔️ {html.escape(l.replace("[+]", "").strip())}</div>'
                elif l.strip().startswith("[-]"):
                    explain_items_html += f'<div class="explain-neg">⚠️ {html.escape(l.replace("[-]", "").strip())}</div>'
                else:
                    explain_items_html += f'<div>• {html.escape(l.strip())}</div>'
                    
            sys_fit = html.escape(str(p.get('system_fit', 'Possession: ★★★★☆')))
            card_html = f"""<div class="briefing-card">
                <h4 style="color:#00F2FE;margin-bottom:1rem;">💡 Why Sign This Player? (Percentile-Backed Rationale)</h4>
                {explain_items_html}
                <hr style="border-color:rgba(255,255,255,0.1);margin:1rem 0;">
                <p style="margin-bottom:0.3rem;"><b>Tactical System Fit:</b></p>
                <code style="color:#FEE140;background:rgba(0,0,0,0.3);padding:0.4rem;border-radius:6px;display:block;">{sys_fit}</code>
            </div>"""
            st.markdown(card_html, unsafe_allow_html=True)
            
        st.markdown("#### 🔍 Cosine Similarity Engine: Cheaper Budget Alternatives & Similar Profiles")
        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            alts = str(p.get("cheaper_alternatives", "No alternatives found")).split(" • ")
            alts_html = "".join([f'<p style="color:#38EF7D;margin:0.5rem 0;">👉 <b>{html.escape(a)}</b></p>' for a in alts])
            st.markdown(f'<div class="metric-card"><h5 style="color:#00F2FE;">💎 Cheaper Budget Alternatives (High Match • Lower Valuation)</h5>{alts_html}</div>', unsafe_allow_html=True)
        with sim_c2:
            sims = str(p.get("similar_players", "No similar players found")).split(" • ")
            sims_html = "".join([f'<p style="color:#E2E8F0;margin:0.5rem 0;">• {html.escape(s)}</p>' for s in sims])
            st.markdown(f'<div class="metric-card"><h5 style="color:#94A3B8;">👥 Most Similar European Profiles</h5>{sims_html}</div>', unsafe_allow_html=True)

    # ==========================================
    # TAB 2: RI LEADERBOARD
    # ==========================================
    with tab2:
        st.markdown("### 🏆 Recruitment Index (RI) Global Leaderboard")
        search_query = st.text_input("🔍 Search Players / Clubs (narrows current filters):", "").strip()
        # Always start from sidebar-filtered f_df so league/position/RI/price scope is preserved;
        # use regex=False so special characters in the query are treated as literals, not regex.
        display_df = f_df
        if search_query:
            display_df = f_df[
                f_df["player_name"].str.contains(search_query, case=False, na=False, regex=False) |
                f_df["club_name"].str.contains(search_query, case=False, na=False, regex=False)
            ]
            
        display_cols = ["shirt_number", "player_name", "club_name", "competition_name", "position", "age", "mv_display", 
                        "recruitment_index", "ability_score", "context_score", "market_score", "recommendation", "risk_profile"]
        valid_cols = [c for c in display_cols if c in display_df.columns]
        
        st.dataframe(
            display_df[valid_cols].sort_values("recruitment_index", ascending=False).rename(columns={
                "shirt_number": "#", "player_name": "Player", "club_name": "Club", "competition_name": "League", "position": "Position",
                "age": "Age", "mv_display": "Market Val", "recruitment_index": "RI ⭐", "ability_score": "Ability",
                "context_score": "Context", "market_score": "Market", "recommendation": "Recommendation", "risk_profile": "Risk"
            }).style.format({
                "RI ⭐": "{:.1f}", "Ability": "{:.1f}", "Context": "{:.1f}", "Market": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True,
            height=500
        )

    # ==========================================
    # TAB 3: INTERACTIVE RADAR & COMPARE
    # ==========================================
    with tab3:
        st.markdown("### ⚔️ Head-to-Head Player Scouting Radar")
        rc1, rc2 = st.columns(2)
        with rc1:
            p1_label = st.selectbox("Select Player 1:", options=player_labels, index=0)
        with rc2:
            p2_label = st.selectbox("Select Player 2:", options=player_labels, index=min(1, len(player_labels)-1))

        p1 = df[df["_label"] == p1_label].iloc[0]
        p2 = df[df["_label"] == p2_label].iloc[0]
        
        # Plotly Radar Chart
        categories = ["Ability Score", "Context Score", "Market Score", "RI Master Score", "Minutes Reliability", "Age Potential"]
        
        def get_radar_vals(row):
            mins_score = min(100, (row.get("minutes_played", 1000) / 2500.0) * 100)
            age_score = 100 if row["age"] <= 24 else (85 if row["age"] <= 29 else 65)
            return [row["ability_score"], row["context_score"], row["market_score"], row["recruitment_index"], mins_score, age_score]
            
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=get_radar_vals(p1), theta=categories, fill='toself', name=p1['player_name'], line_color='#00F2FE'))
        fig_radar.add_trace(go.Scatterpolar(r=get_radar_vals(p2), theta=categories, fill='toself', name=p2['player_name'], line_color='#38EF7D'))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.1)')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0'),
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ==========================================
    # TAB 4: MARKET VALUATION vs RI ANALYTICS
    # ==========================================
    with tab4:
        st.markdown("### 📈 Market Valuation vs. Recruitment Index Scatter")
        if not f_df.empty:
            scatter_df = f_df.assign(mv_m=f_df["market_value"] / 1e6)
            fig_scatter = px.scatter(
                scatter_df, x="mv_m", y="recruitment_index", color="recommendation", size="ability_score",
                hover_data=["player_name", "club_name", "age", "position"],
                labels={"mv_m": "Market Valuation (€M)", "recruitment_index": "Recruitment Index (RI)"},
                title="Finding Market Inefficiencies: High RI Targets at Low Valuation"
            )
            fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,27,34,0.5)', font=dict(color='#E2E8F0'))
            st.plotly_chart(fig_scatter, use_container_width=True)

if __name__ == "__main__":
    main()
