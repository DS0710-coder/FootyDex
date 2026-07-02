#!/usr/bin/env python3
"""
FootyDex — Football Transfer Intelligence Dashboard (Moneyball Edition)
Streamlit Application: Interactive, visually stunning, dark-themed dashboard with Plotly visualisations,
custom styling, player comparisons, and deep-dive transfer analytics.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set Page Configuration
st.set_page_config(
    page_title="FootyDex | Transfer Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic Dark Theme & Typography CSS
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

/* Glassmorphic Metric Cards */
.metric-card {
    background: rgba(22, 27, 34, 0.7);
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
    font-size: 1.8rem;
    font-weight: 700;
    color: #F8FAFC;
    font-family: 'Outfit', sans-serif;
}

.metric-sub {
    font-size: 0.85rem;
    color: #38EF7D;
    margin-top: 0.3rem;
    font-weight: 500;
}

.metric-sub.danger {
    color: #FF0844;
}

/* Label Badges */
.badge {
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    display: inline-block;
}
.badge-Bargain { background: rgba(56, 239, 125, 0.15); color: #38EF7D; border: 1px solid #38EF7D; }
.badge-Overpriced { background: rgba(255, 8, 68, 0.15); color: #FF0844; border: 1px solid #FF0844; }
.badge-Hidden-Gem { background: rgba(0, 242, 254, 0.15); color: #00F2FE; border: 1px solid #00F2FE; }
.badge-Fair-Value { background: rgba(249, 212, 35, 0.15); color: #F9D423; border: 1px solid #F9D423; }
.badge-High-Risk { background: rgba(255, 177, 153, 0.15); color: #FFB199; border: 1px solid #FFB199; }

/* Streamlit Tabs Customization */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: rgba(15, 23, 42, 0.6);
    padding: 8px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    height: 44px;
    white-space: pre-wrap;
    background-color: transparent;
    border-radius: 8px;
    color: #94A3B8;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.2) 0%, rgba(79, 172, 254, 0.2) 100%) !important;
    color: #00F2FE !important;
    border: 1px solid rgba(0, 242, 254, 0.4);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Color Palette Mapping
LABEL_COLORS = {
    "Bargain": "#38EF7D",
    "Hidden Gem": "#00F2FE",
    "Fair Value": "#F9D423",
    "High Risk": "#FFB199",
    "Overpriced": "#FF0844"
}

@st.cache_data
def load_data():
    data_path = "data/moneyball_players.csv"
    transfers_path = "data/transfers.csv"
    
    if not os.path.exists(data_path):
        return None, None
        
    df = pd.read_csv(data_path)
    df_trans = pd.read_csv(transfers_path) if os.path.exists(transfers_path) else pd.DataFrame()
    
    # Format currency display helper
    df["mv_display"] = df["market_value"].apply(lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else f"€{x/1e3:.0f}K")
    df["fee_display"] = df["transfer_fee"].apply(lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else (f"€{x/1e3:.0f}K" if x > 0 else "Free / Academy"))
    return df, df_trans

def main():
    st.markdown('<div class="main-header">⚽ FootyDex Moneyball</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Football Transfer Intelligence & Valuations Dashboard — Powered by Advanced Z-Score Analytics</div>', unsafe_allow_html=True)
    
    df, df_transfers = load_data()
    if df is None or df.empty:
        st.warning("⚠️ No Moneyball player dataset found! Please run `python3 scripts/collect_data.py` and `python3 scripts/moneyball_score.py` to generate the data.")
        st.info("💡 Tip: You can run a rapid sample collection in your terminal using: `python3 scripts/collect_data.py --limit-per-club 3`")
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.markdown("### 🔍 Strategic Filters")
    
    # 1. League Filter
    all_leagues = sorted(df["competition_name"].dropna().unique())
    selected_leagues = st.sidebar.multiselect("Competition / League", options=all_leagues, default=all_leagues)
    
    # 2. Position Filter
    all_pos = sorted(df["position"].dropna().unique())
    selected_pos = st.sidebar.multiselect("Player Position", options=all_pos, default=all_pos)
    
    # 3. Age Range
    min_age, max_age = int(df["age"].min()), int(df["age"].max())
    selected_age = st.sidebar.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    
    # 4. Market Value Range (€M)
    min_mv, max_mv = float(df["market_value"].min() / 1e6), float(df["market_value"].max() / 1e6)
    selected_mv = st.sidebar.slider("Market Value Range (€M)", min_value=0.0, max_value=max_mv, value=(0.0, max_mv), step=1.0)
    
    # 5. Moneyball Label Filter
    all_labels = sorted(df["moneyball_label"].dropna().unique())
    selected_labels = st.sidebar.multiselect("Moneyball Valuation Label", options=all_labels, default=all_labels)
    
    # Apply Filters
    filtered_df = df[
        (df["competition_name"].isin(selected_leagues)) &
        (df["position"].isin(selected_pos)) &
        (df["age"] >= selected_age[0]) & (df["age"] <= selected_age[1]) &
        (df["market_value"] >= selected_mv[0] * 1e6) & (df["market_value"] <= selected_mv[1] * 1e6) &
        (df["moneyball_label"].isin(selected_labels))
    ].copy()
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"📊 Showing **{len(filtered_df)}** of **{len(df)}** players")

    # --- SUMMARY METRICS CARDS ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_p = len(filtered_df)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Players Filtered</div>
            <div class="metric-value">{total_p}</div>
            <div class="metric-sub">Across {len(selected_leagues)} Leagues</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        avg_score = filtered_df["moneyball_score"].mean() if not filtered_df.empty else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Moneyball Score</div>
            <div class="metric-value">{avg_score:.1f} <span style="font-size:1rem;color:#64748B;">/ 100</span></div>
            <div class="metric-sub">Z-Score Normalized</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        if not filtered_df.empty:
            bargain_row = filtered_df.loc[filtered_df["moneyball_score"].idxmax()]
            b_name = bargain_row["player_name"]
            b_val = f"{bargain_row['moneyball_score']:.1f} ({bargain_row['mv_display']})"
        else:
            b_name, b_val = "N/A", "-"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔥 Biggest Bargain / Gem</div>
            <div class="metric-value" style="font-size:1.4rem;color:#38EF7D;">{b_name}</div>
            <div class="metric-sub">Score: {b_val}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        if not filtered_df.empty:
            over_row = filtered_df.loc[filtered_df["fee_to_value_ratio"].idxmax()] if filtered_df["fee_to_value_ratio"].max() > 0 else filtered_df.loc[filtered_df["moneyball_score"].idxmin()]
            o_name = over_row["player_name"]
            o_val = f"Ratio: {over_row['fee_to_value_ratio']:.1f}x ({over_row['fee_display']})"
        else:
            o_name, o_val = "N/A", "-"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚠️ Most Overpriced</div>
            <div class="metric-value" style="font-size:1.4rem;color:#FF0844;">{o_name}</div>
            <div class="metric-sub danger">{o_val}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dashboard Overview",
        "🕷️ Player Comparison (Radar)",
        "📋 Interactive Data Grid",
        "🔎 Player Deep Dive & Transfers"
    ])

    # === TAB 1: OVERVIEW ===
    with tab1:
        if filtered_df.empty:
            st.info("No players match the current filter criteria.")
        else:
            row1_col1, row1_col2 = st.columns([3, 2])
            
            # 1. Scatter Plot: Moneyball Score vs Market Value
            with row1_col1:
                st.markdown("#### ⚽ Valuation Matrix: Moneyball Score vs. Market Value")
                fig_scatter = px.scatter(
                    filtered_df,
                    x="market_value",
                    y="moneyball_score",
                    color="moneyball_label",
                    color_discrete_map=LABEL_COLORS,
                    size="minutes_played",
                    size_max=24,
                    hover_name="player_name",
                    hover_data={
                        "club_name": True,
                        "position": True,
                        "age": True,
                        "mv_display": True,
                        "moneyball_score": ":.1f",
                        "market_value": False,
                        "moneyball_label": False
                    },
                    template="plotly_dark"
                )
                fig_scatter.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Market Value (€)", gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
                    yaxis=dict(title="Moneyball Valuation Score (0-100)", gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
                    legend=dict(title="Category", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=40, b=20),
                    height=450
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            # 2. Label Distribution Pie/Donut
            with row1_col2:
                st.markdown("#### 🎯 Valuation Category Breakdown")
                label_counts = filtered_df["moneyball_label"].value_counts().reset_index()
                label_counts.columns = ["Label", "Count"]
                fig_pie = px.pie(
                    label_counts,
                    names="Label",
                    values="Count",
                    color="Label",
                    color_discrete_map=LABEL_COLORS,
                    hole=0.55,
                    template="plotly_dark"
                )
                fig_pie.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                    margin=dict(l=10, r=10, t=40, b=20),
                    height=450
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)

            # Row 2: Bar Charts
            row2_col1, row2_col2 = st.columns(2)
            
            # 3. Bar Chart: Top 10 Bargains
            with row2_col1:
                st.markdown("#### 🔥 Top 10 High-Value Bargains & Gems")
                top_bargains = filtered_df[filtered_df["moneyball_label"].isin(["Bargain", "Hidden Gem", "Fair Value"])].nlargest(10, "moneyball_score")
                if not top_bargains.empty:
                    fig_bar1 = px.bar(
                        top_bargains,
                        x="moneyball_score",
                        y="player_name",
                        orientation="h",
                        color="moneyball_label",
                        color_discrete_map=LABEL_COLORS,
                        text="moneyball_score",
                        hover_data=["club_name", "position", "mv_display", "fee_display"],
                        template="plotly_dark"
                    )
                    fig_bar1.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(title="", categoryorder="total ascending"),
                        xaxis=dict(title="Moneyball Score", gridcolor="rgba(255,255,255,0.05)"),
                        showlegend=False,
                        margin=dict(l=10, r=20, t=20, b=20),
                        height=380
                    )
                    fig_bar1.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                    st.plotly_chart(fig_bar1, use_container_width=True)
                else:
                    st.info("No Bargain/Gem players found in current selection.")
                    
            # 4. Bar Chart: Top 10 Overpriced Players
            with row2_col2:
                st.markdown("#### 💸 Top 10 Most Overpriced & High Risk Transfers")
                top_over = filtered_df[filtered_df["moneyball_label"].isin(["Overpriced", "High Risk"])].nsmallest(10, "moneyball_score")
                if top_over.empty:
                    top_over = filtered_df.nsmallest(10, "moneyball_score")
                if not top_over.empty:
                    fig_bar2 = px.bar(
                        top_over,
                        x="moneyball_score",
                        y="player_name",
                        orientation="h",
                        color="moneyball_label",
                        color_discrete_map=LABEL_COLORS,
                        text="fee_to_value_ratio",
                        hover_data=["club_name", "position", "mv_display", "fee_display"],
                        template="plotly_dark"
                    )
                    fig_bar2.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(title="", categoryorder="total descending"),
                        xaxis=dict(title="Moneyball Score (Lower = More Overpriced)", gridcolor="rgba(255,255,255,0.05)"),
                        showlegend=False,
                        margin=dict(l=10, r=20, t=20, b=20),
                        height=380
                    )
                    fig_bar2.update_traces(texttemplate="%{text:.1f}x Ratio", textposition="outside")
                    st.plotly_chart(fig_bar2, use_container_width=True)
                else:
                    st.info("No Overpriced players found in current selection.")

    # === TAB 2: RADAR COMPARISON ===
    with tab2:
        st.markdown("#### 🕷️ Head-to-Head Player Attribute Radar")
        st.caption("Select two players to visually compare their normalized efficiency across 5 key transfer metrics.")
        
        if len(df) < 2:
            st.warning("Need at least 2 players to compare.")
        else:
            c_sel1, c_sel2 = st.columns(2)
            player_list = sorted(df["player_name"].unique())
            
            with c_sel1:
                p1_name = st.selectbox("Select Player 1", options=player_list, index=0)
            with c_sel2:
                p2_idx = 1 if len(player_list) > 1 else 0
                p2_name = st.selectbox("Select Player 2", options=player_list, index=p2_idx)
                
            p1_row = df[df["player_name"] == p1_name].iloc[0]
            p2_row = df[df["player_name"] == p2_name].iloc[0]
            
            # Normalize metrics to 0-100 scale for intuitive radar comparison
            def get_radar_stats(row):
                # 1. Moneyball Score (0-100)
                m_score = min(100, max(0, row["moneyball_score"]))
                # 2. Value Efficiency (1 / (1 + ftv_ratio) * 100) -> higher is better bargain
                v_eff = min(100, max(0, (1.0 / (1.0 + row["fee_to_value_ratio"])) * 120.0))
                # 3. Goal Contributions / 90 (scaled to ~1.0 = 100)
                gc_scale = min(100, max(0, row["goal_contributions_per_90"] * 80.0))
                # 4. Age Optimality (peak at 23.5)
                age_opt = min(100, max(0, np.exp(-((row["age"] - 23.5)**2)/18.0) * 100.0))
                # 5. Experience / Minutes (3000 mins = 100)
                min_scale = min(100, max(0, (row["minutes_played"] / 3000.0) * 100.0))
                
                return [m_score, v_eff, gc_scale, age_opt, min_scale]
                
            categories = ['Moneyball Score', 'Value Efficiency', 'Goal Contrib. / 90', 'Age Optimality', 'Experience / Minutes']
            stats1 = get_radar_stats(p1_row)
            stats2 = get_radar_stats(p2_row)
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=stats1 + [stats1[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=f"{p1_name} ({p1_row['club_name']})",
                line_color='#00F2FE',
                fillcolor='rgba(0, 242, 254, 0.25)'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=stats2 + [stats2[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=f"{p2_name} ({p2_row['club_name']})",
                line_color='#FF0844',
                fillcolor='rgba(255, 8, 68, 0.25)'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)"),
                    bgcolor="rgba(22, 27, 34, 0.5)",
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)")
                ),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Comparison Cards
            cmp1, cmp2 = st.columns(2)
            with cmp1:
                st.markdown(f"**👤 {p1_name}** ({p1_row['competition_name']}) — `{p1_row['moneyball_label']}`")
                st.write(f"- **Club**: {p1_row['club_name']} | **Age**: {p1_row['age']} | **Pos**: {p1_row['position']}")
                st.write(f"- **Market Value**: {p1_row['mv_display']} | **Fee**: {p1_row['fee_display']}")
                st.write(f"- **Goals/90**: `{p1_row['goals_per_90']:.2f}` | **Assists/90**: `{p1_row['assists_per_90']:.2f}`")
            with cmp2:
                st.markdown(f"**👤 {p2_name}** ({p2_row['competition_name']}) — `{p2_row['moneyball_label']}`")
                st.write(f"- **Club**: {p2_row['club_name']} | **Age**: {p2_row['age']} | **Pos**: {p2_row['position']}")
                st.write(f"- **Market Value**: {p2_row['mv_display']} | **Fee**: {p2_row['fee_display']}")
                st.write(f"- **Goals/90**: `{p2_row['goals_per_90']:.2f}` | **Assists/90**: `{p2_row['assists_per_90']:.2f}`")

    # === TAB 3: INTERACTIVE DATA GRID ===
    with tab3:
        st.markdown("#### 📋 Full Enriched Moneyball Player Directory")
        st.caption("Search, filter, sort, and export the comprehensive transfer intelligence dataset.")
        
        display_cols = [
            "player_name", "competition_name", "club_name", "position", "age",
            "mv_display", "fee_display", "fee_to_value_ratio", "goal_contributions_per_90",
            "minutes_played", "moneyball_score", "moneyball_label"
        ]
        
        # Friendly column renaming for table
        col_rename = {
            "player_name": "Player",
            "competition_name": "League",
            "club_name": "Club",
            "position": "Position",
            "age": "Age",
            "mv_display": "Market Value",
            "fee_display": "Transfer Fee",
            "fee_to_value_ratio": "Fee/Val Ratio",
            "goal_contributions_per_90": "GC / 90",
            "minutes_played": "Minutes",
            "moneyball_score": "Moneyball Score",
            "moneyball_label": "Valuation Label"
        }
        
        grid_df = filtered_df[display_cols].rename(columns=col_rename)
        
        st.dataframe(
            grid_df.style.format({
                "Fee/Val Ratio": "{:.2f}x",
                "GC / 90": "{:.2f}",
                "Moneyball Score": "{:.1f}"
            }),
            use_container_width=True,
            height=480
        )
        
        # Download CSV
        csv_data = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Enriched Dataset as CSV",
            data=csv_data,
            file_name="footydex_moneyball_players.csv",
            mime="text/csv",
            use_container_width=False
        )

    # === TAB 4: PLAYER DEEP DIVE ===
    with tab4:
        st.markdown("#### 🔎 Player Intelligence Deep Dive & Full Transfer History")
        
        sel_player_name = st.selectbox("Select Player for Comprehensive Profile Analysis", options=sorted(df["player_name"].unique()))
        p_data = df[df["player_name"] == sel_player_name].iloc[0]
        
        d_col1, d_col2 = st.columns([1, 2])
        
        with d_col1:
            badge_class = p_data["moneyball_label"].replace(" ", "-")
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin-bottom:0.2rem;color:#00F2FE;">{p_data['player_name']}</h3>
                <div style="color:#94A3B8;margin-bottom:1rem;">{p_data['club_name']} • {p_data['competition_name']}</div>
                <div class="badge badge-{badge_class}">{p_data['moneyball_label']}</div>
                <hr style="border-color:rgba(255,255,255,0.08);margin:1rem 0;">
                <p><b>Position:</b> {p_data['position']}</p>
                <p><b>Age:</b> {p_data['age']} yrs | <b>Foot:</b> {str(p_data.get('foot', 'right')).capitalize()}</p>
                <p><b>Nationality:</b> {p_data['nationality']}</p>
                <p><b>Market Value:</b> <span style="color:#38EF7D;font-weight:700;">{p_data['mv_display']}</span></p>
                <p><b>Latest Transfer Fee:</b> {p_data['fee_display']}</p>
                <p><b>Fee-to-Value Ratio:</b> <code>{p_data['fee_to_value_ratio']:.2f}x</code></p>
            </div>
            """, unsafe_allow_html=True)
            
        with d_col2:
            st.markdown("##### ⚡ Performance Efficiency & Moneyball Metrics")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Moneyball Score", f"{p_data['moneyball_score']:.1f} / 100")
                st.metric("Total Goals", f"{int(p_data['goals'])}")
            with m_col2:
                st.metric("Goal Contrib. / 90", f"{p_data['goal_contributions_per_90']:.2f}")
                st.metric("Total Assists", f"{int(p_data['assists'])}")
            with m_col3:
                st.metric("Minutes Played", f"{int(p_data['minutes_played'])}'")
                st.metric("Goals / 90", f"{p_data['goals_per_90']:.2f}")
                
            st.markdown("##### 🔄 Full Transfer Development History")
            if not df_transfers.empty and "player_id" in df_transfers.columns:
                p_transfers = df_transfers[df_transfers["player_id"] == str(p_data["player_id"])].sort_values(by="season", ascending=False)
                if not p_transfers.empty:
                    t_display = p_transfers[["season", "transfer_date", "club_from_name", "club_to_name", "market_value_at_transfer", "transfer_fee"]].copy()
                    t_display.columns = ["Season", "Date", "From Club", "To Club", "MV at Transfer (€)", "Fee (€)"]
                    t_display["MV at Transfer (€)"] = t_display["MV at Transfer (€)"].apply(lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else f"€{x/1e3:.0f}K")
                    t_display["Fee (€)"] = t_display["Fee (€)"].apply(lambda x: f"€{x/1e6:.1f}M" if x >= 1e6 else (f"€{x/1e3:.0f}K" if x > 0 else "Free / Loan"))
                    st.dataframe(t_display, use_container_width=True)
                else:
                    st.info("No detailed transfer records found for this player in local database.")
            else:
                st.info("Transfer history database is currently empty.")

if __name__ == "__main__":
    main()
