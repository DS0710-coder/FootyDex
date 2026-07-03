# ⚽ FootyDex — Football Transfer Intelligence Dashboard (Moneyball Edition)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?style=for-the-badge&logo=plotly)
![Docker](https://img.shields.io/badge/Docker-Self--Hosted_API-2496ED?style=for-the-badge&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**FootyDex** is an enterprise-grade football recruitment intelligence and transfer valuation platform. 

> **FootyDex doesn't try to answer:**  
> *"Who is the best player?"*  
> **It answers:**  
> *"If your club had €50M to spend today... who is the smartest signing?"*

Built to uncover market inefficiencies, high-leverage contractual bargains, and cheaper budget alternatives across Europe's top 5 leagues, FootyDex combines custom data engineering with unsupervised machine learning (**K-Means Clustering** & **Cosine Similarity Engine**) and a narrative-first scouting UI.

---

## 🏗️ Architecture: The 3-Pillar Recruitment Engine

Unlike simple statistical dashboards that mix on-pitch output with market price, FootyDex separates recruitment analysis into distinct, logical pillars that mirror how professional Champions League scouting departments operate:

```text
RAW DATA SOURCES
├── Transfermarkt Local Docker API (Biometrics, Transfers, Contract Expiry Date, Injuries, Market Value)
├── FBref / soccerdata (Standard, Shooting, Passing, Defense, Possession, GCA, Match Logs)
└── Config Directory (`config/league_strength.json`, `position_weights.json`, `market_weights.json`)
        │
        ▼
FEATURE ENGINEERING & CENTRAL FEATURE STORE
├── Data Cleaning & Harmonization (Diacritic/ASCII normalization)
├── Per-90 & Volume Normalization
├── Temporal Weighting (Current Season 60% • Previous 30% • Two Years Ago 10%)
└── CENTRAL FEATURE STORE (Standardized z-score features serving downstream engines without recalculation)
        │
        ▼
THE THREE EVALUATION PILLARS
├── PILLAR 1: Footballing Ability & Execution (Position Impact • Technical • Tactical • Physical • Form)
├── PILLAR 2: Context & Environment (League Strength • Team Dominance • Big Match & Pressure Output)
└── PILLAR 3: Market & Leverage (Contract Years Remaining ⭐ • Age Trajectory • Selling Club Leverage • Injuries)
        │
        ▼
RECRUITMENT INTELLIGENCE MODELS (ML & Inference Layer)
├── Recruitment Index (RI) Engine (Holistic proprietary 0–100 recruitment rating)
├── Tactical Profile Discovery Engine (Unsupervised K-Means clustering into 13 scouting archetypes)
├── Similarity Engine (Cosine similarity for Most Similar Profiles & Cheaper Budget Alternatives ⭐)
├── Replacement Value Engine (+X RI above average positional starter)
└── Explainability Engine (Automated percentile-backed + and - narrative generation)
        │
        ▼
SCOUTING REPORT GENERATION (Narrative-First UI)
```

```text
+------------------------------------------------------------------------------------+
|                             FOOTYDEX DATA PIPELINE                                 |
+------------------------------------------------------------------------------------+
|                                                                                    |
|  [ Docker: Self-Hosted Transfermarkt API ]       [ soccerdata / FBref Scraper ]    |
|        (http://localhost:8000)                        (Selenium / WebDriver)       |
|                   │                                             │                  |
|                   ▼                                             ▼                  |
|       scripts/collect_data.py                    scripts/collect_fbref_stats.py    |
|                   │                                             │                  |
|                   ▼                                             ▼                  |
|           data/players.csv                           data/fbref_stats.csv          |
|           data/transfers.csv                                    │                  |
|                   │                                             │                  |
|                   +----------------------┬----------------------+                  |
|                                          │                                         |
|                                          ▼                                         |
|                           scripts/moneyball_score.py                               |
|                  (Z-Score Feature Engineering & Categorization)                    |
|                                          │                                         |
|                                          ▼                                         |
|                             data/moneyball_players.csv                             |
|                                          │                                         |
|                                          ▼                                         |
|                           dashboard/app.py (Streamlit UI)                          |
+------------------------------------------------------------------------------------+
```

### 1. Why Live Scraping First?
Initially, the objective was to build a live, on-demand transfer analytics engine. To achieve this without relying on third-party paid APIs, I deployed a self-hosted **Transfermarkt REST API running inside a Docker container**. This foundational step provided hands-on engineering experience in containerization, microservice architectures, rate-limited HTTP polling, and custom currency/attribute parsing across thousands of player profiles in Europe's top 5 leagues (**Premier League, La Liga, Bundesliga, Serie A, and Ligue 1**).

### 2. The Real-World Limitations
During scaling and testing, two significant engineering bottlenecks emerged:
* **Frontend Component Migrations**: Transfermarkt migrated several of its statistical tables (such as goals, assists, and minutes played) to dynamic Svelte web components. This broke standard REST/HTML table scrapers on the `/stats` endpoints, returning empty payloads for key performance metrics.
* **Rate-Limiting & IP Blocking Risks**: Attempting to execute live, multi-request scraping sweeps across thousands of player profiles on every user dashboard visit introduced severe latency and risked permanent IP bans from target servers.

### 3. The Engineering Pivot & Solution
Recognizing when an architectural approach no longer scales is a critical engineering maturity marker. To build a robust, production-grade system, I executed a strategic three-part pivot:

1. **Decoupled & Specialized Data Sources**: I refined the Dockerized Transfermarkt scraper (`scripts/collect_data.py`) to specialize exclusively in what it does best—extracting club hierarchies, player biometric profiles, current valuations (€), and historical transfer fee records.
2. **Supplementing Performance with FBref**: To resolve the broken statistical endpoint limitation, I identified a high-fidelity alternative data source. Using the Python `soccerdata` library, I built a secondary automated pipeline (`scripts/collect_fbref_stats.py`) to scrape standard season performance and **Expected Goals (xG)** statistics directly from **FBref**.
3. **Transition to Scheduled Batch Refresh**: Instead of fragile on-demand live scraping, the architecture was transitioned to a scheduled periodic batch processing workflow. Because major football transfer windows only open twice a year (summer and January), running scheduled data refreshes guarantees 100% data freshness for strategic analysis while completely eliminating runtime scraping latency, API timeouts, and IP blocking risks.

---

## 📐 The Moneyball Valuation Engine

Once raw profile and performance datasets are collected, `scripts/moneyball_score.py` merges them using normalized string matching (handling diacritics and naming variations across Transfermarkt and FBref) and computes an objective **0–100 Moneyball Valuation Score**.

### Mathematical Model & Feature Engineering
Rather than using raw counts, all player statistics are transformed into efficiency rates and evaluated against league-wide distributions using **Z-Score Normalization**:

$$Z = \frac{X - \mu}{\sigma}$$

The composite valuation score is weighted across four strategic dimensions:
1. **Fee-to-Value Efficiency (40% Weight)**: Measures market exploitation. Evaluates the ratio of a player's latest transfer fee to their true market value ($\text{Fee} / \text{Market Value}$). Lower ratios (or free/academy transfers) receive high positive z-scores.
2. **Attacking & Expected Efficiency / 90 (30% Weight)**: Evaluates non-penalty goals, assists, and Expected Goals per 90 minutes ($\text{GC/90} + \text{xG/90}$), rewarding high-impact output scaled by playing time.
3. **Age Optimality (20% Weight)**: Applies a Gaussian decay reward distribution peaking at age **23.5** (the statistical beginning of a footballer's prime physical and financial resale value window):
   $$\text{Age Reward} = \exp\left(-\frac{(\text{Age} - 23.5)^2}{18}\right)$$
4. **Experience & Reliability (10% Weight)**: Log-transformed total minutes played ($\ln(1 + \text{Minutes})$) to ensure statistical reliability and penalize small sample sizes.

### Strategic Categorization
Players are automatically classified into actionable transfer market categories:
* 🟢 **Bargain**: High Moneyball valuation score ($>75$) with a favorable fee-to-value ratio ($<0.8x$).
* 💎 **Hidden Gem**: Elite statistical performers ($>70$ score) currently undervalued by the general market (Market Value $< \text{€15M}$).
* 🟡 **Fair Value**: Balanced performers whose transfer fee accurately reflects their on-pitch output ($40 \le \text{Score} \le 75$).
* 🔴 **Overpriced**: Low efficiency output ($<40$ score) combined with inflated transfer acquisition costs ($>1.3x$ market value).
* ⚠️ **High Risk**: Older players ($>29$ years) acquired for fees exceeding their declining capital market value.

---

## 🖥️ Interactive Streamlit Dashboard

The frontend (`dashboard/app.py`) delivers a visually stunning, dark-themed glassmorphic user interface powered by custom CSS typography (`Outfit` and `Inter` fonts) and interactive Plotly visualisations:

* **⚽ Valuation Matrix**: An interactive scatter plot mapping Market Value vs. Moneyball Score, dynamically sized by minutes played and colored by strategic label.
* **🔥 Bargains vs. 💸 Overpriced Leaders**: Instant horizontal bar charts highlighting the top 10 highest-value transfer opportunities and the most over-leveraged transfer risks.
* **🕷️ Head-to-Head Radar Comparisons**: Select any two players in Europe to compare their normalized 0–100 attributes across Moneyball Score, Value Efficiency, Goal Contributions / 90, Age Optimality, and Experience.
* **📋 Enriched Data Directory**: Filter by league, position, age range, market value, and valuation label with full CSV export capabilities.
* **🔎 Player Deep Dive**: Examine individual biometric profiles, real FBref xG performance metrics, and chronological season-by-season transfer history development.

---

## 🚀 Quickstart Guide

### Prerequisites
* Python 3.10+
* Docker (optional, if running the self-hosted Transfermarkt API locally)
* Google Chrome (required for Selenium/FBref automated data extraction)

### 1. Installation
Clone the repository and install the required Python dependencies:
```bash
git clone https://github.com/DS0710-coder/FootyDex.git
cd FootyDex
pip install -r requirements.txt
```

### 2. Run the Data Pipeline & Recruitment Models (Scheduled Refresh)
Execute the specialized collectors and machine learning models to generate fresh recruitment datasets:
```bash
# Step 1: Collect profile, contract expiration dates, injuries, and valuations from Transfermarkt
python3 scripts/collect_data.py

# Step 2: Scrape multi-table performance statistics from FBref
python3 scripts/collect_fbref_stats.py

# Step 3: Execute Feature Store, the 3 Pillars, Cosine Similarity, K-Means Clustering & Explainability
python3 scripts/recruitment_engine.py
```
*(Note: For rapid testing or demonstrations, you can limit collection using `python3 scripts/collect_data.py --limit-per-club 3` or `python3 scripts/collect_fbref_stats.py --league "ENG-Premier League"`).*

### 3. Launch the Dashboard
Start the interactive Streamlit application:
```bash
streamlit run dashboard/app.py
```
Open your browser to `http://localhost:8501` to explore the transfer intelligence dashboard!

---

## 📝 License
This project is open-source and licensed under the MIT License.
