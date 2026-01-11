import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime


# --- 1. FEATURE ENGINEERING CLASS ---
class CricketFeatureEngineering:
    def transform_user_input(self, user_data):
        features = {}

        # Basic Numeric Inputs [cite: 13, 16]
        features['innings'] = user_data['innings']
        features['over'] = user_data['over']
        features['ball'] = user_data['ball']
        features['current_score'] = user_data['current_score']
        features['current_wickets'] = user_data['current_wickets']
        features['runs_last_5'] = user_data['runs_last_5']
        features['runs_last_10'] = user_data['runs_last_10']
        features['target'] = user_data.get('target', 0)

        # Convert Categorical Strings to Numbers [cite: 12, 13, 16]
        features['toss_decision'] = 1 if user_data['toss_decision'] == 'bat' else 0

        # Ball Calculations [cite: 13]
        features['balls_bowled'] = (user_data['over'] * 6) + user_data['ball']
        features['balls_remaining'] = 120 - features['balls_bowled']
        features['wickets_remaining'] = 10 - user_data['current_wickets']

        # Run Rates [cite: 13]
        features['current_run_rate'] = (user_data['current_score'] / features['balls_bowled'] * 6) if features[
                                                                                                          'balls_bowled'] > 0 else 0.0
        features['run_rate_last_5'] = (user_data['runs_last_5'] / 5 * 6)
        features['run_rate_last_10'] = (user_data['runs_last_10'] / 10 * 6)

        # Match Phase Logic [cite: 13, 14]
        features['is_powerplay'] = 1 if user_data['over'] < 6 else 0
        features['is_middle'] = 1 if 6 <= user_data['over'] < 16 else 0
        features['is_death'] = 1 if user_data['over'] >= 16 else 0
        features['match_phase'] = 0 if features['is_powerplay'] else (1 if features['is_middle'] else 2)

        # Toss and Team Status [cite: 14]
        features['toss_winner_batting'] = 1 if user_data['batting_team'] == user_data['toss_winner'] else 0
        features['toss_bat'] = features['toss_decision']
        features['toss_field'] = 1 - features['toss_decision']

        # Required Runs (2nd Innings) [cite: 13, 14]
        if user_data['innings'] == 2:
            features['required_runs'] = user_data['target'] - user_data['current_score']
            features['required_run_rate'] = (features['required_runs'] / features['balls_remaining'] * 6) if features[
                                                                                                                 'balls_remaining'] > 0 else 0.0
        else:
            features['required_runs'] = 0
            features['required_run_rate'] = 0.0

        # Time Features
        now = datetime.now()
        features['year'] = now.year
        features['month'] = now.month
        features['day_of_week'] = now.weekday()

        # Placeholders for Categorical/Reduced columns [cite: 14, 15]
        reduced_cols = [
            'venue_reduced', 'city_reduced', 'batting_team_reduced',
            'bowling_team_reduced', 'team1_reduced', 'team2_reduced', 'toss_winner_reduced'
        ]
        for col in reduced_cols:
            features[col] = 0

        # Return as DataFrame with specific order required by model [cite: 12, 13, 14, 15]
        order = [
            'toss_decision', 'innings', 'over', 'ball', 'current_score', 'current_wickets',
            'balls_bowled', 'balls_remaining', 'runs_last_5', 'runs_last_10', 'current_run_rate',
            'run_rate_last_5', 'run_rate_last_10', 'wickets_remaining', 'match_phase',
            'is_powerplay', 'is_middle', 'is_death', 'required_runs', 'required_run_rate',
            'toss_winner_batting', 'toss_bat', 'toss_field', 'year', 'month', 'day_of_week',
            'venue_reduced', 'city_reduced', 'batting_team_reduced', 'bowling_team_reduced',
            'team1_reduced', 'team2_reduced', 'toss_winner_reduced'
        ]
        return pd.DataFrame([features])[order]


# --- 2. STREAMLIT APP SETUP ---
st.set_page_config(page_title="Cricket Win Predictor", layout="wide")

# Session State for History
if "history" not in st.session_state:
    st.session_state.history = []

# --- STYLING (Title, Sidebar, Background) ---
st.markdown("""
    <style>
    .title-box { display: inline-block; padding: 18px 42px; font-size: 42px; font-weight: 700; color: #f8fafc; background-color: #040b13; border-radius: 14px; border: 1px solid #1f2933; letter-spacing: 1.2px; }
    section[data-testid="stSidebar"] { background-color: #040b13; }
    section[data-testid="stSidebar"] * { color: #f5f3f0; }
    section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] .stNumberInput input { background-color: #28394b !important; color: #ffffff !important; border-radius: 8px; border: 1px solid rgba(255,255,255,0.15); }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div { background-color: #2f4a67; color: #ffffff !important; border-radius: 8px; }
    .stApp { background: radial-gradient(circle at top, #1a4370 0%, #091625 40%, #122c4b 100%); }
    section[data-testid="stMain"] { background: transparent; }
    div[data-testid="stTabs"] { width: 100%; }
    button[data-baseweb="tab"] { flex: 1; justify-content: center; font-size: 18px; }
    button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #ff4b4b; font-weight: 600; }
    </style>
    <div style="text-align:center; margin-bottom:32px;"><div class="title-box">🏏 Cricket Prediction Model</div></div>
""", unsafe_allow_html=True)





st.markdown(
    """
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #040b13 !important;
        color: white !important;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-size: 16px;
        font-weight: 600;
        border: none;
        transition: background-color 0.25s ease;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1f4e83 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)




fe = CricketFeatureEngineering()

# Sidebar: Configuration
st.sidebar.header("Configuration")
model_type = st.sidebar.selectbox("Choose Model", ["XGBoost", "CatBoost"])

if st.sidebar.button("Reset Graph History"):
    st.session_state.history = []


@st.cache_resource
def load_model(name):
    model_files = {"XGBoost": "xgboost_.pkl", "CatBoost": "catboost_.pkl"}
    with open(model_files[name], "rb") as f:
        return pickle.load(f)


model = load_model(model_type)

# Sidebar: Match Parameters
st.sidebar.divider()
st.sidebar.subheader("Match Info")
batting_team = st.sidebar.selectbox("Batting Team",
                                    ["India", "Australia", "England", "Pakistan", "South Africa", "New Zealand",
                                     "West Indies", "Sri Lanka"])
bowling_team = st.sidebar.selectbox("Bowling Team",
                                    ["India", "Australia", "England", "Pakistan", "South Africa", "New Zealand",
                                     "West Indies", "Sri Lanka"])
toss_winner = st.sidebar.selectbox("Toss Winner", [batting_team, bowling_team])
toss_decision = st.sidebar.radio("Toss Decision", ["bat", "field"])
innings = st.sidebar.radio("Innings", [1, 2])
venue = st.sidebar.text_input("Venue", "Eden Gardens")
city = st.sidebar.text_input("City", "Kolkata")

st.sidebar.subheader("Match Status")
current_score = st.sidebar.number_input("Current Score", min_value=0, value=100)
current_wickets = st.sidebar.number_input("Wickets Lost", min_value=0, max_value=10, value=2)
over = st.sidebar.number_input("Over Number (0-19)", min_value=0, max_value=19, value=10)
ball = st.sidebar.number_input("Ball in Over (1-6)", min_value=1, max_value=6, value=1)

target = st.sidebar.number_input("Target Score", min_value=1, value=180) if innings == 2 else 0
runs_last_5 = st.sidebar.number_input("Runs in last 5 balls", min_value=0, value=7)
runs_last_10 = st.sidebar.number_input("Runs in last 10 balls", min_value=0, value=12)

# Data Preparation
user_data = {
    'batting_team': batting_team, 'bowling_team': bowling_team, 'toss_winner': toss_winner,
    'toss_decision': toss_decision, 'innings': innings, 'venue': venue, 'city': city,
    'current_score': current_score, 'current_wickets': current_wickets, 'over': over,
    'ball': ball, 'target': target, 'runs_last_5': runs_last_5, 'runs_last_10': runs_last_10
}

# --- 3. MAIN INTERFACE ---
tab1, tab2 = st.tabs(["Match Summary", "Prediction Results"])

with tab1:
    st.markdown("<h2 style='font-size:30px; margin-bottom:18px;'>Current Situation</h2>", unsafe_allow_html=True)
    st.write(f"**{batting_team}** is batting against **{bowling_team}**")
    st.write(f"Location: {venue}, {city}")
    st.write(f"Toss: {toss_winner} won and chose to {toss_decision}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{current_score}/{current_wickets}")
    c2.metric("Overs", f"{over}.{ball}")
    if innings == 2: c3.metric("Target", f"{target}")

with tab2:
    if st.button("Predict Probability", type="primary"):
        # Transform data
        input_df = fe.transform_user_input(user_data)

        # Predict Probabilities [cite: 1, 521, 522]
        if model_type == "XGBoost":
            probs = model.predict_proba(input_df)[0]
            win_p = probs[1] * 100
        else:  # CatBoost
            probs = model.predict_proba(input_df)
            win_p = probs[0][1] * 100

        # Save to history
        st.session_state.history.append({
            'Over': over + (ball / 6),
            'Win Probability': win_p
        })

        # Visualization: Gauge and Line Chart
        col_left, col_right = st.columns([1, 2])

        with col_left:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=win_p,
                title={'text': f"{batting_team} Win %", 'font': {'color': "white"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': "white"},
                    'steps': [{'range': [0, 50], 'color': "red"}, {'range': [50, 100], 'color': "green"}]
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_right:
            st.subheader("Win Probability Over Time")
            if len(st.session_state.history) > 0:
                h_df = pd.DataFrame(st.session_state.history)
                fig_line = go.Figure(go.Scatter(
                    x=h_df['Over'], y=h_df['Win Probability'],
                    mode='lines+markers', line=dict(color='#22c55e', width=3)
                ))
                fig_line.update_layout(
                    xaxis_title="Overs", yaxis_title="Win %", yaxis_range=[0, 100],
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white"}
                )
                st.plotly_chart(fig_line, use_container_width=True)

        # Feature Importance
        st.subheader("Key Influencing Factors")
        feat_imp = pd.Series(model.feature_importances_, index=input_df.columns)
        st.bar_chart(feat_imp.sort_values(ascending=False).head(8))

        st.success(f"Prediction Complete: **{win_p:.2f}%** chance for **{batting_team}**.")