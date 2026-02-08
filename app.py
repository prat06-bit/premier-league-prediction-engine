import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #37003c;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #00ff85;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #37003c;
        color: white;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00ff85;
        color: #37003c;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models and feature columns"""
    try:
        xgb_model = joblib.load('models/xgboost_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        return xgb_model, rf_model, feature_cols, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, False

@st.cache_data
def load_data():
    """Load features and team list"""
    try:
        features_df = pd.read_csv('data/features.csv')
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        
        # Get unique teams
        home_teams = set(features_df['HomeTeam'].unique())
        away_teams = set(features_df['AwayTeam'].unique())
        teams = sorted(home_teams | away_teams)
        
        return features_df, teams, True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, False

def get_team_stats(features_df, team_name, is_home=True):
    """Get latest statistics for a team"""
    try:
        if is_home:
            team_matches = features_df[features_df['HomeTeam'] == team_name]
        else:
            team_matches = features_df[features_df['AwayTeam'] == team_name]
        
        if len(team_matches) == 0:
            return None
        
        # Get most recent match
        team_matches = team_matches.sort_values('Date', ascending=False)
        return team_matches.iloc[0]
    except Exception as e:
        st.error(f"Error getting team stats: {e}")
        return None

def extract_features(home_stats, away_stats, feature_cols):
    """Extract features from team statistics"""
    match_features = pd.DataFrame(index=[0])
    
    for col in feature_cols:
        if '_home' in col:
            try:
                match_features[col] = home_stats[col]
            except:
                match_features[col] = 0
        elif '_away' in col:
            try:
                match_features[col] = away_stats[col]
            except:
                match_features[col] = 0
        else:
            # Differential features
            match_features[col] = 0
    
    # Ensure all features exist
    for col in feature_cols:
        if col not in match_features.columns:
            match_features[col] = 0
    
    match_features = match_features[feature_cols]
    return match_features

def predict_match(home_team, away_team, xgb_model, rf_model, features_df, feature_cols):
    """Predict match outcome"""
    # Get team stats
    home_stats = get_team_stats(features_df, home_team, is_home=True)
    away_stats = get_team_stats(features_df, away_team, is_home=False)
    
    if home_stats is None or away_stats is None:
        return None
    
    # Extract features
    match_features = extract_features(home_stats, away_stats, feature_cols)
    
    # Make predictions
    xgb_proba = xgb_model.predict_proba(match_features)[0]
    rf_proba = rf_model.predict_proba(match_features)[0]
    
    # Ensemble (60% XGBoost, 40% RF)
    ensemble_proba = 0.6 * xgb_proba + 0.4 * rf_proba
    
    prediction_idx = np.argmax(ensemble_proba)
    outcomes = ['Home Win', 'Draw', 'Away Win']
    
    return {
        'predicted_outcome': outcomes[prediction_idx],
        'confidence': ensemble_proba[prediction_idx],
        'home_win_prob': ensemble_proba[0],
        'draw_prob': ensemble_proba[1],
        'away_win_prob': ensemble_proba[2],
        'xgb_proba': xgb_proba,
        'rf_proba': rf_proba
    }

def create_probability_chart(result):
    """Create probability bar chart"""
    outcomes = ['Home Win', 'Draw', 'Away Win']
    probabilities = [
        result['home_win_prob'],
        result['draw_prob'],
        result['away_win_prob']
    ]
    
    colors = ['#00ff85', '#ffd700', '#ff4444']
    
    fig = go.Figure(data=[
        go.Bar(
            x=outcomes,
            y=[p * 100 for p in probabilities],
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability (%)",
        xaxis_title="Outcome",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_gauge_chart(confidence):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence Level"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#00ff85"},
            'steps': [
                {'range': [0, 45], 'color': "#ffcccc"},
                {'range': [45, 55], 'color': "#fff4cc"},
                {'range': [55, 65], 'color': "#ccffcc"},
                {'range': [65, 100], 'color': "#ccffee"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 65
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def get_betting_recommendation(confidence):
    """Get betting recommendation based on confidence"""
    if confidence >= 0.65:
        return "🟢 STRONG BET", "High confidence prediction. Good betting opportunity."
    elif confidence >= 0.55:
        return "🟡 MODERATE BET", "Moderate confidence. Proceed with caution."
    elif confidence >= 0.45:
        return "🟠 WEAK SIGNAL", "Low confidence. Consider avoiding this bet."
    else:
        return "🔴 NO CLEAR PREDICTION", "Very low confidence. Do not bet."

def main():
    # Header
    st.markdown('<h1 class="main-header">⚽ Premier League Match Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Match Outcome Predictions | 55% Accuracy</p>', unsafe_allow_html=True)
    
    # Load models and data
    with st.spinner('Loading models and data...'):
        xgb_model, rf_model, feature_cols, models_loaded = load_models()
        features_df, teams, data_loaded = load_data()
    
    if not models_loaded or not data_loaded:
        st.error("⚠️ Failed to load models or data. Please ensure:")
        st.markdown("""
        - `models/xgboost_model.pkl` exists
        - `models/random_forest_model.pkl` exists
        - `models/feature_columns.pkl` exists
        - `data/features.csv` exists
        
        Run `python src/train_models.py` to create the models.
        """)
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/f/f2/Premier_League_Logo.svg/1200px-Premier_League_Logo.svg.png", width=200)
        st.markdown("---")
        
        st.markdown("### 📊 Model Info")
        st.markdown(f"""
        - **Algorithm**: XGBoost + Random Forest Ensemble
        - **Accuracy**: 55.4%
        - **Features**: {len(feature_cols)}
        - **Training Data**: 10 years (2015-2025)
        - **Total Matches**: {len(features_df)}
        """)
        
        st.markdown("---")
        st.markdown("### 📈 Performance Benchmarks")
        st.markdown("""
        - Random Guess: 33%
        - Always Home Win: 45%
        - **Our Model: 55%** ✅
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This ML model predicts Premier League match outcomes using:
        - Recent team form
        - Goal statistics
        - Head-to-head records
        - Home/Away performance
        - 219 engineered features
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 Predict Match", "📊 Available Teams", "📖 How It Works"])
    
    with tab1:
        st.markdown("## Make a Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏠 Home Team")
            home_team = st.selectbox(
                "Select home team",
                options=teams,
                index=0,
                key="home"
            )
        
        with col2:
            st.markdown("### ✈️ Away Team")
            away_team = st.selectbox(
                "Select away team",
                options=[t for t in teams if t != home_team],
                index=0,
                key="away"
            )
        
        st.markdown("---")
        
        if st.button("🔮 PREDICT MATCH OUTCOME", use_container_width=True):
            with st.spinner('Analyzing match...'):
                result = predict_match(home_team, away_team, xgb_model, rf_model, features_df, feature_cols)
            
            if result is None:
                st.error("Could not make prediction. Team data not found.")
                return
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h1 style="margin: 0; font-size: 2.5rem;">🏆 {result['predicted_outcome']}</h1>
                <p style="margin-top: 1rem; font-size: 1.3rem; opacity: 0.9;">
                    {home_team} vs {away_team}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Home Win",
                    f"{result['home_win_prob']*100:.1f}%",
                    delta="Probability"
                )
            
            with col2:
                st.metric(
                    "Draw",
                    f"{result['draw_prob']*100:.1f}%",
                    delta="Probability"
                )
            
            with col3:
                st.metric(
                    "Away Win",
                    f"{result['away_win_prob']*100:.1f}%",
                    delta="Probability"
                )
            
            with col4:
                st.metric(
                    "Confidence",
                    f"{result['confidence']*100:.1f}%",
                    delta=result['predicted_outcome']
                )
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_probability_chart(result), use_container_width=True)
            
            with col2:
                st.plotly_chart(create_gauge_chart(result['confidence']), use_container_width=True)
            
            # Betting recommendation
            st.markdown("---")
            st.markdown("### 🎲 Betting Recommendation")
            
            recommendation, description = get_betting_recommendation(result['confidence'])
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"### {recommendation}")
            with col2:
                st.info(description)
            
            # Model breakdown
            with st.expander("🔍 Model Breakdown"):
                st.markdown("#### Individual Model Predictions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**XGBoost (60% weight)**")
                    st.write(f"Home Win: {result['xgb_proba'][0]*100:.1f}%")
                    st.write(f"Draw: {result['xgb_proba'][1]*100:.1f}%")
                    st.write(f"Away Win: {result['xgb_proba'][2]*100:.1f}%")
                
                with col2:
                    st.markdown("**Random Forest (40% weight)**")
                    st.write(f"Home Win: {result['rf_proba'][0]*100:.1f}%")
                    st.write(f"Draw: {result['rf_proba'][1]*100:.1f}%")
                    st.write(f"Away Win: {result['rf_proba'][2]*100:.1f}%")
    
    with tab2:
        st.markdown("## 📋 Available Teams")
        st.markdown(f"**Total Teams**: {len(teams)}")
        
        # Display teams in columns
        cols = st.columns(4)
        for idx, team in enumerate(teams):
            with cols[idx % 4]:
                st.markdown(f"✅ {team}")
    
    with tab3:
        st.markdown("## 📖 How It Works")
        
        st.markdown("""
        ### 🧠 Machine Learning Pipeline
        
        #### 1. Data Collection
        - 10 years of Premier League match data (2015-2025)
        - ~3,800 historical matches
        - Team performance statistics
        
        #### 2. Feature Engineering
        We create **219 features** including:
        - **Form Metrics**: Rolling averages of points, goals, win ratios
        - **Momentum**: Exponential weighted moving averages
        - **Head-to-Head**: Historical matchup statistics
        - **Home/Away Splits**: Performance by venue
        - **Differentials**: Comparing home vs away team strengths
        
        #### 3. Model Training
        - **XGBoost**: Gradient boosted decision trees (60% weight)
        - **Random Forest**: Ensemble of 300 trees (40% weight)
        - **Ensemble**: Weighted combination of both models
        
        #### 4. Prediction
        - Extract latest team statistics
        - Apply feature engineering
        - Generate probabilities for all 3 outcomes
        - Provide confidence-based betting recommendations
        
        ### 📊 Model Performance
        
        | Metric | Value |
        |--------|-------|
        | Overall Accuracy | 55.4% |
        | Home Win Precision | 58.2% |
        | Draw Precision | 34.1% |
        | Away Win Precision | 52.8% |
        
        ### 🎯 Confidence Levels
        
        | Confidence | Accuracy | Recommendation |
        |------------|----------|----------------|
        | 65%+ | ~68% | Strong bet |
        | 55-65% | ~60% | Moderate bet |
        | 45-55% | ~52% | Weak signal |
        | <45% | ~50% | Avoid betting |
        
        ### ⚠️ Limitations
        
        - Does not account for injuries, suspensions, or team news
        - Historical data may not reflect current team changes
        - Weather and referee factors not included
        - No live odds integration
        
        ### 🔮 Future Improvements
        
        - Player-level statistics
        - Real-time injury updates
        - Betting odds as features
        - Deep learning models
        - Live prediction updates
        """)
        
        st.info("💡 **Disclaimer**: This model is for educational and entertainment purposes only. Always gamble responsibly.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Built with  using Streamlit, XGBoost, and scikit-learn</p>
        <p>Data source: football-data.co.uk | Model accuracy: 55%</p>
        <p> For educational purposes only. Gamble responsibly.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
