import pandas as pd
import numpy as np
from typing import List

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load and prepare the raw match data."""
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Date"])
    return df

def create_match_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary outcome columns and points."""
    df = df.copy()
    df["home_win"] = (df["FTR"] == "H").astype(int)
    df["away_win"] = (df["FTR"] == "A").astype(int)
    df["draw"] = (df["FTR"] == "D").astype(int)
    df["home_points"] = df["home_win"] * 3 + df["draw"]
    df["away_points"] = df["away_win"] * 3 + df["draw"]
    return df

def create_team_perspective_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert match data to team perspective (one row per team per match)."""
    home = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", 
                "home_points", "home_win"]].copy()
    home.columns = ["Date", "Team", "Opponent", "GoalsFor", "GoalsAgainst", 
                    "Points", "Win"]
    home["is_home"] = 1

    away = df[["Date", "AwayTeam", "HomeTeam", "FTAG", "FTHG", 
                "away_points", "away_win"]].copy()
    away.columns = ["Date", "Team", "Opponent", "GoalsFor", "GoalsAgainst", 
                    "Points", "Win"]
    away["is_home"] = 0

    team_df = pd.concat([home, away], ignore_index=True).sort_values(["Team", "Date"]).reset_index(drop=True)
    return team_df

def create_rolling_features(team_df: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
    """Create rolling window features for multiple time windows."""
    team_df = team_df.copy()
    
    for window in windows:
        # Basic rolling averages
        for col in ["GoalsFor", "GoalsAgainst", "Points"]:
            team_df[f"{col}_avg_{window}"] = (
                team_df.groupby("Team")[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
        
        # Goal difference
        team_df[f"goal_diff_avg_{window}"] = (
            team_df[f"GoalsFor_avg_{window}"] - team_df[f"GoalsAgainst_avg_{window}"]
        )
        
        # Win ratio
        team_df[f"win_ratio_{window}"] = (
            team_df.groupby("Team")["Win"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        
        # Draw ratio
        team_df[f"draw_ratio_{window}"] = (
            team_df.groupby("Team")["Points"]
            .transform(lambda x: ((x.shift(1) == 1).rolling(window, min_periods=1).mean()))
        )
        
        # Clean sheet ratio
        team_df[f"clean_sheet_ratio_{window}"] = (
            team_df.groupby("Team")["GoalsAgainst"]
            .transform(lambda x: (x.shift(1) == 0).rolling(window, min_periods=1).mean())
        )
        
        # Failed to score ratio
        team_df[f"failed_to_score_ratio_{window}"] = (
            team_df.groupby("Team")["GoalsFor"]
            .transform(lambda x: (x.shift(1) == 0).rolling(window, min_periods=1).mean())
        )
        
        # Standard deviation
        team_df[f"points_std_{window}"] = (
            team_df.groupby("Team")["Points"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).std().fillna(0))
        )
        
        # Max goals scored/conceded
        team_df[f"max_goals_scored_{window}"] = (
            team_df.groupby("Team")["GoalsFor"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
        )
        
        team_df[f"max_goals_conceded_{window}"] = (
            team_df.groupby("Team")["GoalsAgainst"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
        )
    
    # Exponentially weighted moving average
    team_df["points_ewm_5"] = (
        team_df.groupby("Team")["Points"]
        .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    )
    
    team_df["goals_for_ewm_5"] = (
        team_df.groupby("Team")["GoalsFor"]
        .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    )
    
    team_df["goals_against_ewm_5"] = (
        team_df.groupby("Team")["GoalsAgainst"]
        .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    )
    
    # Win streak
    team_df["win_streak"] = (
        team_df.groupby("Team")["Win"]
        .transform(lambda x: calculate_streak_fixed(x.shift(1)))
    )
    
    # Unbeaten streak
    team_df["unbeaten_streak"] = (
        team_df.groupby("Team")["Points"]
        .transform(lambda x: calculate_unbeaten_streak(x.shift(1)))
    )
    
    # Days since last match
    team_df["days_rest"] = (
        team_df.groupby("Team")["Date"]
        .diff()
        .dt.days
        .fillna(7)
    )
    
    # Points sum
    for window in [3, 5, 10]:
        team_df[f"points_sum_{window}"] = (
            team_df.groupby("Team")["Points"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
        )
    
    # Goal trends
    team_df["goals_for_trend_5"] = (
        team_df.groupby("Team")["GoalsFor"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean() - 
                           x.shift(6).rolling(5, min_periods=3).mean())
        .fillna(0)
    )
    
    team_df["goals_against_trend_5"] = (
        team_df.groupby("Team")["GoalsAgainst"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean() - 
                           x.shift(6).rolling(5, min_periods=3).mean())
        .fillna(0)
    )
    
    return team_df

def calculate_streak_fixed(wins: pd.Series) -> pd.Series:
    """Calculate current win streak."""
    streak = []
    current_streak = 0
    
    for val in wins:
        if pd.isna(val):
            current_streak = 0
        elif val == 1:
            current_streak += 1
        else:
            current_streak = 0
        streak.append(current_streak)
    
    return pd.Series(streak, index=wins.index)

def calculate_unbeaten_streak(points: pd.Series) -> pd.Series:
    """Calculate current unbeaten streak."""
    streak = []
    current_streak = 0
    
    for val in points:
        if pd.isna(val):
            current_streak = 0
        elif val > 0:
            current_streak += 1
        else:
            current_streak = 0
        streak.append(current_streak)
    
    return pd.Series(streak, index=points.index)

def create_head_to_head_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """Create head-to-head features."""
    team_df = team_df.copy()
    
    team_df["h2h_win_pct_3"] = (
        team_df.groupby(["Team", "Opponent"])["Win"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    
    team_df["h2h_goals_for_avg_3"] = (
        team_df.groupby(["Team", "Opponent"])["GoalsFor"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    
    team_df["h2h_goals_against_avg_3"] = (
        team_df.groupby(["Team", "Opponent"])["GoalsAgainst"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    
    team_df["h2h_last_result"] = (
        team_df.groupby(["Team", "Opponent"])["Points"]
        .transform(lambda x: x.shift(1))
        .fillna(1)
    )
    
    return team_df

def merge_features(df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """Merge team features back to match level."""
    home_feats = team_df[team_df["is_home"] == 1].copy()
    away_feats = team_df[team_df["is_home"] == 0].copy()
    
    # CRITICAL: Only select NUMERIC feature columns (exclude Date, Team, Opponent, etc.)
    exclude_cols = ["Date", "Team", "Opponent", "GoalsFor", "GoalsAgainst", 
                    "Points", "Win", "is_home"]
    feature_cols = [col for col in team_df.columns if col not in exclude_cols]
    
    print(f"Merging {len(feature_cols)} feature columns...")
    
    # Merge home features
    features = df.merge(
        home_feats[["Date", "Team"] + feature_cols],
        left_on=["Date", "HomeTeam"],
        right_on=["Date", "Team"],
        how="left"
    )
    
    # Merge away features
    features = features.merge(
        away_feats[["Date", "Team"] + feature_cols],
        left_on=["Date", "AwayTeam"],
        right_on=["Date", "Team"],
        how="left",
        suffixes=("_home", "_away")
    )
    
    # Drop the Team columns created by merge
    features = features.drop(columns=["Team_home", "Team_away"], errors="ignore")
    
    return features

def create_differential_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create differential features."""
    features = features.copy()
    
    for window in [3, 5, 10]:
        features[f"form_diff_{window}"] = (
            features[f"Points_avg_{window}_home"] - features[f"Points_avg_{window}_away"]
        )
        
        features[f"goal_diff_form_{window}"] = (
            features[f"goal_diff_avg_{window}_home"] - features[f"goal_diff_avg_{window}_away"]
        )
        
        features[f"win_ratio_diff_{window}"] = (
            features[f"win_ratio_{window}_home"] - features[f"win_ratio_{window}_away"]
        )
        
        features[f"attack_diff_{window}"] = (
            features[f"GoalsFor_avg_{window}_home"] - features[f"GoalsFor_avg_{window}_away"]
        )
        
        features[f"defense_diff_{window}"] = (
            features[f"GoalsAgainst_avg_{window}_away"] - features[f"GoalsAgainst_avg_{window}_home"]
        )
    
    features["consistency_diff_5"] = (
        features["points_std_5_away"] - features["points_std_5_home"]
    )
    
    features["momentum_diff"] = (
        features["points_ewm_5_home"] - features["points_ewm_5_away"]
    )
    
    features["rest_advantage"] = (
        features["days_rest_home"] - features["days_rest_away"]
    )
    
    features["win_streak_diff"] = (
        features["win_streak_home"] - features["win_streak_away"]
    )
    
    features["unbeaten_streak_diff"] = (
        features["unbeaten_streak_home"] - features["unbeaten_streak_away"]
    )
    
    features["clean_sheet_diff_5"] = (
        features["clean_sheet_ratio_5_home"] - features["clean_sheet_ratio_5_away"]
    )
    
    features["h2h_advantage"] = (
        features["h2h_win_pct_3_home"] - features["h2h_win_pct_3_away"]
    )
    
    features["attack_trend_diff"] = (
        features["goals_for_trend_5_home"] - features["goals_for_trend_5_away"]
    )
    
    features["defense_trend_diff"] = (
        features["goals_against_trend_5_away"] - features["goals_against_trend_5_home"]
    )
    
    return features

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n[1/7] Loading data...")
    df = load_and_prepare_data("data/merged_matches.csv")
    print(f"  ✓ Loaded {len(df)} matches")
    
    # Create match outcomes
    print("[2/7] Creating match outcomes...")
    df = create_match_outcomes(df)
    
    # Convert to team perspective
    print("[3/7] Converting to team perspective...")
    team_df = create_team_perspective_df(df)
    print(f"  ✓ Created {len(team_df)} team-match records")
    
    # Create rolling features
    print("[4/7] Creating rolling features...")
    team_df = create_rolling_features(team_df, windows=[3, 5, 10])
    
    # Create head-to-head features
    print("[5/7] Creating head-to-head features...")
    team_df = create_head_to_head_features(team_df)
    
    # Merge back to match level
    print("[6/7] Merging features to match level...")
    features = merge_features(df, team_df)
    
    # Create differential features
    print("[7/7] Creating differential features...")
    features = create_differential_features(features)
    
    # Data quality check
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    print(f"Rows before cleaning: {len(features)}")
    print(f"Rows with any NaN: {features.isna().any(axis=1).sum()}")
    
    # Clean up
    nan_threshold = 0.3
    nan_count = features.isna().sum(axis=1)
    features_clean = features[nan_count < (len(features.columns) * nan_threshold)].copy()
    
    print(f"Rows after cleaning: {len(features_clean)}")
    
    # Fill remaining NaN with 0
    features_clean = features_clean.fillna(0)
    
    # Save
    features_clean.to_csv("data/features.csv", index=False)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"✓ Feature dataset created: {features_clean.shape}")
    
    # Count feature columns (exclude metadata)
    metadata_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
                     'home_win', 'away_win', 'draw', 'home_points', 'away_points']
    feature_cols = [c for c in features_clean.columns if c not in metadata_cols]
    
    print(f"✓ Number of feature columns: {len(feature_cols)}")
    print(f"✓ Date range: {features_clean['Date'].min()} to {features_clean['Date'].max()}")
    print(f"✓ Saved to: data/features.csv")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()