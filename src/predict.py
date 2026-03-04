import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_models():
    try:
        xgb_model = joblib.load('models/xgboost_model.pkl')
        rf_model = joblib.load('models/random_forest_model.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        print(" Models loaded successfully")
        return xgb_model, rf_model, feature_cols
    except FileNotFoundError as e:
        print(f" Error: Model files not found!")
        print(f"   Please run: python src/train_models.py")
        return None, None, None

def get_team_latest_stats(features_df, team_name, is_home=True):
    if is_home:
        team_matches = features_df[features_df['HomeTeam'] == team_name]
    else:
        team_matches = features_df[features_df['AwayTeam'] == team_name]
    
    if len(team_matches) == 0:
        raise ValueError(f"Team '{team_name}' not found in data!")
    
    team_matches = team_matches.sort_values('Date', ascending=False)
    latest = team_matches.iloc[0]
    
    return latest

def predict_match(home_team, away_team, use_ensemble=True):
    xgb_model, rf_model, feature_cols = load_models()
    if xgb_model is None:
        return None
    
    try:
        features_df = pd.read_csv('data/features.csv')
        features_df['Date'] = pd.to_datetime(features_df['Date'])
    except FileNotFoundError:
        print(" Error: data/features.csv not found!")
        return None
    
    print(f"\n{'='*60}")
    print(f"PREDICTING: {home_team} vs {away_team}")
    print(f"{'='*60}")
    
    try:
        home_stats = get_team_latest_stats(features_df, home_team, is_home=True)
        away_stats = get_team_latest_stats(features_df, away_team, is_home=False)
        
        match_features = pd.DataFrame(index=[0])
        
        for col in feature_cols:
            if '_home' in col:
                try:
                    match_features[col] = home_stats[col]
                except KeyError:
                    match_features[col] = 0
            elif '_away' in col:
                try:
                    match_features[col] = away_stats[col]
                except KeyError:
                    match_features[col] = 0
            else:
                match_features[col] = 0
        
        for col in feature_cols:
            if col not in match_features.columns:
                match_features[col] = 0
        
        match_features = match_features[feature_cols]
        
        xgb_proba = xgb_model.predict_proba(match_features)[0]
        
        if use_ensemble and rf_model is not None:
            rf_proba = rf_model.predict_proba(match_features)[0]
            proba = 0.6 * xgb_proba + 0.4 * rf_proba
            model_used = "Ensemble"
        else:
            proba = xgb_proba
            model_used = "XGBoost"
        
        prediction_idx = np.argmax(proba)
        outcomes = ['Home Win', 'Draw', 'Away Win']
        predicted_outcome = outcomes[prediction_idx]
        confidence = proba[prediction_idx]
        
        print(f"\nModel: {model_used}")
        print(f"\nPredicted Outcome: {predicted_outcome}")
        print(f"Confidence: {confidence:.1%}")
        print(f"\nProbabilities:")
        print(f"  Home Win ({home_team}): {proba[0]:.1%}")
        print(f"  Draw:                    {proba[1]:.1%}")
        print(f"  Away Win ({away_team}): {proba[2]:.1%}")
        
        print(f"\nBetting Recommendation:")
        if confidence >= 0.65:
            print(f" Strong bet on {predicted_outcome}")
        elif confidence >= 0.55:
            print(f" Moderate bet on {predicted_outcome}")
        elif confidence >= 0.45:
            print(f" Weak signal - Consider avoiding")
        else:
            print(f" No clear prediction - Avoid betting")
        
        print(f"{'='*60}\n")
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_outcome': predicted_outcome,
            'confidence': confidence,
            'probabilities': {
                'home_win': proba[0],
                'draw': proba[1],
                'away_win': proba[2]
            },
            'model': model_used
        }
        
    except ValueError as e:
        print(f" Error: {e}")
        return None
    except Exception as e:
        print(f" Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

def show_available_teams():
    try:
        features_df = pd.read_csv('data/features.csv')
        home_teams = set(features_df['HomeTeam'].unique())
        away_teams = set(features_df['AwayTeam'].unique())
        all_teams = sorted(home_teams | away_teams)
        
        print("\n" + "="*60)
        print("AVAILABLE TEAMS")
        print("="*60)
        for i, team in enumerate(all_teams, 1):
            print(f"{i:2d}. {team}")
        print(f"\nTotal: {len(all_teams)} teams")
        print("="*60)
        
        return all_teams
    except FileNotFoundError:
        print(" Error: data/features.csv not found!")
        return []

def predict_multiple_matches(matches):
    results = []
    for home, away in matches:
        result = predict_match(home, away, use_ensemble=True)
        if result:
            results.append(result)
    return results

def main():
    print("\n" + "="*60)
    print(" "*15 + "MATCH PREDICTION SYSTEM")
    print("="*60)
    
    teams = show_available_teams()
    
    if not teams:
        return
    
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    example_matches = [
        ("Arsenal", "Chelsea"),
        ("Liverpool", "Man United"),
        ("Man City", "Tottenham")
    ]
 
    for home, away in example_matches:
        try:
            predict_match(home, away, use_ensemble=True)
        except:
            print(f" Could not predict {home} vs {away} - team names may not exist\n")
    
    print("\n" + "="*60)
    print("CUSTOM PREDICTIONS")
    print("="*60)
    print("\nTo predict a custom match, modify the example_matches list in this file,")
    print("or use Python interactively:")
    print("\n  >>> from predict import predict_match")
    print("  >>> predict_match('Arsenal', 'Chelsea')")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()