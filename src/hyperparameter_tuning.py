import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, make_scorer
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import re
from datetime import datetime

def sanitize_feature_names(feature_cols):
    sanitized = []
    for col in feature_cols:
        clean_name = col.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_')
        sanitized.append(clean_name)
    return sanitized

def load_and_prepare_data(filepath: str = 'data/features.csv'):
    print("\n" + "="*80)
    print("LOADING DATA FOR HYPERPARAMETER TUNING")
    print("="*80)
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f" Loaded {len(df)} matches")
    
    exclude_cols = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
        'home_win', 'away_win', 'draw', 'home_points', 'away_points', 'target',
        'Div', 'Season', 'Time', 'Referee', 'HTR', 'HTHG', 'HTAG',
        'Date_home', 'Date_away', 'Team_home', 'Team_away', 
        'Opponent_home', 'Opponent_away'
    ]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    original_feature_cols = feature_cols.copy()
    feature_cols_clean = sanitize_feature_names(feature_cols)
    feature_name_mapping = dict(zip(original_feature_cols, feature_cols_clean))
    
    df = df.rename(columns=feature_name_mapping)
    feature_cols = feature_cols_clean
    
    print(f" Features: {len(feature_cols)}")
    
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df['target'] = df['FTR'].map(target_map)
    df = df.dropna(subset=['target'])
    
    X = df[feature_cols].fillna(0)
    y = df['target'].astype(int)
    
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        raise ValueError(f"Non-numeric columns found: {list(non_numeric)}")
    
    return X, y, feature_cols

def tune_xgboost(X, y, n_iter: int = 50):
    print("\n" + "="*80)
    print("TUNING XGBOOST HYPERPARAMETERS")
    print("="*80)
    
    param_grid = {
        'max_depth': [4, 6, 8, 10, 12],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'n_estimators': [200, 300, 400, 500],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 2.0]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    print(f"\nSearching {n_iter} parameter combinations...")
    print(f"Using TimeSeriesSplit with 5 folds")
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    search.fit(X, y)
    
    print("\n" + "="*80)
    print("TUNING RESULTS ARE")
    print("="*80)
    print(f"\n✓ Best CV Score: {search.best_score_:.4f} ({search.best_score_*100:.2f}%)")
    print(f"\nBest Parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param:20s}: {value}")
    
    print("\n" + "-"*80)
    print("Top 5 Configurations:")
    print("-"*80)
    results_df = pd.DataFrame(search.cv_results_)
    top_5 = results_df.nsmallest(5, 'rank_test_score')[
        ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
    ]
    
    for idx, row in top_5.iterrows():
        print(f"\nRank {int(row['rank_test_score'])}:")
        print(f"  Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        print(f"  Params: {row['params']}")
    
    return search.best_estimator_, search.best_params_

def tune_random_forest(X, y, n_iter: int = 30):
    print("\n" + "="*80)
    print("TUNING RANDOM FOREST HYPERPARAMETERS")
    print("="*80)
    
    param_grid = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [2, 4, 6, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    print(f"\nSearching {n_iter} parameter combinations...")
    print(f"Using TimeSeriesSplit with 5 folds")
    print(f"This may take 10-20 minutes...\n")
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    search.fit(X, y)
    
    print("\n" + "="*80)
    print("TUNING RESULTS")
    print("="*80)
    print(f"\n Best CV Score: {search.best_score_:.4f} ({search.best_score_*100:.2f}%)")
    print(f"\nBest Parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param:20s}: {value}")
    
    return search.best_estimator_, search.best_params_

def save_tuned_models(xgb_model, xgb_params, rf_model, rf_params, feature_cols):
    os.makedirs('models/tuned', exist_ok=True)
    
    joblib.dump(xgb_model, 'models/tuned/xgboost_tuned.pkl')
    joblib.dump(rf_model, 'models/tuned/random_forest_tuned.pkl')
    joblib.dump(feature_cols, 'models/tuned/feature_columns.pkl')
    
    with open('models/tuned/best_parameters.txt', 'w') as f:
        f.write("TUNED HYPERPARAMETERS\n")
        f.write("="*80 + "\n")
        f.write(f"Tuned on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("XGBoost Best Parameters:\n")
        f.write("-"*80 + "\n")
        for param, value in xgb_params.items():
            f.write(f"  {param:20s}: {value}\n")
        
        f.write("\n\nRandom Forest Best Parameters:\n")
        f.write("-"*80 + "\n")
        for param, value in rf_params.items():
            f.write(f"  {param:20s}: {value}\n")
    
    print("\n Tuned models saved:")
    print("  - models/tuned/xgboost_tuned.pkl")
    print("  - models/tuned/random_forest_tuned.pkl")
    print("  - models/tuned/feature_columns.pkl")
    print("  - models/tuned/best_parameters.txt")

def main():
    print("\n" + "="*80)
    print(" "*20 + "HYPERPARAMETER TUNING PIPELINE")
    print("="*80)
    
    if not os.path.exists('data/features.csv'):
        print("\n Error: data/features.csv not found!")
        print("   Please run feature_engineering.py first.")
        return
    
    X, y, feature_cols = load_and_prepare_data()
    
    print("\n" + "="*80)
    print("STEP 1: TUNING XGBOOST (This will take 10-30 minutes)")
    print("="*80)
    xgb_model, xgb_params = tune_xgboost(X, y, n_iter=50)
    
    print("\n" + "="*80)
    print("STEP 2: TUNING RANDOM FOREST (This will take 10-20 minutes)")
    print("="*80)
    rf_model, rf_params = tune_random_forest(X, y, n_iter=30)
    
    save_tuned_models(xgb_model, xgb_params, rf_model, rf_params, feature_cols)
    
    print("\n" + "="*80)
    print("TUNING COMPLETE!")
    print("="*80)
    print("\n Next steps:")
    print("  1. Check models/tuned/best_parameters.txt for the best parameters")
    print("  2. Update train_models.py with these parameters")
    print("  3. Re-train with the optimized parameters")
    print("\n Tuned models saved and ready to use!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
