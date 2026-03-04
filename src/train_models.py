import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import re
from datetime import datetime

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def sanitize_feature_names(feature_cols):
    sanitized = []
    for col in feature_cols:
        clean_name = col.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_')
        sanitized.append(clean_name)
    return sanitized

def load_and_split_data(filepath: str, test_size: float = 0.2):
    """Load features and split by time (crucial for sports betting!)"""
    print("\n" + "="*80)
    print("STEP 1: LOADING AND SPLITTING DATA")
    print("="*80)
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} matches")
    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    exclude_cols = [
        'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
        'home_win', 'away_win', 'draw', 'home_points', 'away_points', 'target',
        'Div', 'Season', 'Time', 'Referee', 'HTR', 'HTHG', 'HTAG',
        'Date_home', 'Date_away', 'Team_home', 'Team_away', 
        'Opponent_home', 'Opponent_away'
    ]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  Feature columns: {len(feature_cols)}")
    
    original_feature_cols = feature_cols.copy()
    feature_cols_clean = sanitize_feature_names(feature_cols)
    
    feature_name_mapping = dict(zip(original_feature_cols, feature_cols_clean))
    
    df = df.rename(columns=feature_name_mapping)
    feature_cols = feature_cols_clean
    
    print(f"  Sanitized feature names for XGBoost compatibility")
    
    if 'FTR' not in df.columns:
        raise ValueError("FTR column not found!")
    
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df['target'] = df['FTR'].map(target_map)
    
    if df['target'].isna().any():
        print(f"\n⚠ Warning: {df['target'].isna().sum()} invalid FTR values found")
        df = df.dropna(subset=['target'])
        print(f"  New size: {len(df)}")
    
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['target'].astype(int)
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df['target'].astype(int)
    
    print(f"\n Train set: {len(train_df)} matches")
    print(f"  Period: {train_df['Date'].min().date()} to {train_df['Date'].max().date()}")
    print(f"\n Test set: {len(test_df)} matches")
    print(f"  Period: {test_df['Date'].min().date()} to {test_df['Date'].max().date()}")
    
    print(f"\nClass distribution:")
    dist = y_train.value_counts(normalize=True).sort_index()
    print(f"  Home Win: {dist[0]:.1%}")
    print(f"  Draw:     {dist[1]:.1%}")
    print(f"  Away Win: {dist[2]:.1%}")
    
    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        raise ValueError(f"Non-numeric columns found: {list(non_numeric)}")
    
    print(f"\n All features are numeric and ready for training")
    
    return X_train, X_test, y_train, y_test, feature_cols, test_df

def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n" + "="*80)
    print("STEP 2: TRAINING XGBOOST MODEL")
    print("="*80)
    
    params = {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(**params)
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"\n Training complete!")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    if train_acc - test_acc > 0.05:
        print(f"Overfitting gap: {(train_acc-test_acc)*100:.2f}%")
    
    return model, test_preds, test_proba, test_acc

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier"""
    print("\n" + "="*80)
    print("STEP 3: TRAINING RANDOM FOREST MODEL")
    print("="*80)
    
    params = {
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_proba = model.predict_proba(X_train)
    test_proba = model.predict_proba(X_test)
    
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"\nTraining complete!")
    print(f"  Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    if train_acc - test_acc > 0.05:
        print(f"  ⚠ Overfitting gap: {(train_acc-test_acc)*100:.2f}%")
    
    return model, test_preds, test_proba, test_acc

def create_ensemble(xgb_proba, rf_proba, y_test, weights=[0.6, 0.4]):
    print("\n" + "="*80)
    print("STEP 4: CREATING ENSEMBLE MODEL")
    print("="*80)
    
    ensemble_proba = (weights[0] * xgb_proba + weights[1] * rf_proba)
    ensemble_pred = np.argmax(ensemble_proba, axis=1)
    
    accuracy = accuracy_score(y_test, ensemble_pred)
    
    print(f"\nEnsemble Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return ensemble_pred, ensemble_proba, accuracy

def print_detailed_metrics(y_test, y_pred, model_name):
    print(f"\n{model_name} - Detailed Metrics:")
    print("-" * 80)
    
    report = classification_report(
        y_test, y_pred, 
        target_names=['Home Win', 'Draw', 'Away Win'],
        output_dict=True
    )
    
    print("\nPer-Class Performance:")
    for outcome in ['Home Win', 'Draw', 'Away Win']:
        print(f"\n  {outcome}:")
        print(f"    Precision: {report[outcome]['precision']:.4f}")
        print(f"    Recall:    {report[outcome]['recall']:.4f}")
        print(f"    F1-Score:  {report[outcome]['f1-score']:.4f}")
        print(f"    Support:   {int(report[outcome]['support'])}")

def plot_confusion_matrix(y_test, y_pred, title, save_path):
    cm = confusion_matrix(y_test, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Home', 'Draw', 'Away'],
                yticklabels=['Home', 'Draw', 'Away'])
    ax1.set_title(f'{title} - Counts')
    ax1.set_ylabel('Actual')
    ax1.set_xlabel('Predicted')
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', ax=ax2,
                xticklabels=['Home', 'Draw', 'Away'],
                yticklabels=['Home', 'Draw', 'Away'])
    ax2.set_title(f'{title} - Percentages')
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_cols, model_name, top_n=20):
    """Plot feature importance"""
    if not hasattr(model, 'feature_importances_'):
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_cols[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features - {model_name}')
    plt.tight_layout()
    
    save_path = f'models/{model_name.lower().replace(" ", "_")}_features.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nTop 10 Features ({model_name}):")
    top_10 = np.argsort(importances)[-10:][::-1]
    for i, idx in enumerate(top_10, 1):
        print(f"  {i:2d}. {feature_cols[idx]:35s} {importances[idx]:.4f}")

def analyze_betting(test_df, predictions, probabilities, model_name):
    print(f"\n{'='*80}")
    print(f"BETTING ANALYSIS - {model_name}")
    print("="*80)
    
    test_df = test_df.copy()
    test_df['prediction'] = predictions
    test_df['max_proba'] = np.max(probabilities, axis=1)
    
    target_map = {'H': 0, 'D': 1, 'A': 2}
    test_df['actual'] = test_df['FTR'].map(target_map)
    test_df['correct'] = (test_df['prediction'] == test_df['actual']).astype(int)
    
    for threshold in [0.50, 0.55, 0.60, 0.65]:
        bets = test_df[test_df['max_proba'] >= threshold]
        
        if len(bets) > 0:
            correct = bets['correct'].sum()
            total = len(bets)
            acc = correct / total
            coverage = total / len(test_df)
            
            print(f"\nConfidence ≥ {threshold:.0%}:")
            print(f"  Bets:     {total} ({coverage:.1%})")
            print(f"  Accuracy: {acc:.1%} ({correct}/{total})")

def compare_models(xgb_acc, rf_acc, ens_acc):
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    models = ['XGBoost', 'Random Forest', 'Ensemble']
    accs = [xgb_acc, rf_acc, ens_acc]
    
    best_idx = np.argmax(accs)
    
    for model, acc in zip(models, accs):
        marker = " ⭐" if acc == accs[best_idx] else ""
        print(f"  {model:15s}: {acc*100:.2f}%{marker}")

def save_models(xgb_model, rf_model, feature_cols):
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    
    metadata = {
        'training_date': datetime.now().isoformat(),
        'n_features': len(feature_cols)
    }
    joblib.dump(metadata, 'models/metadata.pkl')
    
    print("\n Models saved to models/")

def main():
    print("\n" + "="*80)
    print(" "*20 + "FOOTBALL PREDICTION MODEL")
    print("="*80)
    
    if not os.path.exists('data/features.csv'):
        print("\n Error: data/features.csv not found!")
        print("   Run: python src/feature_engineering.py")
        return
    
    X_train, X_test, y_train, y_test, feature_cols, test_df = \
        load_and_split_data('data/features.csv', test_size=0.2)
    
    xgb_model, xgb_pred, xgb_proba, xgb_acc = \
        train_xgboost(X_train, y_train, X_test, y_test)
    print_detailed_metrics(y_test, xgb_pred, 'XGBoost')
    
    rf_model, rf_pred, rf_proba, rf_acc = \
        train_random_forest(X_train, y_train, X_test, y_test)
    print_detailed_metrics(y_test, rf_pred, 'Random Forest')
    
    ens_pred, ens_proba, ens_acc = \
        create_ensemble(xgb_proba, rf_proba, y_test)
    print_detailed_metrics(y_test, ens_pred, 'Ensemble')
    
    compare_models(xgb_acc, rf_acc, ens_acc)
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    os.makedirs('models', exist_ok=True)
    
    plot_confusion_matrix(y_test, xgb_pred, 'XGBoost', 'models/xgb_cm.png')
    plot_confusion_matrix(y_test, rf_pred, 'Random Forest', 'models/rf_cm.png')
    plot_confusion_matrix(y_test, ens_pred, 'Ensemble', 'models/ens_cm.png')
    print(" Confusion matrices saved")
    
    plot_feature_importance(xgb_model, feature_cols, 'XGBoost', top_n=20)
    plot_feature_importance(rf_model, feature_cols, 'Random Forest', top_n=20)
    print(" Feature importance saved")
    
    analyze_betting(test_df, xgb_pred, xgb_proba, 'XGBoost')
    analyze_betting(test_df, ens_pred, ens_proba, 'Ensemble')
    
    save_models(xgb_model, rf_model, feature_cols)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  XGBoost:       {xgb_acc*100:.2f}%")
    print(f"  Random Forest: {rf_acc*100:.2f}%")
    print(f"  Ensemble:      {ens_acc*100:.2f}%")
    
    if ens_acc < 0.52:
        print("\n Results close to random - consider more data/features")
    elif ens_acc > 0.56:
        print("\n Excellent results! Consider real betting backtests")
    else:
        print("\n Good results! Model shows predictive power")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()