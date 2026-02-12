import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import joblib
import os
import re
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

def sanitize_feature_names(feature_cols):
    """ Feature names for XGBoost compatibility"""
    sanitized = []
    for col in feature_cols:
        clean_name = col.replace('[', '_').replace(']', '_').replace('<', '_lt_').replace('>', '_gt_')
        sanitized.append(clean_name)
    return sanitized

def load_data(filepath: str = "data/features.csv"):
    """Load and prepare data for SHAP analysis"""
    print("\n" + "="*80)
    print("LOADING DATA FOR SHAP ANALYSIS")
    print("="*80)
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✓ Loaded {len(df)} matches")
    
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df['target'] = df['FTR'].map(target_map)
    df = df.dropna(subset=['target'])
    
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
    
    X = df[feature_cols].fillna(0)
    y = df['target'].astype(int)
    
    print(f" Features: {len(feature_cols)}")
    print(f" Target distribution:")
    dist = y.value_counts(normalize=True).sort_index()
    print(f"  Home Win: {dist[0]:.1%}")
    print(f"  Draw:     {dist[1]:.1%}")
    print(f"  Away Win: {dist[2]:.1%}")
    
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        raise ValueError(f"Non-numeric columns found: {list(non_numeric)}")
    
    return X, y, feature_cols, df

def train_model_for_shap(X, y):
    """Train XGBoost model for SHAP analysis"""
    print("\n" + "="*80)
    print("TRAINING MODEL FOR SHAP ANALYSIS")
    print("="*80)
    
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    print("\nTraining model...")
    model.fit(X, y, verbose=False)
    
    accuracy = model.score(X, y)
    print(f" Training complete - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return model

def generate_shap_summary(model, X, feature_cols, class_names=['Home Win', 'Draw', 'Away Win']):
    """Generate SHAP summary plots for all classes"""
    print("\n" + "="*80)
    print("GENERATING SHAP VALUES")
    print("="*80)
    
    print("\nCreating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    
    if len(X) > 5000:
        print(f"Dataset is large ({len(X)} samples), using random sample of 2000 for SHAP...")
        sample_idx = np.random.choice(len(X), 2000, replace=False)
        X_sample = X.iloc[sample_idx]
    else:
        print(f"Computing SHAP values for all {len(X)} samples...")
        X_sample = X
    
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values_list = shap_values
    else:

        if len(shap_values.shape) == 3:
            shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        else:
            shap_values_list = [shap_values]
    
    print(" SHAP values computed")
    
    os.makedirs('models/shap_analysis', exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING SHAP VISUALIZATIONS")
    print("="*80)
    
    for class_idx, class_name in enumerate(class_names):
        print(f"\nGenerating plots for: {class_name}")
        
        try:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values_list[class_idx],
                X_sample,
                feature_names=feature_cols,
                show=False,
                max_display=20
            )
            plt.title(f'SHAP Summary - {class_name}', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f'models/shap_analysis/shap_summary_{class_name.lower().replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Summary plot saved")
        except Exception as e:
            print(f"  ⚠ Summary plot failed: {e}")
        
        try:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values_list[class_idx],
                X_sample,
                feature_names=feature_cols,
                plot_type="bar",
                show=False,
                max_display=20
            )
            plt.title(f'SHAP Feature Importance - {class_name}', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f'models/shap_analysis/shap_bar_{class_name.lower().replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Bar plot saved")
        except Exception as e:
            print(f"  ⚠ Bar plot failed: {e}")
    
    print(f"\nGenerating combined feature importance...")
    
    try:
        mean_abs_shap = np.zeros(X_sample.shape[1])
        for class_shap in shap_values_list:
            mean_abs_shap += np.abs(class_shap).mean(axis=0)
        mean_abs_shap /= len(shap_values_list)
        
        top_indices = np.argsort(mean_abs_shap)[-25:][::-1]
        
        plt.figure(figsize=(12, 10))
        plt.barh(range(len(top_indices)), mean_abs_shap[top_indices])
        plt.yticks(range(len(top_indices)), [feature_cols[i] for i in top_indices])
        plt.xlabel('Mean |SHAP value| (across all classes)', fontsize=12)
        plt.title('Top 25 Most Important Features (All Classes Combined)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('models/shap_analysis/shap_overall_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Overall importance plot saved")
    except Exception as e:
        print(f"Overall importance plot failed: {e}")
    
    return shap_values_list, X_sample

def print_feature_insights(shap_values, X_sample, feature_cols, top_n=10):
    """Print insights about top features"""
    print("\n" + "="*80)
    print("TOP FEATURE INSIGHTS")
    print("="*80)
    
    class_names = ['Home Win', 'Draw', 'Away Win']
    
    for class_idx, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print("-" * 80)
        
        try:
            mean_abs_shap = np.abs(shap_values[class_idx]).mean(axis=0)

            top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
            
            print(f"\nTop {top_n} features driving {class_name} predictions:\n")
            for rank, idx in enumerate(top_indices, 1):
                feature_name = feature_cols[idx]
                importance = mean_abs_shap[idx]
                
                mean_shap = shap_values[class_idx][:, idx].mean()
                direction = "increases" if mean_shap > 0 else "decreases"
                
                print(f"  {rank:2d}. {feature_name:45s} | Impact: {importance:.4f} | {direction} probability")
        except Exception as e:
            print(f"  Error computing insights: {e}")

def create_dependence_plots(shap_values, X_sample, feature_cols, top_features=5):
    print("\n" + "="*80)
    print("CREATING DEPENDENCE PLOTS")
    print("="*80)
    
    os.makedirs('models/shap_analysis/dependence', exist_ok=True)
    
    class_names = ['Home Win', 'Draw', 'Away Win']
    
    for class_idx, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        
        try:
            mean_abs_shap = np.abs(shap_values[class_idx]).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[-top_features:][::-1]
            
            for idx in top_indices:
                feature_name = feature_cols[idx]
                print(f"  Creating dependence plot for: {feature_name}")
                
                try:
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(
                        idx,
                        shap_values[class_idx],
                        X_sample,
                        feature_names=feature_cols,
                        show=False
                    )
                    plt.title(f'SHAP Dependence - {feature_name} ({class_name})', 
                             fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    
                    safe_feature_name = feature_name.replace('/', '_').replace('\\', '_')
                    safe_class_name = class_name.lower().replace(' ', '_')
                    plt.savefig(
                        f'models/shap_analysis/dependence/{safe_class_name}_{safe_feature_name}.png',
                        dpi=300, bbox_inches='tight'
                    )
                    plt.close()
                except Exception as e:
                    print(f"    Warning: Could not create dependence plot: {e}")
        except Exception as e:
            print(f"  Error processing class: {e}")
    
    print("\n✓ Dependence plots saved to models/shap_analysis/dependence/")

def main():
    """Main SHAP analysis pipeline"""
    print("\n" + "="*80)
    print(" "*25 + "SHAP ANALYSIS")
    print(" "*15 + "Understanding Model Predictions")
    print("="*80)
    
    if not os.path.exists('data/features.csv'):
        print("\n Error: data/features.csv not found!")
        print("   Please run feature_engineering.py first.")
        return
    
    X, y, feature_cols, df = load_data()
    
    model = train_model_for_shap(X, y)
    
    shap_values, X_sample = generate_shap_summary(model, X, feature_cols)
    
    print_feature_insights(shap_values, X_sample, feature_cols, top_n=10)
    
    create_dependence_plots(shap_values, X_sample, feature_cols, top_features=5)
    
    print("\n" + "="*80)
    print("SAVING ANALYSIS RESULTS")
    print("="*80)
    
    joblib.dump(model, 'models/shap_analysis/model.pkl')
    joblib.dump({
        'shap_values': shap_values,
        'X_sample': X_sample,
        'feature_cols': feature_cols
    }, 'models/shap_analysis/shap_data.pkl')
        
    print("\n" + "="*80)
    print("="*80)
    print("\nGenerated files in models/shap_analysis/:")
    print(" Summary plots (3): shap_summary_*.png")
    print(" Bar plots (3): shap_bar_*.png")
    print(" Overall importance: shap_overall_importance.png")
    print(" Dependence plots (15+): dependence/*.png")
    print(" Model: model.pkl")
    print(" SHAP data: shap_data.pkl")
    
    print("\n" + "="*80)
    print("HOW TO INTERPRET SHAP PLOTS")
    print("="*80)   
    print("="*80 + "\n")

if __name__ == "__main__":
    main()