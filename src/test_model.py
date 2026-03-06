import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/features.csv")

rf = joblib.load("models/tuned/random_forest_tuned.pkl")
xgb = joblib.load("models/tuned/xgboost_tuned.pkl")
fc = joblib.load("models/tuned/feature_columns.pkl")

X = df[fc]
y = df["FTR"]

pred_rf = rf.predict(X)
pred_xgb = xgb.predict(X)

print("RF accuracy:", accuracy_score(y, pred_rf))
print("XGB accuracy:", accuracy_score(y, pred_xgb))