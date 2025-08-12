import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import joblib
import os

print("🏦 Training House Edge Prediction Model")
print("="*50)

# Load data
df = pd.read_csv("data/regular_season_totals_2010_2024.csv")

# Use existing FG3_PCT field
df['3P%'] = df['FG3_PCT']

# Sort and create rolling features
df = df.sort_values(by=["TEAM_NAME", "GAME_DATE"])
rolling_features = ['FG_PCT', '3P%', 'REB', 'AST', 'STL', 'TOV']
for feature in rolling_features:
    df[f'{feature}_rolling'] = df.groupby("TEAM_NAME")[feature].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

# Extract home and away stats
home_df = df[df['MATCHUP'].str.contains("vs")].copy()
away_df = df[df['MATCHUP'].str.contains("@")].copy()

home_df['GAME_ID'] = home_df['GAME_ID'].astype(str)
away_df['GAME_ID'] = away_df['GAME_ID'].astype(str)

merged = pd.merge(
    home_df[['GAME_ID', 'TEAM_NAME', 'WL'] + [f"{f}_rolling" for f in rolling_features]],
    away_df[['GAME_ID', 'TEAM_NAME'] + [f"{f}_rolling" for f in rolling_features]],
    on='GAME_ID',
    suffixes=('_HOME', '_AWAY')
)

merged['WIN'] = merged['WL'].apply(lambda x: 1 if x == 'W' else 0)

features = [f + '_rolling_HOME' for f in rolling_features] + [f + '_rolling_AWAY' for f in rolling_features]
X = merged[features]
y = merged['WIN']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=4, eval_metric='logloss'))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

# Evaluation
print("🎯 Model Performance for House Edge Calculation:")
print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
print(f"Log Loss       : {log_loss(y_test, y_prob):.4f}")
print(f"ROC AUC Score  : {roc_auc_score(y_test, y_prob):.4f}")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(pipe, "model/xgb_pipeline.pkl")
print("\n✅ House edge prediction model saved to model/xgb_pipeline.pkl")
print("📊 Model ready for odds calculation to ensure house profitability")