import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os

# Load data
df = pd.read_csv("data/regular_season_totals_2010_2024.csv")
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# Use existing FG3_PCT field
df['3P%'] = df['FG3_PCT']

# Sort and create MORE rolling features with different windows
df = df.sort_values(by=["TEAM_NAME", "GAME_DATE"])

# Expanded rolling features
base_features = ['FG_PCT', '3P%', 'REB', 'AST', 'STL', 'TOV', 'PTS']
rolling_windows = [3, 5, 10]  # Multiple window sizes

for window in rolling_windows:
    for feature in base_features:
        df[f'{feature}_rolling_{window}'] = df.groupby("TEAM_NAME")[feature].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

# Add trend features (recent vs longer term)
for feature in base_features:
    if f'{feature}_rolling_3' in df.columns and f'{feature}_rolling_10' in df.columns:
        df[f'{feature}_trend'] = df[f'{feature}_rolling_3'] - df[f'{feature}_rolling_10']

# Add team strength features
df['win_numeric'] = (df['WL'] == 'W').astype(int)
df['team_win_pct'] = df.groupby('TEAM_NAME')['win_numeric'].transform(
    lambda x: x.shift(1).rolling(10, min_periods=1).mean()
)

# Add rest days feature
df['days_since_last'] = df.groupby('TEAM_NAME')['GAME_DATE'].diff().dt.days
df['days_since_last'] = df['days_since_last'].fillna(2)  # Default 2 days rest

# Add home/away recent performance
df['is_home'] = df['MATCHUP'].str.contains('vs').astype(int)
df['home_win_pct'] = df.groupby(['TEAM_NAME'])['win_numeric'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)

# Extract home and away stats
home_df = df[df['MATCHUP'].str.contains("vs")].copy()
away_df = df[df['MATCHUP'].str.contains("@")].copy()

home_df['GAME_ID'] = home_df['GAME_ID'].astype(str)
away_df['GAME_ID'] = away_df['GAME_ID'].astype(str)

# Expanded feature list
feature_columns = []
for window in rolling_windows:
    for feature in base_features:
        feature_columns.append(f'{feature}_rolling_{window}')

# Add trend features
for feature in base_features:
    if f'{feature}_trend' in df.columns:
        feature_columns.append(f'{feature}_trend')

# Add other features
feature_columns.extend(['team_win_pct', 'days_since_last', 'home_win_pct'])

print(f"Using {len(feature_columns)} features for training")

merged = pd.merge(
    home_df[['GAME_ID', 'TEAM_NAME', 'WL'] + feature_columns],
    away_df[['GAME_ID', 'TEAM_NAME'] + feature_columns],
    on='GAME_ID',
    suffixes=('_HOME', '_AWAY')
)

merged['WIN'] = merged['WL'].apply(lambda x: 1 if x == 'W' else 0)

# Create final feature list
features = [f + '_HOME' for f in feature_columns] + [f + '_AWAY' for f in feature_columns]
X = merged[features]
y = merged['WIN']

# Remove rows with NaN values
X = X.dropna()
y = y.loc[X.index]

print(f"Training data shape: {X.shape}")
print(f"Feature count: {len(features)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning
param_grid = {
    'xgb__n_estimators': [200, 300],
    'xgb__max_depth': [4, 6, 8],
    'xgb__learning_rate': [0.05, 0.1, 0.15],
    'xgb__subsample': [0.8, 0.9],
    'xgb__colsample_bytree': [0.8, 0.9, 1.0]
}

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
])

# Grid search with cross-validation
print("Performing hyperparameter tuning...")
grid_search = GridSearchCV(
    pipe, 
    param_grid, 
    cv=3, 
    scoring='neg_log_loss',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {-grid_search.best_score_:.4f}")

# Get best model
best_model = grid_search.best_estimator_

# Probability calibration to improve confidence
print("Applying probability calibration...")
calibrated_model = CalibratedClassifierCV(best_model, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)

# Evaluation on both models
y_pred_base = best_model.predict(X_test)
y_prob_base = best_model.predict_proba(X_test)[:, 1]

y_pred_cal = calibrated_model.predict(X_test)
y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

print("Base Model:")
print(f"Accuracy       : {accuracy_score(y_test, y_pred_base):.4f}")
print(f"Log Loss       : {log_loss(y_test, y_prob_base):.4f}")
print(f"ROC AUC Score  : {roc_auc_score(y_test, y_prob_base):.4f}")

print("\nCalibrated Model:")
print(f"Accuracy       : {accuracy_score(y_test, y_pred_cal):.4f}")
print(f"Log Loss       : {log_loss(y_test, y_prob_cal):.4f}")
print(f"ROC AUC Score  : {roc_auc_score(y_test, y_prob_cal):.4f}")

# Confidence distribution analysis
print("\nProbability Distribution Analysis:")
print("Base Model:")
print(f"  Mean probability: {y_prob_base.mean():.3f}")
print(f"  Std probability: {y_prob_base.std():.3f}")
print(f"  Min probability: {y_prob_base.min():.3f}")
print(f"  Max probability: {y_prob_base.max():.3f}")

print("Calibrated Model:")
print(f"  Mean probability: {y_prob_cal.mean():.3f}")
print(f"  Std probability: {y_prob_cal.std():.3f}")
print(f"  Min probability: {y_prob_cal.min():.3f}")
print(f"  Max probability: {y_prob_cal.max():.3f}")

# Choose better model based on log loss
if log_loss(y_test, y_prob_cal) < log_loss(y_test, y_prob_base):
    final_model = calibrated_model
    model_name = "calibrated_xgb_pipeline.pkl"
    print(f"\nUsing calibrated model (better log loss)")
else:
    final_model = best_model
    model_name = "xgb_pipeline.pkl"
    print(f"\nUsing base model (better log loss)")

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(final_model, f"model/{model_name}")
print(f"Model saved to model/{model_name}")

# Feature importance
if hasattr(final_model, 'feature_importances_'):
    feature_imp = final_model.feature_importances_
elif hasattr(final_model.base_estimator, 'feature_importances_'):
    feature_imp = final_model.base_estimator.feature_importances_
else:
    feature_imp = final_model.steps[-1][1].feature_importances_

# Show top 10 most important features
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': feature_imp
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))