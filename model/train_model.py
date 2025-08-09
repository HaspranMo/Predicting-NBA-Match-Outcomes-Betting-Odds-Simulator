import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import joblib

# Load dataset
df = pd.read_csv('data/regular_season_totals_2010_2024.csv')

# Convert GAME_DATE to datetime
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# Only keep essential columns
df = df[['TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'AST', 'REB']]

# Sort data for rolling mean
df = df.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE'])

# Compute rolling averages of last 5 games for each team
for stat in ['PTS', 'AST', 'REB']:
    df[f'{stat}_AVG'] = df.groupby('TEAM_ABBREVIATION')[stat].transform(lambda x: x.shift().rolling(5).mean())

# Drop games without rolling stats (first 5 games for each team)
df = df.dropna(subset=['PTS_AVG', 'AST_AVG', 'REB_AVG'])

# Identify home/away based on "vs." in MATCHUP
df['IS_HOME'] = df['MATCHUP'].str.contains('vs.')

# Split into home/away rows
home_df = df[df['IS_HOME']].copy()
away_df = df[~df['IS_HOME']].copy()

# Rename for merge
home_df = home_df.rename(columns={
    'TEAM_ABBREVIATION': 'HOME_TEAM',
    'PTS_AVG': 'HOME_PTS_AVG',
    'AST_AVG': 'HOME_AST_AVG',
    'REB_AVG': 'HOME_REB_AVG',
    'WL': 'HOME_WL'
})

away_df = away_df.rename(columns={
    'TEAM_ABBREVIATION': 'AWAY_TEAM',
    'PTS_AVG': 'AWAY_PTS_AVG',
    'AST_AVG': 'AWAY_AST_AVG',
    'REB_AVG': 'AWAY_REB_AVG',
    'WL': 'AWAY_WL'
})

# Merge home and away stats on GAME_ID
games = pd.merge(
    home_df[['GAME_ID', 'GAME_DATE', 'HOME_TEAM', 'HOME_PTS_AVG', 'HOME_AST_AVG', 'HOME_REB_AVG', 'HOME_WL']],
    away_df[['GAME_ID', 'AWAY_TEAM', 'AWAY_PTS_AVG', 'AWAY_AST_AVG', 'AWAY_REB_AVG']],
    on='GAME_ID'
)

# Compute feature differences
games['PTS_DIFF'] = games['HOME_PTS_AVG'] - games['AWAY_PTS_AVG']
games['AST_DIFF'] = games['HOME_AST_AVG'] - games['AWAY_AST_AVG']
games['REB_DIFF'] = games['HOME_REB_AVG'] - games['AWAY_REB_AVG']

# Target: 1 if home team wins, else 0
games['HOME_WIN'] = games['HOME_WL'].apply(lambda x: 1 if x == 'W' else 0)

# Prepare data for training
X = games[['PTS_DIFF', 'AST_DIFF', 'REB_DIFF']]
y = games['HOME_WIN']

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Build pipeline with standard scaler and logistic regression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Train model
pipe.fit(X_train, y_train)

# Predict on test set
y_probs = pipe.predict_proba(X_test)[:, 1]
y_preds = (y_probs > 0.5).astype(int)

# Evaluate model
print("📊 Evaluation on Test Set:")
print(f"Accuracy     : {accuracy_score(y_test, y_preds):.4f}")
print(f"Log Loss     : {log_loss(y_test, y_probs):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")

# Save model
joblib.dump(pipe, 'model/logistic_pipeline.pkl')
print("✅ Model saved to model/logistic_pipeline.pkl")
