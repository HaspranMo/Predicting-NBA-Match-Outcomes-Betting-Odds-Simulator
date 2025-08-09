import pandas as pd
import numpy as np
import joblib
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['logistic', 'xgb'], default='logistic',
                    help='Choose model: logistic or xgb (default: logistic)')
args = parser.parse_args()
model_type = args.model

# Load trained model
model_path = f'model/{model_type}_pipeline.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)

# Load raw data
df = pd.read_csv('data/regular_season_totals_2010_2024.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# Use existing FG3_PCT field to match training data
df['3P%'] = df['FG3_PCT']

# Sort by team and date to match training data processing
df = df.sort_values(by=['TEAM_NAME', 'GAME_DATE'])

# Create rolling features to match training data
rolling_features = ['FG_PCT', '3P%', 'REB', 'AST', 'STL', 'TOV']
for feature in rolling_features:
    df[f'{feature}_rolling'] = df.groupby('TEAM_NAME')[feature].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )

# Extract home and away games
home_df = df[df['MATCHUP'].str.contains('vs')].copy()
away_df = df[df['MATCHUP'].str.contains('@')].copy()

# Ensure GAME_ID is string type
home_df['GAME_ID'] = home_df['GAME_ID'].astype(str)
away_df['GAME_ID'] = away_df['GAME_ID'].astype(str)

# Merge home and away data to create games
games = pd.merge(
    home_df[['GAME_ID', 'TEAM_NAME', 'WL'] + [f'{f}_rolling' for f in rolling_features]],
    away_df[['GAME_ID', 'TEAM_NAME'] + [f'{f}_rolling' for f in rolling_features]],
    on='GAME_ID',
    suffixes=('_HOME', '_AWAY')
)

# Create target variable (for reference, not used in prediction)
games['WIN'] = games['WL'].apply(lambda x: 1 if x == 'W' else 0)

# Define features to match training data exactly
features = [f + '_rolling_HOME' for f in rolling_features] + [f + '_rolling_AWAY' for f in rolling_features]

# Check if all required features exist
missing_features = [f for f in features if f not in games.columns]
if missing_features:
    raise ValueError(f"Missing features required by model: {missing_features}")

# Prepare feature matrix
X_future = games[features]

# Remove rows with NaN values
X_future = X_future.dropna()
games_clean = games.loc[X_future.index]

# Predict probabilities
try:
    home_win_prob = model.predict_proba(X_future)[:, 1]
    games_clean['HOME_WIN_PROB'] = home_win_prob
    games_clean['AWAY_WIN_PROB'] = 1 - home_win_prob
    
    # Convert to betting odds
    games_clean['HOME_ODDS'] = 1 / games_clean['HOME_WIN_PROB']
    games_clean['AWAY_ODDS'] = 1 / games_clean['AWAY_WIN_PROB']
    
except Exception as e:
    print(f"Error during prediction: {e}")
    print(f"Expected features: {features}")
    print(f"Available features: {X_future.columns.tolist()}")
    print(f"X_future shape: {X_future.shape}")
    raise

# Prepare output
output = games_clean[['TEAM_NAME_HOME', 'HOME_WIN_PROB', 'HOME_ODDS', 
                     'TEAM_NAME_AWAY', 'AWAY_WIN_PROB', 'AWAY_ODDS']]
output.columns = ['Home', 'Home Win %', 'Home Odds', 'Away', 'Away Win %', 'Away Odds']

# Save output
output.to_csv('data/predicted_odds_2025.csv', index=False)
print("Predictions saved to data/predicted_odds_2025.csv")
print(f"Total predictions: {len(output)}")
print(f"Model type used: {model_type}")