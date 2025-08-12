import pandas as pd
import numpy as np
import joblib
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['logistic', 'xgb', 'calibrated_xgb'], default='xgb',
                    help='Choose model: logistic, xgb, or calibrated_xgb (default: xgb)')
args = parser.parse_args()
model_type = args.model

# Load trained model - try different paths
model_paths = [
    f'model/{model_type}_pipeline.pkl',
    f'model/calibrated_xgb_pipeline.pkl' if model_type == 'calibrated_xgb' else None
]

model_path = None
for path in model_paths:
    if path and os.path.exists(path):
        model_path = path
        break

if not model_path:
    print(f"Model files not found. Available models:")
    if os.path.exists('model'):
        for f in os.listdir('model'):
            if f.endswith('.pkl'):
                print(f"  - {f}")
    # Default to original xgb model
    model_path = 'model/xgb_pipeline.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model files found in model/ directory")

print(f"Loading model from: {model_path}")
model = joblib.load(model_path)

# Load raw data
df = pd.read_csv('data/regular_season_totals_2010_2024.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# Use existing FG3_PCT field to match training data
df['3P%'] = df['FG3_PCT']

# Sort by team and date to match training data processing
df = df.sort_values(by=['TEAM_NAME', 'GAME_DATE'])

# Create ORIGINAL rolling features to match the existing model
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

# Merge home and away data to create games - USING ORIGINAL FEATURES
games = pd.merge(
    home_df[['GAME_ID', 'TEAM_NAME', 'WL'] + [f'{f}_rolling' for f in rolling_features]],
    away_df[['GAME_ID', 'TEAM_NAME'] + [f'{f}_rolling' for f in rolling_features]],
    on='GAME_ID',
    suffixes=('_HOME', '_AWAY')
)

# Create target variable (for reference, not used in prediction)
games['WIN'] = games['WL'].apply(lambda x: 1 if x == 'W' else 0)

# Define features to match ORIGINAL training data exactly
features = [f + '_rolling_HOME' for f in rolling_features] + [f + '_rolling_AWAY' for f in rolling_features]

print(f"Using original features: {features}")

# Check if all required features exist
missing_features = [f for f in features if f not in games.columns]
if missing_features:
    print(f"Missing features: {missing_features}")
    print(f"Available columns: {list(games.columns)}")
    raise ValueError(f"Missing features required by model: {missing_features}")

# Prepare feature matrix
X_future = games[features]

# Remove rows with NaN values
X_future = X_future.dropna()
games_clean = games.loc[X_future.index]

print(f"Predicting on {len(X_future)} games with {len(features)} features")

# Predict probabilities
try:
    home_win_prob = model.predict_proba(X_future)[:, 1]
    games_clean['HOME_WIN_PROB'] = home_win_prob
    games_clean['AWAY_WIN_PROB'] = 1 - home_win_prob

    # IMPORTANT: Keep the original odds calculation with 5% margin
    margin = 0.05
    games_clean['HOME_ODDS'] = (1 / games_clean['HOME_WIN_PROB']) * (1 - margin)
    games_clean['AWAY_ODDS'] = (1 / games_clean['AWAY_WIN_PROB']) * (1 - margin)
    
    # Add confidence metrics for analysis
    games_clean['MAX_PROB'] = np.maximum(games_clean['HOME_WIN_PROB'], games_clean['AWAY_WIN_PROB'])
    games_clean['PROB_DIFF'] = np.abs(games_clean['HOME_WIN_PROB'] - games_clean['AWAY_WIN_PROB'])
    
    print(f"Prediction Statistics:")
    print(f"Mean home win probability: {home_win_prob.mean():.3f}")
    print(f"Std home win probability: {home_win_prob.std():.3f}")
    print(f"Min probability: {home_win_prob.min():.3f}")
    print(f"Max probability: {home_win_prob.max():.3f}")
    print(f"Max confidence: {games_clean['MAX_PROB'].max():.3f}")
    print(f"Games with >70% confidence: {(games_clean['MAX_PROB'] > 0.7).sum()}")
    print(f"Games with >80% confidence: {(games_clean['MAX_PROB'] > 0.8).sum()}")
    print(f"Games with >90% confidence: {(games_clean['MAX_PROB'] > 0.9).sum()}")
    
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

# Quick analysis of edges
print("\n" + "="*50)
print("EDGE ANALYSIS")
print("="*50)

# Calculate edges
output['home_implied'] = 1.0 / output['Home Odds']
output['away_implied'] = 1.0 / output['Away Odds']
output['home_edge'] = output['Home Win %'] - output['home_implied']
output['away_edge'] = output['Away Win %'] - output['away_implied']
output['best_edge'] = np.maximum(output['home_edge'], output['away_edge'])

print(f"Edge statistics:")
print(f"Best edge range: {output['best_edge'].min():.6f} to {output['best_edge'].max():.6f}")
print(f"Mean best edge: {output['best_edge'].mean():.6f}")
print(f"Positive edges: {(output['best_edge'] > 0).sum()} out of {len(output)}")

if (output['best_edge'] > 0).sum() > 0:
    print("🎉 Found some positive edges!")
    positive_edges = output[output['best_edge'] > 0]
    print(f"Best positive edge: {positive_edges['best_edge'].max():.6f}")
    print(f"Average positive edge: {positive_edges['best_edge'].mean():.6f}")
else:
    print("❌ No positive edges found")
    print("Need to improve model or try different strategies")