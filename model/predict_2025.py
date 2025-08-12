import pandas as pd
import numpy as np
import joblib
import argparse
import os

def prob_to_odds(p, margin=0.05):
    """
    Calculate house odds ensuring 5% margin
    Given win probability 'p', return adjusted odds that give the house a fixed margin
    """
    p = min(max(p, 0.01), 0.99)  # avoid extreme values
    fair_odds = 1 / p
    adjusted_odds = fair_odds * (1 - margin)  # house takes margin
    return round(adjusted_odds, 2)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['logistic', 'xgb'], default='xgb',
                    help='Choose model: logistic or xgb (default: xgb)')
args = parser.parse_args()
model_type = args.model

print(f"🏦 Generating House Profitable Odds for 2025 Season")
print(f"📊 Using {model_type.upper()} model with 5% house margin")
print("="*60)

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

print(f"📈 Predicting outcomes for {len(X_future)} games")

# Predict probabilities
try:
    home_win_prob = model.predict_proba(X_future)[:, 1]
    games_clean['HOME_WIN_PROB'] = home_win_prob
    games_clean['AWAY_WIN_PROB'] = 1 - home_win_prob

    # CRITICAL: Calculate house-favorable odds with 5% margin
    margin = 0.05  # House margin - DO NOT CHANGE
    games_clean['HOME_ODDS'] = games_clean['HOME_WIN_PROB'].apply(lambda p: prob_to_odds(p, margin))
    games_clean['AWAY_ODDS'] = games_clean['AWAY_WIN_PROB'].apply(lambda p: prob_to_odds(p, margin))
    
    print(f"💰 House Margin Applied: {margin:.1%}")
    print(f"🎯 Odds calculated to ensure house profitability")
    
except Exception as e:
    print(f"❌ Error during prediction: {e}")
    raise

# Prepare output
output = games_clean[['TEAM_NAME_HOME', 'HOME_WIN_PROB', 'HOME_ODDS', 
                     'TEAM_NAME_AWAY', 'AWAY_WIN_PROB', 'AWAY_ODDS']]
output.columns = ['Home', 'Home Win %', 'Home Odds', 'Away', 'Away Win %', 'Away Odds']

# Save output
output.to_csv('data/house_odds_2025.csv', index=False)
print(f"\n✅ House-profitable odds saved to data/house_odds_2025.csv")
print(f"📊 Total games for betting: {len(output)}")
print(f"🏦 Model type used: {model_type}")

# Quick house profit analysis
print(f"\n💡 HOUSE PROFIT PREVIEW:")
sample_analysis = []
for _, row in output.head(5).iterrows():
    home_prob = row['Home Win %']
    away_prob = row['Away Win %']
    home_odds = row['Home Odds']
    away_odds = row['Away Odds']
    
    # Calculate expected house profit per $1 bet (50/50 split assumption)
    home_house_profit = (1 - home_prob) * 1 - home_prob * (home_odds - 1)
    away_house_profit = (1 - away_prob) * 1 - away_prob * (away_odds - 1)
    avg_profit = (home_house_profit + away_house_profit) / 2
    
    print(f"   {row['Home']} vs {row['Away']}: ${avg_profit:.4f} profit per $1 bet")

print(f"\n🎰 Run house profit analysis: python house_profit_simulator.py")