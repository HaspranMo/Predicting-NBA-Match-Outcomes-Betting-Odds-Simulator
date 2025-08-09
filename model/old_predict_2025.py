import pandas as pd
import numpy as np
import joblib
import random

# Load model
model = joblib.load('model/logistic_pipeline.pkl')

# Load historical data
df = pd.read_csv('data/regular_season_totals_2010_2024.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df[['TEAM_ABBREVIATION', 'GAME_DATE', 'MATCHUP', 'PTS', 'AST', 'REB']]
df = df.sort_values(by=['TEAM_ABBREVIATION', 'GAME_DATE'])

# Keep only 2023-24 season games
df = df[df['GAME_DATE'].dt.year >= 2023]

# Calculate rolling stats (last 5 games)
for stat in ['PTS', 'AST', 'REB']:
    df[f'{stat}_AVG'] = df.groupby('TEAM_ABBREVIATION')[stat].transform(lambda x: x.shift().rolling(5).mean())

# Keep only rows with complete rolling averages
df = df.dropna(subset=['PTS_AVG', 'AST_AVG', 'REB_AVG'])

# Get last rolling stats per team
latest_stats = df.groupby('TEAM_ABBREVIATION').tail(1)
teams = latest_stats['TEAM_ABBREVIATION'].unique().tolist()

# Generate synthetic 2025 matchups: each team plays 5 random others
games = []
for home_team in teams:
    possible_opponents = [t for t in teams if t != home_team]
    opponents = random.sample(possible_opponents, 5)

    for away_team in opponents:
        home_row = latest_stats[latest_stats['TEAM_ABBREVIATION'] == home_team].iloc[0]
        away_row = latest_stats[latest_stats['TEAM_ABBREVIATION'] == away_team].iloc[0]

        game = {
            'HOME_TEAM': home_team,
            'AWAY_TEAM': away_team,
            'PTS_DIFF': home_row['PTS_AVG'] - away_row['PTS_AVG'],
            'AST_DIFF': home_row['AST_AVG'] - away_row['AST_AVG'],
            'REB_DIFF': home_row['REB_AVG'] - away_row['REB_AVG'],
        }
        games.append(game)

# Convert to DataFrame
predict_df = pd.DataFrame(games)

# Predict win probabilities
X_pred = predict_df[['PTS_DIFF', 'AST_DIFF', 'REB_DIFF']]
predict_df['HOME_WIN_PROB'] = model.predict_proba(X_pred)[:, 1]
predict_df['AWAY_WIN_PROB'] = 1 - predict_df['HOME_WIN_PROB']

# Convert to European odds with margin
def prob_to_odds(p, margin=0.05):
    """
    Given win probability `p`, return adjusted odds that give the house a fixed margin.
    """
    p = min(max(p, 0.01), 0.99)  # avoid extreme values
    fair_odds = 1 / p
    adjusted_odds = fair_odds * (1 - margin)  # house takes margin
    return round(adjusted_odds, 2)


predict_df['HOME_ODDS'] = predict_df['HOME_WIN_PROB'].apply(lambda x: prob_to_odds(x))
predict_df['AWAY_ODDS'] = predict_df['AWAY_WIN_PROB'].apply(lambda x: prob_to_odds(x))

# Save to CSV
predict_df.to_csv('data/predicted_odds_2025.csv', index=False)
print("✅ Predictions saved to data/predicted_odds_2025.csv")
