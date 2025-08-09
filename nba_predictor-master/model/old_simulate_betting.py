import pandas as pd
import numpy as np

# Load predicted matchup data with win probabilities and odds
df = pd.read_csv("data/predicted_odds_2025.csv")

# Initialize tracking variables
total_profit = 0.0         # Total profit the platform makes
total_bets = 0             # Total number of individual bets
bet_amount = 1.00          # Each bettor wagers $1
np.random.seed(42)         # Seed for reproducible randomness

# Loop through each game and simulate outcomes
for _, row in df.iterrows():
    home_prob = row['HOME_WIN_PROB']   # Model-predicted home win probability
    away_prob = row['AWAY_WIN_PROB']   # Model-predicted away win probability
    home_odds = row['HOME_ODDS']       # Decimal odds for home win
    away_odds = row['AWAY_ODDS']       # Decimal odds for away win

    # Simulate the match outcome using win probability
    home_wins = np.random.rand() < home_prob  # True if home team wins

    # Assume two bettors: one bets on home, one on away, both $1
    platform_income = 2 * bet_amount  # The platform collects $2 total

    # Determine payout based on actual result
    if home_wins:
        platform_payout = bet_amount * home_odds  # Pays out to home bettor
    else:
        platform_payout = bet_amount * away_odds  # Pays out to away bettor

    # Platform profit = income - payout
    profit = platform_income - platform_payout
    total_profit += profit
    total_bets += 2  # Two individual $1 bets per game

# Compute average profit per bet
avg_profit_per_bet = total_profit / total_bets

# Report results
print("📊 Betting Simulation Summary")
print(f"Total Bets Made     : {total_bets}")
print(f"Total Platform Profit: ${total_profit:.2f}")
print(f"Avg Profit per Bet  : ${avg_profit_per_bet:.4f}")
