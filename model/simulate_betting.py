import pandas as pd
import numpy as np
import argparse

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['logistic', 'xgb'], default='logistic',
                    help='Choose model for simulation (default: logistic)')
parser.add_argument('--strategy', choices=['high_confidence', 'value_betting', 'always_favorite'], 
                    default='high_confidence', help='Betting strategy')
parser.add_argument('--threshold', type=float, default=0.6, 
                    help='Confidence threshold for betting (default: 0.6)')
args = parser.parse_args()

# Load predictions
df = pd.read_csv("data/predicted_odds_2025.csv")

# Clean percentage strings if they exist
if df['Home Win %'].dtype == 'object':
    df['Home Win %'] = df['Home Win %'].str.rstrip('%').astype(float) / 100
if df['Away Win %'].dtype == 'object':
    df['Away Win %'] = df['Away Win %'].str.rstrip('%').astype(float) / 100

print(f"Loaded {len(df)} predictions")
print(f"Using strategy: {args.strategy}")
print(f"Confidence threshold: {args.threshold}")

# Simulation with different strategies
def simulate_betting_strategy(df, strategy, threshold=0.6, bet_amount=100):
    """
    Simulate betting with different strategies using Monte Carlo simulation
    Since we don't have actual game results, we'll simulate based on predicted probabilities
    """
    total_profit = 0
    total_bets = 0
    wins = 0
    losses = 0
    
    np.random.seed(42)  # For reproducible results
    
    for i, row in df.iterrows():
        home_prob = row['Home Win %']
        away_prob = row['Away Win %']
        home_odds = row['Home Odds']
        away_odds = row['Away Odds']
        
        bet_placed = False
        bet_on_home = False
        chosen_prob = 0
        chosen_odds = 0
        
        # Different betting strategies
        if strategy == 'high_confidence':
            # Only bet when confidence is above threshold
            if home_prob >= threshold:
                bet_placed = True
                bet_on_home = True
                chosen_prob = home_prob
                chosen_odds = home_odds
            elif away_prob >= threshold:
                bet_placed = True
                bet_on_home = False
                chosen_prob = away_prob
                chosen_odds = away_odds
                
        elif strategy == 'value_betting':
            # Look for value bets (when odds imply lower probability than our prediction)
            implied_home_prob = 1 / home_odds
            implied_away_prob = 1 / away_odds
            
            if home_prob > implied_home_prob + 0.05:  # 5% edge
                bet_placed = True
                bet_on_home = True
                chosen_prob = home_prob
                chosen_odds = home_odds
            elif away_prob > implied_away_prob + 0.05:
                bet_placed = True
                bet_on_home = False
                chosen_prob = away_prob
                chosen_odds = away_odds
                
        elif strategy == 'always_favorite':
            # Always bet on the favorite
            if home_prob > away_prob:
                bet_placed = True
                bet_on_home = True
                chosen_prob = home_prob
                chosen_odds = home_odds
            else:
                bet_placed = True
                bet_on_home = False
                chosen_prob = away_prob
                chosen_odds = away_odds
        
        if bet_placed:
            total_bets += 1
            
            # Simulate game outcome based on predicted probability
            # This is the key: use probability to simulate actual outcomes
            random_outcome = np.random.random()
            
            if bet_on_home:
                actual_home_wins = random_outcome < home_prob
                if actual_home_wins:
                    profit = bet_amount * (home_odds - 1)
                    wins += 1
                else:
                    profit = -bet_amount
                    losses += 1
            else:
                actual_away_wins = random_outcome < away_prob
                if actual_away_wins:
                    profit = bet_amount * (away_odds - 1)
                    wins += 1
                else:
                    profit = -bet_amount
                    losses += 1
            
            total_profit += profit
    
    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'total_profit': total_profit,
        'win_rate': wins / total_bets if total_bets > 0 else 0,
        'avg_profit_per_bet': total_profit / total_bets if total_bets > 0 else 0,
        'roi': (total_profit / (total_bets * bet_amount)) * 100 if total_bets > 0 else 0
    }

# Run simulation
results = simulate_betting_strategy(df, args.strategy, args.threshold)

# Display results
print("\n" + "="*50)
print("BETTING SIMULATION RESULTS")
print("="*50)
print(f"Strategy: {args.strategy}")
print(f"Total Predictions Available: {len(df)}")
print(f"Total Bets Placed: {results['total_bets']}")
print(f"Bets Won: {results['wins']}")
print(f"Bets Lost: {results['losses']}")
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Total Profit: ${results['total_profit']:.2f}")
print(f"Average Profit per Bet: ${results['avg_profit_per_bet']:.2f}")
print(f"Return on Investment (ROI): {results['roi']:.2f}%")

if results['total_profit'] > 0:
    print("✅ PROFITABLE STRATEGY!")
else:
    print("❌ LOSING STRATEGY")

# Additional analysis
print("\n" + "="*50)
print("STRATEGY COMPARISON")
print("="*50)

strategies = ['high_confidence', 'value_betting', 'always_favorite']
thresholds = [0.55, 0.60, 0.65, 0.70] if args.strategy == 'high_confidence' else [args.threshold]

best_roi = -100
best_config = None

for strategy in strategies:
    for threshold in thresholds:
        result = simulate_betting_strategy(df, strategy, threshold)
        if result['total_bets'] > 0:
            print(f"{strategy} (threshold={threshold:.2f}): "
                  f"ROI={result['roi']:.2f}%, "
                  f"Bets={result['total_bets']}, "
                  f"Win Rate={result['win_rate']:.2%}")
            
            if result['roi'] > best_roi:
                best_roi = result['roi']
                best_config = (strategy, threshold)

if best_config:
    print(f"\n🏆 BEST STRATEGY: {best_config[0]} with threshold {best_config[1]:.2f}")
    print(f"   Best ROI: {best_roi:.2f}%")