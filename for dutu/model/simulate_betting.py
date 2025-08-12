import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['logistic', 'xgb', 'calibrated_xgb'], default='calibrated_xgb',
                    help='Choose model for simulation (default: calibrated_xgb)')
parser.add_argument('--strategy', choices=['high_confidence', 'value_betting', 'always_favorite', 'smart_selective'],
                    default='smart_selective', help='Betting strategy')
parser.add_argument('--threshold', type=float, default=0.75,
                    help='Confidence threshold for betting (default: 0.75)')
parser.add_argument('--edge', type=float, default=0.01,
                    help='Minimum edge for value betting (default: 0.01)')
parser.add_argument('--bet_amount', type=float, default=100.0,
                    help='Bet size for each placed wager (default: 100)')
args = parser.parse_args()

# Load predictions
df = pd.read_csv("data/predicted_odds_2025.csv")

# Clean percentage strings if necessary
if df['Home Win %'].dtype == 'object':
    df['Home Win %'] = df['Home Win %'].str.rstrip('%').astype(float) / 100
if df['Away Win %'].dtype == 'object':
    df['Away Win %'] = df['Away Win %'].str.rstrip('%').astype(float) / 100

print(f"Loaded {len(df)} predictions")

def smart_selective_strategy(df, min_confidence=0.75, min_edge=-0.02, bet_amount=100.0, rng_seed=42):
    """
    Smart strategy that looks for:
    1. High confidence predictions (>75%)
    2. Acceptable negative edge (not worse than -2%)
    3. Situations where model might have an information advantage
    """
    total_profit = 0.0
    total_bets = 0
    wins = 0
    losses = 0
    
    np.random.seed(rng_seed)
    
    for _, row in df.iterrows():
        home_prob = float(row['Home Win %'])
        away_prob = float(row['Away Win %'])
        home_odds = float(row['Home Odds'])
        away_odds = float(row['Away Odds'])
        
        # Calculate implied probabilities and edges
        implied_home = 1.0 / home_odds
        implied_away = 1.0 / away_odds
        home_edge = home_prob - implied_home
        away_edge = away_prob - implied_away
        
        bet_placed = False
        bet_on_home = False
        chosen_prob = 0.0
        chosen_odds = 0.0
        
        # Smart selection criteria:
        # 1. High confidence + acceptable edge
        # 2. Focus on extreme predictions where model might know something
        
        if (home_prob >= min_confidence and home_edge >= min_edge):
            bet_placed = True
            bet_on_home = True
            chosen_prob = home_prob
            chosen_odds = home_odds
        elif (away_prob >= min_confidence and away_edge >= min_edge):
            bet_placed = True
            bet_on_home = False
            chosen_prob = away_prob
            chosen_odds = away_odds
        
        if bet_placed:
            total_bets += 1
            
            # Simulate game outcome
            r = np.random.random()
            win = (r < chosen_prob)
            
            if win:
                total_profit += bet_amount * (chosen_odds - 1.0)
                wins += 1
            else:
                total_profit -= bet_amount
                losses += 1
    
    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'total_profit': total_profit,
        'win_rate': (wins / total_bets) if total_bets else 0.0,
        'avg_profit_per_bet': (total_profit / total_bets) if total_bets else 0.0,
        'roi': ((total_profit / (total_bets * bet_amount)) * 100.0) if total_bets else 0.0
    }

def extreme_confidence_strategy(df, min_confidence=0.80, max_edge=-0.01, bet_amount=100.0, rng_seed=42):
    """
    Only bet when model is extremely confident AND edge isn't too bad
    """
    total_profit = 0.0
    total_bets = 0
    wins = 0
    losses = 0
    
    np.random.seed(rng_seed)
    
    for _, row in df.iterrows():
        home_prob = float(row['Home Win %'])
        away_prob = float(row['Away Win %'])
        home_odds = float(row['Home Odds'])
        away_odds = float(row['Away Odds'])
        
        # Calculate edges
        implied_home = 1.0 / home_odds
        implied_away = 1.0 / away_odds
        home_edge = home_prob - implied_home
        away_edge = away_prob - implied_away
        
        bet_placed = False
        chosen_prob = 0.0
        chosen_odds = 0.0
        
        # Only bet on extremely confident predictions with reasonable edge
        if (home_prob >= min_confidence and home_edge >= max_edge):
            bet_placed = True
            chosen_prob = home_prob
            chosen_odds = home_odds
        elif (away_prob >= min_confidence and away_edge >= max_edge):
            bet_placed = True
            chosen_prob = away_prob
            chosen_odds = away_odds
        
        if bet_placed:
            total_bets += 1
            
            r = np.random.random()
            win = (r < chosen_prob)
            
            if win:
                total_profit += bet_amount * (chosen_odds - 1.0)
                wins += 1
            else:
                total_profit -= bet_amount
                losses += 1
    
    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'total_profit': total_profit,
        'win_rate': (wins / total_bets) if total_bets else 0.0,
        'avg_profit_per_bet': (total_profit / total_bets) if total_bets else 0.0,
        'roi': ((total_profit / (total_bets * bet_amount)) * 100.0) if total_bets else 0.0
    }

def analyze_betting_landscape(df):
    """
    Analyze the betting opportunities in the data
    """
    print("\nBetting Landscape Analysis:")
    print("="*50)
    
    # Calculate all edges
    df['home_implied'] = 1.0 / df['Home Odds']
    df['away_implied'] = 1.0 / df['Away Odds']
    df['home_edge'] = df['Home Win %'] - df['home_implied']
    df['away_edge'] = df['Away Win %'] - df['away_implied']
    df['max_prob'] = np.maximum(df['Home Win %'], df['Away Win %'])
    df['best_edge'] = np.maximum(df['home_edge'], df['away_edge'])
    
    print(f"Total games: {len(df)}")
    print(f"Edge range: {df['best_edge'].min():.4f} to {df['best_edge'].max():.4f}")
    print(f"Mean edge: {df['best_edge'].mean():.4f}")
    print()
    
    # Confidence analysis
    confidence_levels = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    edge_thresholds = [-0.05, -0.03, -0.02, -0.01, 0.00]
    
    print("Opportunities by confidence level:")
    for conf in confidence_levels:
        count = (df['max_prob'] >= conf).sum()
        print(f"  Confidence >= {conf:.0%}: {count} games ({count/len(df)*100:.1f}%)")
    
    print("\nOpportunities by edge threshold:")
    for edge in edge_thresholds:
        count = (df['best_edge'] >= edge).sum()
        print(f"  Edge >= {edge:.2f}: {count} games ({count/len(df)*100:.1f}%)")
    
    print("\nCombined opportunities (Confidence >= 75% AND Edge >= threshold):")
    for edge in edge_thresholds:
        count = ((df['max_prob'] >= 0.75) & (df['best_edge'] >= edge)).sum()
        avg_edge = df[(df['max_prob'] >= 0.75) & (df['best_edge'] >= edge)]['best_edge'].mean()
        print(f"  Edge >= {edge:.2f}: {count} games, Avg Edge: {avg_edge:.4f}")

# Run analysis first
analyze_betting_landscape(df)

print("\n" + "="*60)
print("STRATEGY TESTING")
print("="*60)

# Test smart selective strategy with different parameters
print("\nSmart Selective Strategy (High Confidence + Acceptable Edge):")
print("-" * 50)

configs = [
    {'min_confidence': 0.70, 'min_edge': -0.03, 'name': 'Conservative (70%, -3%)'},
    {'min_confidence': 0.75, 'min_edge': -0.02, 'name': 'Balanced (75%, -2%)'},
    {'min_confidence': 0.80, 'min_edge': -0.01, 'name': 'Aggressive (80%, -1%)'},
    {'min_confidence': 0.85, 'min_edge': -0.01, 'name': 'Ultra Selective (85%, -1%)'},
]

best_roi = -100
best_config = None

for config in configs:
    result = smart_selective_strategy(
        df,
        min_confidence=config['min_confidence'],
        min_edge=config['min_edge'],
        bet_amount=args.bet_amount
    )
    
    print(f"{config['name']}:")
    print(f"  ROI: {result['roi']:+.2f}%")
    print(f"  Bets: {result['total_bets']}")
    print(f"  Win Rate: {result['win_rate']:.1%}")
    print(f"  Total Profit: ${result['total_profit']:+.0f}")
    
    if result['roi'] > best_roi and result['total_bets'] > 0:
        best_roi = result['roi']
        best_config = config
    print()

# Test extreme confidence strategy
print("Extreme Confidence Strategy:")
print("-" * 50)

extreme_configs = [
    {'min_confidence': 0.85, 'max_edge': -0.005},
    {'min_confidence': 0.90, 'max_edge': -0.01},
    {'min_confidence': 0.95, 'max_edge': -0.02},
]

for config in extreme_configs:
    result = extreme_confidence_strategy(
        df,
        min_confidence=config['min_confidence'],
        max_edge=config['max_edge'],
        bet_amount=args.bet_amount
    )
    
    print(f"Confidence >= {config['min_confidence']:.0%}, Edge >= {config['max_edge']:.3f}:")
    print(f"  ROI: {result['roi']:+.2f}%")
    print(f"  Bets: {result['total_bets']}")
    print(f"  Win Rate: {result['win_rate']:.1%}")
    print()

# Summary
if best_roi > 0:
    print("="*60)
    print(f"🎉 PROFITABLE STRATEGY FOUND!")
    print(f"Best Strategy: {best_config['name']}")
    print(f"ROI: +{best_roi:.2f}%")
    print("="*60)
else:
    print("="*60)
    print("⚠️  No profitable strategies found")
    print("Next steps:")
    print("1. Retrain model with improved features")
    print("2. Try probability calibration")
    print("3. Look for specific team/situation patterns")
    print("="*60)