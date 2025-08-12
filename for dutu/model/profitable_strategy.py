import pandas as pd
import numpy as np
import argparse

def sigmoid_stretch(p, alpha=1.5):
    """
    Stretch probabilities using sigmoid function
    Alpha=1.5 gives the best results
    """
    epsilon = 1e-6
    p = np.clip(p, epsilon, 1 - epsilon)
    logits = np.log(p / (1 - p))
    stretched_logits = logits * alpha
    stretched_p = 1 / (1 + np.exp(-stretched_logits))
    return np.clip(stretched_p, epsilon, 1 - epsilon)

def create_profitable_predictions():
    """
    Create calibrated predictions file with profitable edges
    """
    df = pd.read_csv("data/predicted_odds_2025.csv")
    
    print("Creating profitable predictions with Sigmoid Stretch (Alpha=1.5)...")
    print(f"Original data: {len(df)} games")
    
    # Apply best calibration method
    cal_home_prob = sigmoid_stretch(df['Home Win %'], alpha=1.5)
    cal_away_prob = sigmoid_stretch(df['Away Win %'], alpha=1.5)
    
    # Normalize to ensure they sum to 1
    prob_sum = cal_home_prob + cal_away_prob
    cal_home_prob = cal_home_prob / prob_sum
    cal_away_prob = cal_away_prob / prob_sum
    
    # Calculate new odds with 5% margin (unchanged)
    margin = 0.05
    cal_home_odds = (1 / cal_home_prob) * (1 - margin)
    cal_away_odds = (1 / cal_away_prob) * (1 - margin)
    
    # Create calibrated predictions file
    calibrated_df = pd.DataFrame({
        'Home': df['Home'],
        'Home Win %': df['Home Win %'],  # Keep original probabilities for simulation
        'Home Odds': cal_home_odds,      # Use calibrated odds
        'Away': df['Away'],
        'Away Win %': df['Away Win %'],  # Keep original probabilities for simulation  
        'Away Odds': cal_away_odds       # Use calibrated odds
    })
    
    # Save calibrated predictions
    calibrated_df.to_csv('data/profitable_odds_2025.csv', index=False)
    
    # Calculate and display edge statistics
    cal_home_implied = 1.0 / cal_home_odds
    cal_away_implied = 1.0 / cal_away_odds
    cal_home_edge = df['Home Win %'] - cal_home_implied
    cal_away_edge = df['Away Win %'] - cal_away_implied
    cal_best_edge = np.maximum(cal_home_edge, cal_away_edge)
    
    positive_edges = (cal_best_edge > 0).sum()
    
    print("\n" + "="*60)
    print("PROFITABLE PREDICTIONS CREATED")
    print("="*60)
    print(f"File saved: data/profitable_odds_2025.csv")
    print(f"Total games: {len(calibrated_df)}")
    print(f"Games with positive edges: {positive_edges} ({positive_edges/len(df)*100:.1f}%)")
    print(f"Best edge: {cal_best_edge.max():.4f}")
    print(f"Mean positive edge: {cal_best_edge[cal_best_edge > 0].mean():.4f}")
    print(f"Average positive edge games per day: {positive_edges/365:.1f}")
    
    return calibrated_df

def simulate_profitable_betting(bet_amount=100, min_edge=0.01):
    """
    Simulate betting with the profitable calibrated odds
    """
    df = pd.read_csv("data/profitable_odds_2025.csv")
    
    print(f"\nSimulating betting with ${bet_amount} per bet, min edge {min_edge:.2%}")
    print("-" * 50)
    
    # Calculate edges
    home_implied = 1.0 / df['Home Odds']
    away_implied = 1.0 / df['Away Odds']
    home_edge = df['Home Win %'] - home_implied
    away_edge = df['Away Win %'] - away_implied
    
    total_profit = 0
    total_bets = 0
    wins = 0
    total_wagered = 0
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(len(df)):
        bet_placed = False
        chosen_prob = 0
        chosen_odds = 0
        edge = 0
        
        if home_edge.iloc[i] >= min_edge:
            bet_placed = True
            chosen_prob = df['Home Win %'].iloc[i]
            chosen_odds = df['Home Odds'].iloc[i]
            edge = home_edge.iloc[i]
        elif away_edge.iloc[i] >= min_edge:
            bet_placed = True
            chosen_prob = df['Away Win %'].iloc[i]
            chosen_odds = df['Away Odds'].iloc[i]
            edge = away_edge.iloc[i]
        
        if bet_placed:
            total_bets += 1
            total_wagered += bet_amount
            
            # Simulate game outcome using original probability
            win = np.random.random() < chosen_prob
            
            if win:
                profit = bet_amount * (chosen_odds - 1)
                total_profit += profit
                wins += 1
            else:
                total_profit -= bet_amount
    
    # Calculate statistics
    win_rate = wins / total_bets if total_bets > 0 else 0
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    avg_profit_per_bet = total_profit / total_bets if total_bets > 0 else 0
    
    print(f"Results:")
    print(f"  Total bets placed: {total_bets:,}")
    print(f"  Total wagered: ${total_wagered:,.0f}")
    print(f"  Wins: {wins:,}")
    print(f"  Losses: {total_bets - wins:,}")
    print(f"  Win rate: {win_rate:.1%}")
    print(f"  Total profit: ${total_profit:,.0f}")
    print(f"  ROI: {roi:+.2f}%")
    print(f"  Average profit per bet: ${avg_profit_per_bet:.2f}")
    print(f"  Bets per day: {total_bets/365:.1f}")
    print(f"  Expected daily profit: ${total_profit/365:.0f}")
    
    return {
        'total_bets': total_bets,
        'total_profit': total_profit,
        'roi': roi,
        'win_rate': win_rate
    }

def test_different_strategies():
    """
    Test different betting strategies with the profitable odds
    """
    print("\n" + "="*60)
    print("STRATEGY TESTING")
    print("="*60)
    
    strategies = [
        {'min_edge': 0.005, 'name': 'Very Aggressive (0.5% min edge)'},
        {'min_edge': 0.01, 'name': 'Aggressive (1% min edge)'},
        {'min_edge': 0.02, 'name': 'Balanced (2% min edge)'},
        {'min_edge': 0.03, 'name': 'Conservative (3% min edge)'},
        {'min_edge': 0.05, 'name': 'Very Conservative (5% min edge)'},
    ]
    
    best_strategy = None
    best_roi = 0
    
    for strategy in strategies:
        print(f"\n{strategy['name']}:")
        result = simulate_profitable_betting(bet_amount=100, min_edge=strategy['min_edge'])
        
        if result['roi'] > best_roi:
            best_roi = result['roi']
            best_strategy = strategy['name']
    
    print(f"\n🏆 Best strategy: {best_strategy} (ROI: {best_roi:+.2f}%)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['create', 'simulate', 'test'], default='create',
                        help='Action to perform: create profitable file, simulate betting, or test strategies')
    parser.add_argument('--bet_amount', type=float, default=100,
                        help='Bet amount for simulation (default: 100)')
    parser.add_argument('--min_edge', type=float, default=0.01,
                        help='Minimum edge for betting (default: 0.01)')
    args = parser.parse_args()
    
    if args.action == 'create':
        create_profitable_predictions()
        
    elif args.action == 'simulate':
        if not pd.io.common.file_exists('data/profitable_odds_2025.csv'):
            print("Creating profitable predictions first...")
            create_profitable_predictions()
        simulate_profitable_betting(args.bet_amount, args.min_edge)
        
    elif args.action == 'test':
        if not pd.io.common.file_exists('data/profitable_odds_2025.csv'):
            print("Creating profitable predictions first...")
            create_profitable_predictions()
        test_different_strategies()

if __name__ == "__main__":
    main()