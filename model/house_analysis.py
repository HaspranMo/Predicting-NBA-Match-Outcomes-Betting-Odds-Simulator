import pandas as pd
import numpy as np
import argparse

def analyze_house_profitability(data_file='data/house_odds_2025.csv'):
    """
    分析赌场在当前赔率设置下的盈利能力
    """
    print("🏦 HOUSE PROFITABILITY ANALYSIS")
    print("=" * 60)
    
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Loaded {len(df)} games from {data_file}")
    except FileNotFoundError:
        print(f"❌ File not found: {data_file}")
        print("Please run: python predict_2025.py --model xgb")
        return None, None
    
    total_house_profit = 0
    profitable_games = 0
    house_profits = []
    
    print(f"\nAnalyzing {len(df)} games...")
    
    for _, row in df.iterrows():
        home_prob = row['Home Win %']
        away_prob = row['Away Win %']
        home_odds = row['Home Odds']
        away_odds = row['Away Odds']
        
        # Calculate house expected profit per $1 bet
        # If player bets $1 on home team:
        home_house_profit = (1 - home_prob) * 1 - home_prob * (home_odds - 1)
        # If player bets $1 on away team:
        away_house_profit = (1 - away_prob) * 1 - away_prob * (away_odds - 1)
        
        # Average profit (assuming balanced betting)
        game_profit = (home_house_profit + away_house_profit) / 2
        house_profits.append(game_profit)
        total_house_profit += game_profit
        
        if game_profit > 0:
            profitable_games += 1
    
    avg_profit_per_game = total_house_profit / len(df)
    
    print(f"\n📊 PROFITABILITY RESULTS:")
    print(f"Average house profit per $1 bet: ${avg_profit_per_game:.4f}")
    print(f"Profitable games: {profitable_games}/{len(df)} ({profitable_games/len(df)*100:.1f}%)")
    print(f"Target profit: $0.050 per $1 bet")
    
    if avg_profit_per_game >= 0.05:
        print(f"✅ SUCCESS: House profit {avg_profit_per_game:.4f} MEETS target!")
        status = "PROFITABLE"
    elif avg_profit_per_game >= 0.03:
        print(f"⚠️  CAUTION: House profit {avg_profit_per_game:.4f} below target but acceptable")
        status = "ACCEPTABLE"
    else:
        print(f"❌ INSUFFICIENT: House profit {avg_profit_per_game:.4f} below target")
        print(f"   Recommend increasing margin by {0.05 - avg_profit_per_game:.4f}")
        status = "INSUFFICIENT"
    
    return avg_profit_per_game, profitable_games, status

def simulate_betting_volume(data_file='data/house_odds_2025.csv', bets_per_game=10000):
    """
    模拟大量投注下的赌场收益
    """
    print(f"\n🎰 BETTING VOLUME SIMULATION")
    print("=" * 60)
    
    df = pd.read_csv(data_file)
    
    total_house_revenue = 0
    total_bets_placed = 0
    total_player_wins = 0
    total_house_wins = 0
    
    np.random.seed(42)  # Reproducible results
    
    print(f"Simulating {bets_per_game:,} bets per game...")
    
    for game_idx, row in df.iterrows():
        home_prob = row['Home Win %']
        away_prob = row['Away Win %']
        home_odds = row['Home Odds']
        away_odds = row['Away Odds']
        
        # Simulate betting on this game
        # Assume 50% bet on home, 50% bet on away
        home_bets = bets_per_game // 2
        away_bets = bets_per_game // 2
        bet_amount = 1  # $1 per bet
        
        # Simulate actual game outcome
        home_wins = np.random.random() < home_prob
        
        if home_wins:
            # Home team wins
            # House collects all away bets, pays home bettors
            house_revenue = away_bets * bet_amount - home_bets * (home_odds - 1) * bet_amount
            total_player_wins += home_bets
            total_house_wins += away_bets
        else:
            # Away team wins
            # House collects all home bets, pays away bettors
            house_revenue = home_bets * bet_amount - away_bets * (away_odds - 1) * bet_amount
            total_player_wins += away_bets
            total_house_wins += home_bets
        
        total_house_revenue += house_revenue
        total_bets_placed += bets_per_game
    
    avg_profit_per_bet = total_house_revenue / total_bets_placed
    house_edge_percentage = avg_profit_per_bet * 100
    
    print(f"\n💰 SIMULATION RESULTS:")
    print(f"Total bets simulated: {total_bets_placed:,}")
    print(f"Total house revenue: ${total_house_revenue:,.2f}")
    print(f"Average profit per $1 bet: ${avg_profit_per_bet:.4f}")
    print(f"House edge: {house_edge_percentage:.2f}%")
    print(f"Player wins: {total_player_wins:,} bets")
    print(f"House wins: {total_house_wins:,} bets")
    
    # Season projections
    daily_games = len(df) / 365  # Assuming year-round
    daily_revenue = avg_profit_per_bet * bets_per_game * daily_games
    
    print(f"\n📈 BUSINESS PROJECTIONS:")
    print(f"Expected daily revenue: ${daily_revenue:,.2f}")
    print(f"Expected monthly revenue: ${daily_revenue * 30:,.2f}")
    print(f"Expected annual revenue: ${daily_revenue * 365:,.2f}")
    
    return avg_profit_per_bet, total_house_revenue

def optimize_house_margin(data_file='data/house_odds_2025.csv'):
    """
    分析不同水钱设置下的赌场收益
    """
    print(f"\n⚙️  HOUSE MARGIN OPTIMIZATION")
    print("=" * 60)
    
    # Load base predictions (without margin applied)
    try:
        # We need to recalculate with different margins
        # For this, we'll use the probability data and test different margins
        df = pd.read_csv(data_file)
    except:
        print("❌ Cannot load data for optimization")
        return
    
    margins = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    
    print("Margin | Profit/Bet | Profitable Games | Volume Impact | Effective Profit")
    print("-" * 75)
    
    best_effective_profit = 0
    best_margin = 0.05
    
    for margin in margins:
        total_profit = 0
        profitable_games = 0
        
        for _, row in df.iterrows():
            home_prob = row['Home Win %']
            away_prob = row['Away Win %']
            
            # Recalculate odds with this margin
            adj_home_odds = (1 / home_prob) * (1 - margin)
            adj_away_odds = (1 / away_prob) * (1 - margin)
            
            # Calculate house expected profit
            home_profit = (1 - home_prob) * 1 - home_prob * (adj_home_odds - 1)
            away_profit = (1 - away_prob) * 1 - away_prob * (adj_away_odds - 1)
            
            game_profit = (home_profit + away_profit) / 2
            total_profit += game_profit
            
            if game_profit > 0:
                profitable_games += 1
        
        avg_profit = total_profit / len(df)
        
        # Estimate volume impact (higher margins = fewer bettors)
        # This is a simplified model
        base_volume = 1.0
        if margin <= 0.05:
            volume_factor = base_volume
        else:
            # Each 1% above 5% reduces volume by 15%
            volume_factor = base_volume * (1 - (margin - 0.05) * 1.5)
        
        volume_factor = max(0.3, volume_factor)  # Minimum 30% volume
        effective_profit = avg_profit * volume_factor
        
        if effective_profit > best_effective_profit:
            best_effective_profit = effective_profit
            best_margin = margin
        
        print(f"{margin:6.1%} | ${avg_profit:10.4f} | {profitable_games:15d} | {volume_factor:11.1%} | ${effective_profit:12.4f}")
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"Optimal margin: {best_margin:.1%}")
    print(f"Expected effective profit: ${best_effective_profit:.4f} per bet")
    print(f"Balances profitability with betting volume")

def generate_house_report(data_file='data/house_odds_2025.csv'):
    """
    生成完整的赌场运营报告
    """
    print("\n" + "="*80)
    print("🏦 COMPREHENSIVE HOUSE OPERATIONS REPORT")
    print("="*80)
    
    # Basic profitability analysis
    avg_profit, profitable_games, status = analyze_house_profitability(data_file)
    if avg_profit is None:
        return
    
    # Volume simulation
    simulated_profit, total_revenue = simulate_betting_volume(data_file, 10000)
    
    # Margin optimization
    optimize_house_margin(data_file)
    
    # Final recommendations
    print(f"\n📋 EXECUTIVE SUMMARY:")
    print("-" * 40)
    print(f"Overall Status: {status}")
    print(f"Current Profit Rate: ${avg_profit:.4f} per $1 bet")
    print(f"Profitable Games: {profitable_games}")
    print(f"Simulated Performance: ${simulated_profit:.4f} per $1 bet")
    
    if status == "PROFITABLE":
        print("✅ RECOMMENDATION: Maintain current margin structure")
        print("   System is generating target returns")
    elif status == "ACCEPTABLE":
        print("⚠️  RECOMMENDATION: Consider minor margin increase")
        print("   Monitor betting volume vs profitability")
    else:
        print("❌ RECOMMENDATION: Increase house margin immediately")
        print("   Current structure insufficient for sustainable operations")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"• Target 5% margin translates to $0.050 profit per $1 bet")
    print(f"• Current XGBoost model provides good prediction accuracy")
    print(f"• Balance between margin size and betting volume is crucial")
    print(f"• Monitor actual results vs predictions for model updates")

def main():
    parser = argparse.ArgumentParser(description='House Profitability Analysis System')
    parser.add_argument('--analysis', choices=['profit', 'volume', 'optimize', 'report'], 
                        default='report', help='Type of analysis to run')
    parser.add_argument('--data', default='data/house_odds_2025.csv',
                        help='Data file to analyze')
    args = parser.parse_args()
    
    if args.analysis == 'profit':
        analyze_house_profitability(args.data)
    elif args.analysis == 'volume':
        simulate_betting_volume(args.data)
    elif args.analysis == 'optimize':
        optimize_house_margin(args.data)
    else:
        generate_house_report(args.data)

if __name__ == "__main__":
    main()