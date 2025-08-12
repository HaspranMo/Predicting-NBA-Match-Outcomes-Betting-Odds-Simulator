import pandas as pd
import numpy as np
import argparse

def analyze_house_profitability():
    """
    Analyze house profitability from current odds structure
    Target: $0.05 profit per $1 bet
    """
    try:
        df = pd.read_csv("data/house_odds_2025.csv")
    except FileNotFoundError:
        print("❌ Error: house_odds_2025.csv not found!")
        print("🔧 Please run: python predict_2025.py --model xgb")
        return
    
    print("🏦 HOUSE PROFITABILITY ANALYSIS")
    print("="*60)
    print(f"📊 Analyzing {len(df)} games for house edge")
    
    total_house_profit = 0
    profitable_games = 0
    detailed_analysis = []
    
    for _, row in df.iterrows():
        home_prob = row['Home Win %']
        away_prob = row['Away Win %']
        home_odds = row['Home Odds']
        away_odds = row['Away Odds']
        
        # Calculate house expected profit per $1 bet on each side
        # If player bets $1 on home team:
        # - House wins $1 if away team wins (probability = away_prob)
        # - House loses $(home_odds-1) if home team wins (probability = home_prob)
        home_house_profit = away_prob * 1 - home_prob * (home_odds - 1)
        
        # If player bets $1 on away team:
        away_house_profit = home_prob * 1 - away_prob * (away_odds - 1)
        
        # Assume balanced betting (50% on each side)
        average_profit = (home_house_profit + away_house_profit) / 2
        
        total_house_profit += average_profit
        if average_profit > 0:
            profitable_games += 1
            
        detailed_analysis.append({
            'game': f"{row['Home']} vs {row['Away']}",
            'home_profit': home_house_profit,
            'away_profit': away_house_profit,
            'avg_profit': average_profit
        })
    
    avg_profit_per_game = total_house_profit / len(df)
    
    print(f"\n💰 HOUSE PROFIT SUMMARY:")
    print(f"   Average profit per game: ${avg_profit_per_game:.4f} per $1 bet")
    print(f"   Profitable games: {profitable_games}/{len(df)} ({profitable_games/len(df)*100:.1f}%)")
    print(f"   Target profit: $0.05 per $1 bet")
    
    # Performance evaluation
    if avg_profit_per_game >= 0.05:
        print(f"   ✅ SUCCESS: House margin {avg_profit_per_game:.4f} EXCEEDS target!")
        profit_margin = (avg_profit_per_game - 0.05) / 0.05 * 100
        print(f"   📈 Exceeding target by {profit_margin:.1f}%")
    elif avg_profit_per_game >= 0.045:
        print(f"   ⚠️  CLOSE: House margin {avg_profit_per_game:.4f} near target")
        print(f"   🎯 Within acceptable range for house operations")
    else:
        print(f"   ❌ INSUFFICIENT: House margin {avg_profit_per_game:.4f} below target")
        shortfall = 0.05 - avg_profit_per_game
        increase_needed = shortfall / avg_profit_per_game * 100 if avg_profit_per_game > 0 else 100
        print(f"   📉 Need to increase margin by {increase_needed:.1f}%")
    
    # Show top 5 most/least profitable games
    detailed_analysis.sort(key=lambda x: x['avg_profit'], reverse=True)
    
    print(f"\n🔝 TOP 5 MOST PROFITABLE GAMES:")
    for i, game in enumerate(detailed_analysis[:5], 1):
        print(f"   {i}. {game['game']}: ${game['avg_profit']:.4f} per $1 bet")
    
    print(f"\n🔻 TOP 5 LEAST PROFITABLE GAMES:")
    for i, game in enumerate(detailed_analysis[-5:], 1):
        print(f"   {i}. {game['game']}: ${game['avg_profit']:.4f} per $1 bet")
    
    return avg_profit_per_game, profitable_games

def simulate_betting_volume(bets_per_game=10000):
    """
    Simulate house revenue with realistic betting volume
    """
    try:
        df = pd.read_csv("data/house_odds_2025.csv")
    except FileNotFoundError:
        print("❌ Error: house_odds_2025.csv not found!")
        return
    
    print(f"\n🎰 BETTING VOLUME SIMULATION")
    print("="*60)
    print(f"📊 Simulating {bets_per_game:,} bets per game")
    
    total_house_revenue = 0
    total_volume = 0
    house_wins = 0
    house_losses = 0
    
    np.random.seed(42)  # Reproducible results
    
    for _, row in df.iterrows():
        home_prob = row['Home Win %']
        away_prob = row['Away Win %'] 
        home_odds = row['Home Odds']
        away_odds = row['Away Odds']
        
        # Simulate betting distribution (assume 50/50 split)
        home_bets = bets_per_game // 2
        away_bets = bets_per_game // 2
        game_volume = home_bets + away_bets
        
        # Simulate actual game outcome
        home_wins_game = np.random.random() < home_prob
        
        if home_wins_game:
            # Home team wins: house collects away bets, pays home bets
            house_revenue = away_bets * 1.0 - home_bets * (home_odds - 1)
        else:
            # Away team wins: house collects home bets, pays away bets  
            house_revenue = home_bets * 1.0 - away_bets * (away_odds - 1)
        
        total_house_revenue += house_revenue
        total_volume += game_volume
        
        if house_revenue > 0:
            house_wins += 1
        else:
            house_losses += 1
    
    avg_profit_per_bet = total_house_revenue / total_volume
    house_win_rate = house_wins / len(df)
    
    print(f"\n💵 SIMULATION RESULTS:")
    print(f"   Total betting volume: ${total_volume:,}")
    print(f"   Total house revenue: ${total_house_revenue:,.2f}")
    print(f"   Average profit per $1 bet: ${avg_profit_per_bet:.4f}")
    print(f"   House win rate: {house_win_rate:.1%} of games")
    print(f"   Games house won: {house_wins}")
    print(f"   Games house lost: {house_losses}")
    
    # Revenue projections
    daily_games = len(df) / 365  # Assume season spread over year
    daily_revenue = (total_house_revenue / len(df)) * daily_games
    monthly_revenue = daily_revenue * 30
    annual_revenue = total_house_revenue
    
    print(f"\n📈 REVENUE PROJECTIONS:")
    print(f"   Daily revenue: ${daily_revenue:,.2f}")
    print(f"   Monthly revenue: ${monthly_revenue:,.2f}")
    print(f"   Annual revenue: ${annual_revenue:,.2f}")
    
    return avg_profit_per_bet

def margin_optimization_analysis():
    """
    Analyze optimal house margin for maximum profitability
    """
    try:
        df = pd.read_csv("data/house_odds_2025.csv")
    except FileNotFoundError:
        print("❌ Error: house_odds_2025.csv not found!")
        return
    
    print(f"\n🔧 HOUSE MARGIN OPTIMIZATION")
    print("="*60)
    
    margins = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    
    print("Margin  | Profit/Bet | Profitable Games | Volume Impact | Effective Profit")
    print("-"*75)
    
    for margin in margins:
        total_profit = 0
        profitable_games = 0
        
        for _, row in df.iterrows():
            home_prob = row['Home Win %']
            away_prob = row['Away Win %']
            
            # Recalculate odds with different margin
            home_odds_adj = (1 / home_prob) * (1 - margin)
            away_odds_adj = (1 / away_prob) * (1 - margin)
            
            # Calculate house profit
            home_profit = (1 - home_prob) * 1 - home_prob * (home_odds_adj - 1)
            away_profit = (1 - away_prob) * 1 - away_prob * (away_odds_adj - 1)
            avg_profit = (home_profit + away_profit) / 2
            
            total_profit += avg_profit
            if avg_profit > 0:
                profitable_games += 1
        
        avg_profit = total_profit / len(df)
        
        # Estimate volume impact (higher margins = lower volume)
        baseline_margin = 0.05
        volume_factor = max(0.4, 1 - (margin - baseline_margin) * 3)
        effective_profit = avg_profit * volume_factor
        
        print(f"{margin:6.1%}  | ${avg_profit:9.4f} | {profitable_games:15d} | {volume_factor:10.1%} | ${effective_profit:12.4f}")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   • 5% margin provides good balance of profit and volume")
    print(f"   • Higher margins increase profit per bet but may reduce volume")
    print(f"   • Monitor actual betting patterns to optimize")

def generate_house_report():
    """
    Generate comprehensive house operations report
    """
    print("🏦" + "="*59)
    print("🏦 COMPREHENSIVE HOUSE OPERATIONS REPORT")
    print("🏦" + "="*59)
    
    # Run all analyses
    avg_profit, profitable_games = analyze_house_profitability()
    simulated_profit = simulate_betting_volume(10000)
    margin_optimization_analysis()
    
    print(f"\n📋 EXECUTIVE SUMMARY:")
    print("="*40)
    
    if avg_profit >= 0.05:
        print("✅ HOUSE OPERATIONS: PROFITABLE")
        print(f"   Current profit margin: ${avg_profit:.4f} per $1 bet")
        print(f"   Status: EXCEEDING target of $0.05")
    else:
        print("⚠️  HOUSE OPERATIONS: NEEDS OPTIMIZATION")
        print(f"   Current profit margin: ${avg_profit:.4f} per $1 bet")
        print(f"   Status: BELOW target of $0.05")
    
    print(f"\n🎯 KEY METRICS:")
    print(f"   • Profitable games: {profitable_games}")
    print(f"   • Expected profit per bet: ${avg_profit:.4f}")
    print(f"   • House margin: 5%")
    print(f"   • Volume simulation: 10,000 bets/game")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', 
                        choices=['profit', 'volume', 'optimize', 'report'], 
                        default='report',
                        help='Type of house analysis to run')
    parser.add_argument('--bets_per_game', type=int, default=10000,
                        help='Number of bets to simulate per game')
    args = parser.parse_args()
    
    if args.analysis == 'profit':
        analyze_house_profitability()
    elif args.analysis == 'volume':
        simulate_betting_volume(args.bets_per_game)
    elif args.analysis == 'optimize':
        margin_optimization_analysis()
    else:
        generate_house_report()

if __name__ == "__main__":
    main()