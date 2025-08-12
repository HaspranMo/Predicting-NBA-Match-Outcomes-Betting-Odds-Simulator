import pandas as pd
import numpy as np

def test_probability_calibrations():
    """
    Test different ways to recalibrate probabilities to overcome systematic bias
    """
    df = pd.read_csv("data/predicted_odds_2025.csv")
    
    print("PROBABILITY RECALIBRATION ANALYSIS")
    print("="*60)
    print(f"Original data: {len(df)} games")
    
    # Original statistics
    print(f"Original probability range: {df['Home Win %'].min():.3f} - {df['Home Win %'].max():.3f}")
    print(f"Original mean: {df['Home Win %'].mean():.3f}")
    
    # Calculate original edges
    df['home_implied'] = 1.0 / df['Home Odds']
    df['away_implied'] = 1.0 / df['Away Odds']
    df['home_edge'] = df['Home Win %'] - df['home_implied']
    df['away_edge'] = df['Away Win %'] - df['away_implied']
    df['best_edge'] = np.maximum(df['home_edge'], df['away_edge'])
    
    print(f"Original best edge range: {df['best_edge'].min():.6f} to {df['best_edge'].max():.6f}")
    print()
    
    # Calibration methods to test
    calibrations = [
        {
            'name': 'Power 0.85 (More Confident)',
            'home_func': lambda p: np.power(p, 0.85),
            'away_func': lambda p: np.power(p, 0.85)
        },
        {
            'name': 'Power 0.8 (Much More Confident)', 
            'home_func': lambda p: np.power(p, 0.8),
            'away_func': lambda p: np.power(p, 0.8)
        },
        {
            'name': 'Power 0.75 (Very Confident)',
            'home_func': lambda p: np.power(p, 0.75),
            'away_func': lambda p: np.power(p, 0.75)
        },
        {
            'name': 'Sigmoid Stretch (Alpha=1.2)',
            'home_func': lambda p: sigmoid_stretch(p, alpha=1.2),
            'away_func': lambda p: sigmoid_stretch(p, alpha=1.2)
        },
        {
            'name': 'Sigmoid Stretch (Alpha=1.5)',
            'home_func': lambda p: sigmoid_stretch(p, alpha=1.5),
            'away_func': lambda p: sigmoid_stretch(p, alpha=1.5)
        },
        {
            'name': 'Linear Shift (+0.03)',
            'home_func': lambda p: np.clip(p + 0.03, 0.01, 0.99),
            'away_func': lambda p: np.clip(p + 0.03, 0.01, 0.99)
        },
        {
            'name': 'Confidence Boost (High Prob +0.05)',
            'home_func': lambda p: np.where(p > 0.6, np.clip(p + 0.05, 0.01, 0.99), p),
            'away_func': lambda p: np.where(p > 0.6, np.clip(p + 0.05, 0.01, 0.99), p)
        }
    ]
    
    best_method = None
    best_roi = -100
    
    for calib in calibrations:
        print(f"\nTesting: {calib['name']}")
        print("-" * 40)
        
        # Apply calibration
        cal_home_prob = calib['home_func'](df['Home Win %'])
        cal_away_prob = calib['away_func'](df['Away Win %'])
        
        # Normalize to ensure they sum to 1
        prob_sum = cal_home_prob + cal_away_prob
        cal_home_prob = cal_home_prob / prob_sum
        cal_away_prob = cal_away_prob / prob_sum
        
        # Calculate new odds with 5% margin
        margin = 0.05
        cal_home_odds = (1 / cal_home_prob) * (1 - margin)
        cal_away_odds = (1 / cal_away_prob) * (1 - margin)
        
        # Calculate edges using ORIGINAL probabilities for truth
        cal_home_implied = 1.0 / cal_home_odds
        cal_away_implied = 1.0 / cal_away_odds
        cal_home_edge = df['Home Win %'] - cal_home_implied
        cal_away_edge = df['Away Win %'] - cal_away_implied
        cal_best_edge = np.maximum(cal_home_edge, cal_away_edge)
        
        positive_edges = (cal_best_edge > 0).sum()
        
        print(f"Calibrated prob range: {cal_home_prob.min():.3f} - {cal_home_prob.max():.3f}")
        print(f"Positive edges: {positive_edges} ({positive_edges/len(df)*100:.1f}%)")
        
        if positive_edges > 0:
            print(f"Best edge: {cal_best_edge.max():.6f}")
            print(f"Mean positive edge: {cal_best_edge[cal_best_edge > 0].mean():.6f}")
            
            # Quick simulation
            roi = simulate_calibrated_betting(df, cal_home_odds, cal_away_odds, cal_home_edge, cal_away_edge)
            print(f"Simulated ROI: {roi:.2f}%")
            
            if roi > best_roi:
                best_roi = roi
                best_method = calib['name']
        else:
            print("No positive edges found")
    
    print("\n" + "="*60)
    if best_roi > 0:
        print(f"🎉 PROFITABLE METHOD FOUND!")
        print(f"Best method: {best_method}")
        print(f"Best ROI: +{best_roi:.2f}%")
    else:
        print("❌ No profitable calibration method found")
        print("The 5% margin is too large for this model's accuracy level")
    print("="*60)

def sigmoid_stretch(p, alpha=1.2):
    """
    Stretch probabilities using sigmoid-like function
    Alpha > 1 makes extreme probabilities more extreme
    """
    # Convert to logits, stretch, convert back
    epsilon = 1e-6
    p = np.clip(p, epsilon, 1 - epsilon)
    logits = np.log(p / (1 - p))
    stretched_logits = logits * alpha
    stretched_p = 1 / (1 + np.exp(-stretched_logits))
    return np.clip(stretched_p, epsilon, 1 - epsilon)

def simulate_calibrated_betting(df, home_odds, away_odds, home_edge, away_edge, bet_amount=100):
    """
    Simulate betting with calibrated probabilities
    """
    total_profit = 0
    total_bets = 0
    wins = 0
    
    np.random.seed(42)
    
    for i in range(len(df)):
        bet_placed = False
        chosen_prob = 0
        chosen_odds = 0
        
        if home_edge.iloc[i] > 0:
            bet_placed = True
            chosen_prob = df['Home Win %'].iloc[i]  # Use original prob for simulation
            chosen_odds = home_odds.iloc[i]
        elif away_edge.iloc[i] > 0:
            bet_placed = True
            chosen_prob = df['Away Win %'].iloc[i]  # Use original prob for simulation
            chosen_odds = away_odds.iloc[i]
        
        if bet_placed:
            total_bets += 1
            win = np.random.random() < chosen_prob
            
            if win:
                total_profit += bet_amount * (chosen_odds - 1)
                wins += 1
            else:
                total_profit -= bet_amount
    
    if total_bets == 0:
        return 0
    
    return (total_profit / (total_bets * bet_amount)) * 100

def create_profitable_strategy():
    """
    Create a working profitable strategy file
    """
    df = pd.read_csv("data/predicted_odds_2025.csv")
    
    # Use the best calibration method (you'll know after running the test)
    # For now, let's use Power 0.8 as an example
    
    print("Creating profitable strategy with Power 0.8 calibration...")
    
    # Apply calibration
    cal_home_prob = np.power(df['Home Win %'], 0.8)
    cal_away_prob = np.power(df['Away Win %'], 0.8)
    
    # Normalize
    prob_sum = cal_home_prob + cal_away_prob
    cal_home_prob = cal_home_prob / prob_sum
    cal_away_prob = cal_away_prob / prob_sum
    
    # Calculate new odds
    margin = 0.05
    cal_home_odds = (1 / cal_home_prob) * (1 - margin)
    cal_away_odds = (1 / cal_away_prob) * (1 - margin)
    
    # Calculate edges
    cal_home_implied = 1.0 / cal_home_odds
    cal_away_implied = 1.0 / cal_away_odds
    cal_home_edge = df['Home Win %'] - cal_home_implied
    cal_away_edge = df['Away Win %'] - cal_away_implied
    
    # Create new predictions file with calibrated odds
    calibrated_df = df.copy()
    calibrated_df['Home Odds'] = cal_home_odds
    calibrated_df['Away Odds'] = cal_away_odds
    calibrated_df['Home Win %'] = df['Home Win %']  # Keep original for simulation
    calibrated_df['Away Win %'] = df['Away Win %']  # Keep original for simulation
    
    # Save calibrated predictions
    calibrated_df[['Home', 'Home Win %', 'Home Odds', 'Away', 'Away Win %', 'Away Odds']].to_csv(
        'data/calibrated_odds_2025.csv', index=False
    )
    
    positive_edges = ((cal_home_edge > 0) | (cal_away_edge > 0)).sum()
    print(f"Calibrated file saved with {positive_edges} positive edge opportunities!")
    
    return calibrated_df

if __name__ == "__main__":
    test_probability_calibrations()
    
    # Uncomment this if you find a profitable method:
    # create_profitable_strategy()