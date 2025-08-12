#!/usr/bin/env python3
from flask import Flask, render_template_string
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    # Load house data
    try:
        df = pd.read_csv("data/house_odds_2025.csv")
    except FileNotFoundError:
        return "<h1>Error: Please run predict_2025.py first</h1>"
    
    # Calculate house metrics
    total_profit = 0
    profitable_games = 0
    
    for _, row in df.iterrows():
        home_prob = row['Home Win %']
        away_prob = row['Away Win %']
        home_odds = row['Home Odds']
        away_odds = row['Away Odds']
        
        home_profit = (1 - home_prob) * 1 - home_prob * (home_odds - 1)
        away_profit = (1 - away_prob) * 1 - away_prob * (away_odds - 1)
        avg_profit = (home_profit + away_profit) / 2
        
        total_profit += avg_profit
        if avg_profit > 0:
            profitable_games += 1
    
    house_profit_per_bet = total_profit / len(df)
    
    # Simple HTML template
    template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>House Edge Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; }
            .success { color: #4caf50; font-weight: bold; }
            .warning { color: #ff9800; font-weight: bold; }
            .metrics { display: flex; gap: 20px; margin: 20px 0; }
            .metric { background: #e3f2fd; padding: 15px; border-radius: 8px; text-align: center; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #1976d2; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏦 House Edge Management System</h1>
            
            <div class="metrics">
                <div class="metric">
                    <h3>${{ "%.4f"|format(house_profit) }}</h3>
                    <p>Profit per $1 Bet</p>
                </div>
                <div class="metric">
                    <h3>{{ profitable_games }}</h3>
                    <p>Profitable Games</p>
                </div>
                <div class="metric">
                    <h3>5.0%</h3>
                    <p>House Margin</p>
                </div>
                <div class="metric">
                    <h3>${{ "%.0f"|format(daily_revenue) }}</h3>
                    <p>Est. Daily Revenue</p>
                </div>
            </div>
            
            {% if house_profit >= 0.05 %}
            <p class="success">✅ House operations are PROFITABLE! Target achieved.</p>
            {% else %}
            <p class="warning">⚠️ House margin below target. Current: ${{ "%.4f"|format(house_profit) }}, Target: $0.0500</p>
            {% endif %}
            
            <h2>Sample Games (First 10)</h2>
            <table>
                <tr>
                    <th>Home Team</th>
                    <th>Home Odds</th>
                    <th>Away Team</th>
                    <th>Away Odds</th>
                    <th>House Profit/Bet</th>
                </tr>
                {% for _, row in data.head(10).iterrows() %}
                {% set home_prob = row['Home Win %'] %}
                {% set away_prob = row['Away Win %'] %}
                {% set home_odds = row['Home Odds'] %}
                {% set away_odds = row['Away Odds'] %}
                {% set home_profit = (1 - home_prob) - home_prob * (home_odds - 1) %}
                {% set away_profit = (1 - away_prob) - away_prob * (away_odds - 1) %}
                {% set avg_profit = (home_profit + away_profit) / 2 %}
                <tr>
                    <td>{{ row['Home'] }}</td>
                    <td>{{ "%.2f"|format(home_odds) }}</td>
                    <td>{{ row['Away'] }}</td>
                    <td>{{ "%.2f"|format(away_odds) }}</td>
                    <td style="color: {% if avg_profit >= 0.05 %}#4caf50{% else %}#ff9800{% endif %}">
                        ${{ "%.4f"|format(avg_profit) }}
                    </td>
                </tr>
                {% endfor %}
            </table>
            
            <p style="margin-top: 30px; color: #666; text-align: center;">
                Technology: XGBoost ML Model + 5% House Margin Strategy<br>
                Total Games: {{ data|length }} | Target: $0.05 profit per $1 bet
            </p>
        </div>
    </body>
    </html>
    '''
    
    daily_revenue = house_profit_per_bet * 10000 * (len(df) / 365)
    
    return render_template_string(template, 
                                data=df,
                                house_profit=house_profit_per_bet,
                                profitable_games=profitable_games,
                                daily_revenue=daily_revenue)

if __name__ == "__main__":
    print("🌐 Starting House Edge Web Dashboard...")
    app.run(debug=True, host='0.0.0.0', port=5001)