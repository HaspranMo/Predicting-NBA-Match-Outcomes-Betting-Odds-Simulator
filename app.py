#!/usr/bin/env python3
import os
import sys
from flask import Flask, render_template_string, request
import pandas as pd

# Create Flask app with explicit template folder
app = Flask(__name__, template_folder=os.path.abspath('templates'))

# HTML template as string (to avoid template file issues)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA 2025 Predicted Odds</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .filter-section {
            margin-bottom: 20px;
            text-align: center;
        }
        select {
            padding: 8px 12px;
            font-size: 16px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: white;
            min-width: 200px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
            color: #333;
            position: sticky;
            top: 0;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #e3f2fd;
        }
        .home-team {
            background-color: #e8f5e8 !important;
            font-weight: bold;
        }
        .away-team {
            background-color: #fff3e0 !important;
            font-weight: bold;
        }
        .percentage {
            font-weight: bold;
            color: #1976d2;
        }
        .odds {
            color: #d32f2f;
            font-weight: bold;
        }
        .footer {
            margin-top: 20px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }
        .no-data {
            text-align: center;
            padding: 40px;
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏀 NBA 2025 Predicted Odds</h1>
        
        <div class="filter-section">
            <label for="team-select">Filter by Team:</label>
            <select id="team-select" onchange="filterByTeam()">
                <option value="">All Teams</option>
                {% for team in teams %}
                <option value="{{ team }}" {{ 'selected' if selected == team else '' }}>{{ team }}</option>
                {% endfor %}
            </select>
        </div>

        {% if data %}
        <table>
            <thead>
                <tr>
                    <th>Home Team</th>
                    <th>Home Win %</th>
                    <th>Home Odds</th>
                    <th>Away Team</th>
                    <th>Away Win %</th>
                    <th>Away Odds</th>
                </tr>
            </thead>
            <tbody>
                {% for game in data %}
                <tr>
                    <td class="home-team">{{ game.home_team }}</td>
                    <td class="percentage">{{ "%.1f"|format(game.home_win_pct * 100) }}%</td>
                    <td class="odds">{{ "%.2f"|format(game.home_odds) }}</td>
                    <td class="away-team">{{ game.away_team }}</td>
                    <td class="percentage">{{ "%.1f"|format(game.away_win_pct * 100) }}%</td>
                    <td class="odds">{{ "%.2f"|format(game.away_odds) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <div class="no-data">
            <h3>No data available</h3>
            <p>Please check if the CSV file exists and contains data.</p>
        </div>
        {% endif %}

        <div class="footer">
            <p>Total Predictions: {{ total_count }}</p>
            <p>Data generated using XGBoost machine learning model</p>
        </div>
    </div>

    <script>
        function filterByTeam() {
            const select = document.getElementById('team-select');
            const selectedTeam = select.value;
            
            if (selectedTeam) {
                window.location.href = '/?team=' + encodeURIComponent(selectedTeam);
            } else {
                window.location.href = '/';
            }
        }
    </script>
</body>
</html>
'''

def load_data():
    """Load and validate the CSV data"""
    data_file = 'data/predicted_odds_2025.csv'
    
    print(f"Looking for data file: {os.path.abspath(data_file)}")
    
    if not os.path.exists(data_file):
        print(f"ERROR: {data_file} not found!")
        print("Current directory:", os.getcwd())
        print("Files in current directory:", os.listdir('.'))
        if os.path.exists('data'):
            print("Files in data directory:", os.listdir('data'))
        return None
    
    try:
        df = pd.read_csv(data_file)
        print("✅ Data loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Show first few rows
        print("\nFirst 3 rows:")
        print(df.head(3).to_string())
        
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

@app.route("/")
def index():
    # Load data
    df = load_data()
    if df is None:
        return render_template_string(HTML_TEMPLATE, 
                                    data=[], 
                                    teams=[], 
                                    selected=None, 
                                    total_count=0)
    
    # Get filter parameter
    team_filter = request.args.get("team", "").strip()
    print(f"Team filter: '{team_filter}'")
    
    # Apply team filter
    if team_filter and team_filter != "All Teams":
        filtered_df = df[(df['Home'] == team_filter) | (df['Away'] == team_filter)].copy()
        print(f"Filtered to {len(filtered_df)} rows for team: {team_filter}")
    else:
        filtered_df = df.copy()
        print(f"No filter applied, showing all {len(filtered_df)} rows")
    
    # Get list of unique teams
    home_teams = set(df['Home'].dropna()) if 'Home' in df.columns else set()
    away_teams = set(df['Away'].dropna()) if 'Away' in df.columns else set()
    teams = sorted(home_teams | away_teams)
    print(f"Found {len(teams)} unique teams")
    
    # Convert DataFrame to list of dictionaries
    data_list = []
    for _, row in filtered_df.head(100).iterrows():  # Limit to first 100 for testing
        try:
            data_list.append({
                'home_team': str(row['Home']),
                'home_win_pct': float(row['Home Win %']),
                'home_odds': float(row['Home Odds']),
                'away_team': str(row['Away']),
                'away_win_pct': float(row['Away Win %']),
                'away_odds': float(row['Away Odds'])
            })
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    print(f"Processed {len(data_list)} games for display")
    
    return render_template_string(HTML_TEMPLATE,
                                data=data_list,
                                teams=teams[:20],  # Limit teams for testing
                                selected=team_filter if team_filter else None,
                                total_count=len(data_list))

@app.route("/debug")
def debug():
    """Debug endpoint to check data"""
    df = load_data()
    if df is None:
        return "<h1>No data loaded</h1>"
    
    html = f"""
    <h1>Debug Information</h1>
    <h2>Data Shape: {df.shape}</h2>
    <h2>Columns: {df.columns.tolist()}</h2>
    <h2>First 5 rows:</h2>
    <pre>{df.head().to_string()}</pre>
    <h2>Data Types:</h2>
    <pre>{df.dtypes.to_string()}</pre>
    """
    return html

if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Flask app template folder: {app.template_folder}")
    
    app.run(debug=True, host='0.0.0.0', port=5001)