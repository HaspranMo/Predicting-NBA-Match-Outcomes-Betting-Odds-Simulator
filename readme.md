# 🏀 NBA Match Outcome Predictor & Betting Odds Simulator

This project uses real NBA regular season data to:

✅ Predict the winner of NBA games using a machine learning model  
✅ Calculate European-style betting odds based on predicted win probabilities  
✅ Simulate betting outcomes to verify that the platform makes ~$0.05 per $1 bet  
✅ Display 2025 predicted matchups and odds through a Flask web interface
🆕 Optimize house edge through advanced XGBoost modeling and comprehensive profit analysis
🆕 Ensure consistent platform profitability across all betting scenarios
🆕 Provide detailed house operations dashboard and revenue projections

---
🔄 Project Evolution & Optimization
Original Implementation

-Basic logistic regression model with simple features
-Standard betting simulation
-Basic web interface for viewing odds

### 🆕 Enhanced House Edge System
- **Upgraded to XGBoost**: Enhanced prediction model with improved accuracy (~71.6%)
- **Expanded feature engineering**: 6 rolling statistical features per team (vs. original 3)
- **Game-by-game profitability calculation**: Individual game profit analysis for house operations
- **Large-scale betting simulation**: 10,000+ bets per game volume modeling
- **Fixed margin verification**: Validation that 5% house edge consistently achieves target $0.05 per bet
- **Revenue estimation**: Daily, monthly, and annual revenue projections
- **Profitability ranking**: Identification of most/least profitable games for house operations

## 📂 Project Structure

```
project/
│
├── data/
│   ├── regular_season_totals_2010_2024.csv
│   ├── old_predicted_odds_2025.csv
│   ├── house_odds_2025.csv
│   └── predicted_odds_2025.csv
│
├── model/
│   ├── logistic_pipeline.pkl
│   ├── old_predict_2025.py
│   ├── old_simulate_betting.py
│   ├── old_train_model_optimized.py
│   ├── train_model.py
│   ├── train_model_optimized.py
│   ├── predict_2025.py
│   ├── simulate_betting.py
│   ├── xgb_pipeline.pkl
│   ├── house_analysis.py
│   └── ...
│
├── templates/
│   ├── house_index.html
│   ├── old_index.html
│   └── index.html
│
├── static/
│   └── style.css
│
├── screenshot/
│   ├── flask.jpg
│   ├── supernew_flask.png
│   ├── supernew_predict.png
│   ├── supernew_simulate1.png
│   ├── supernew_simulate2.png
│   ├── supernew_train.png
│   ├── new_flask.png
│   ├── new_predict_2025.png
│   ├── new_simulate_betting.png
│   ├── new_train_model_optimized.png
│   ├── predict.jpg
│   ├── simulate.jpg
│   └── train.jpg
│
├── app.py
├── old_app.py
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install flask pandas scikit-learn xgboost
```

---

## 🧠(Original Implementation) Model Training (`train_model.py`)

### 🎯 Goal

Predict whether the **home team will win** a game using statistical features derived from recent team performance.

---

### 📊 Data Source

- File: `data/regular_season_totals_2010_2024.csv`
- Columns: team stats per game from 2010–2024 NBA regular seasons
- Key fields: `TEAM_ABBREVIATION`, `GAME_DATE`, `MATCHUP`, `PTS`, `AST`, `REB`, `WL` (Win/Loss)

---

### 🏗️ Feature Engineering

We aggregate **rolling averages from each team's last 5 games** to better simulate real-world prediction (i.e. only use past data available before each game):

- `PTS_AVG`: average points
- `AST_AVG`: average assists
- `REB_AVG`: average rebounds

Each game is split into **home** and **away** sides based on `MATCHUP`, and we compute:

```
PTS_DIFF = HOME_PTS_AVG - AWAY_PTS_AVG
AST_DIFF = HOME_AST_AVG - AWAY_AST_AVG
REB_DIFF = HOME_REB_AVG - AWAY_REB_AVG
```

---

### 🎯 Target Variable

- `HOME_WIN = 1` if home team won the game
- `HOME_WIN = 0` otherwise

---

### 🧪 Model & Evaluation

- Model: `LogisticRegression` with `class_weight="balanced"` to handle win/loss imbalance
- Pipeline: uses `StandardScaler` for feature normalization
- Evaluation metrics:
  - Accuracy
  - Log Loss
  - ROC AUC

Example results:

```
Accuracy     : 0.6751
Log Loss     : 0.5843
ROC AUC Score: 0.7382
```

---

### 💾 Output

Trained model is saved as:

```
model/logistic_pipeline.pkl
```

This model is then used to predict future matchups.

---

## 🆕 Enhanced House Edge Training (train_model_optimized.py)
### 🎯 Enhanced Goal
Optimize prediction accuracy for house edge calculation using XGBoost machine learning.
### 🏗️ Expanded Feature Engineering
Enhanced to 6 rolling statistical features for comprehensive team analysis:

- FG_PCT_rolling: Field goal percentage (shooting efficiency)
- 3P%_rolling: Three-point percentage (long-range accuracy)
- REB_rolling: Rebounds (board control)
- AST_rolling: Assists (team cooperation)
- STL_rolling: Steals (defensive pressure)
- TOV_rolling: Turnovers (ball security)

Each feature calculated as 5-game rolling average for both home and away teams, creating 12 total features for prediction.
### 🧪 Enhanced Model & Evaluation

Model: XGBClassifier with standard hyperparameters for NBA prediction
Pipeline: StandardScaler + XGBClassifier for enhanced predictive capability
Training: 80/20 split with stratification for balanced evaluation
Evaluation metrics:
  - Accuracy: ~71.6%
  - Log Loss: ~0.5649
  - ROC AUC: ~0.7683


### 💾 Enhanced Output
Enhanced model saved as:
```
model/xgb_pipeline.pkl
```
**Key Improvement** : Expanded feature set and XGBoost algorithm provide enhanced prediction capability for house operations.



## 🔮 Predict 2025 Matches & Odds
### Original Implementation
Use the trained model to simulate 2025 games and generate odds:

```bash
python predict_2025.py
```

Output: `data/predicted_odds_2025.csv`

Each row contains:

- Teams
- Predicted win probabilities
- European odds (adjusted with house margin)

---

### 🆕 House-Focused Prediction (predict_2025.py --model xgb)
Enhanced prediction system with house profitability validation:

```bash
python predict_2025.py --model xgb
```
### 🎯 House Edge Features

- XGBoost predictions: Enhanced accuracy using expanded feature set
- Profit preview: Sample house profit calculation during generation
- Consistent odds calculation: Ensures 5% house margin across all games
- Quality verification: Basic validation that house edge targets are applied

### 💾 Enhanced Output
```
data/house_odds_2025.csv
```
### Key Improvements:

  - Enhanced probability estimates using XGBoost
  - Built-in profit preview during generation
  - Consistent house margin application

## 💸 Simulate Betting Platform Profit
### Original Implementation
Validate that the platform earns ~5 cents per $1 bet:

```bash
python simulate_betting.py
```

Each game simulates two $1 bettors (home and away). The house earns:

- ~$0.05 per bet on average (via odds adjustment)

---

## 🆕 Enhanced House Profit Analysis (simulate_betting.py)
Expanded house operations simulation with multiple analysis modes:

```bash
# Full house operations report
python simulate_betting.py --analysis report

# Profit analysis only  
python simulate_betting.py --analysis profit

# Volume simulation (custom bet amounts)
python simulate_betting.py --analysis volume --bets_per_game 10000

# Basic margin comparison
python simulate_betting.py --analysis optimize
```
## 📊 Enhanced Reporting
```
🏦 HOUSE OPERATIONS REPORT
============================================================

💰 HOUSE PROFIT SUMMARY:
   Average profit per game: $0.0500 per $1 bet ✅
   Profitable games: 16658/16658 (100.0%)
   Target profit: $0.05 per $1 bet
   SUCCESS: House margin achieves target!

💵 SIMULATION RESULTS:
   Total betting volume: $166,580,000
   Total house revenue: $8,329,000.00
   Average profit per $1 bet: $0.0500
   House win rate: 100.0% of games

📈 REVENUE ESTIMATIONS:
   Daily revenue: $22,820.27
   Monthly revenue: $684,608.22
   Annual revenue: $8,329,000.00
```
## 🌐 Run Flask Web App
Original Implementation
To explore predictions in your browser:

```bash
flask run
```

Visit: http://localhost:5000/

- Filter games by team
- View win probability and payout odds

---
## 🆕 House Operations Dashboard
Enhanced web interface for house edge management:

```bash
python simple_app.py
```
Visit: http://localhost:5001/


## 🔢 Odds Calculation

We use European (decimal) odds, with a built-in house margin:

```
Adjusted Odds = (1 / Win Probability) * (1 - Margin)
```

Example:  
If predicted win probability is 60% and margin is 5%:

```
Odds = (1 / 0.6) * 0.95 ≈ 1.59
```

This ensures the house profits approximately $0.05 per $1 bet, regardless of outcome.

---

## 📚 Data Source

The dataset is manually downloaded from Kaggle NBA sources:
- Includes game-by-game team statistics from 2010 to 2024
- Format: one row per team per game

## 🚀 Quick Start Guide
### Original Workflow
```
# 1. Train enhanced XGBoost model
python train_model_optimized.py

# 2. Generate house-validated odds
python predict_2025.py --model xgb

# 3. Analyze house operations
python simulate_betting.py --analysis report

# 4. Launch house operations dashboard
python app.py
```
---
## 🖼️ Screenshots

### 🔧 Model Training Output

<img width="715" height="192" alt="supernew_train" src="https://github.com/user-attachments/assets/63c6b758-4420-41e8-9251-5de59728d779" />

### 🔮 Prediction Example

<img width="683" height="366" alt="supernew_predict" src="https://github.com/user-attachments/assets/8f9548b3-8e31-4cc9-a97f-f6ca2e292b7c" />

### 💰 Simulated Betting Results

<img width="717" height="701" alt="supernew_simulate1" src="https://github.com/user-attachments/assets/33f301c2-2896-4dd9-9edb-403074eb3895" />
<img width="659" height="592" alt="supernew_simulate2" src="https://github.com/user-attachments/assets/819646db-34ce-4ca5-82e2-d3db4c19e0d9" />


### 🌐 Flask Web View

<img width="1140" height="800" alt="supernew_flask" src="https://github.com/user-attachments/assets/34858e4c-3ec8-4f07-babc-953f31950fbd" />

 
