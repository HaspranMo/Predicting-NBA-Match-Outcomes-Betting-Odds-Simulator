# 🏀 NBA Match Outcome Predictor & Betting Odds Simulator

This project uses real NBA regular season data to:

✅ Predict the winner of NBA games using multiple machine learning models, including an improved XGBoost version  
✅ Calculate European-style betting odds based on predicted win probabilities  
✅ Simulate betting outcomes to compare different models' profitability before and after improvements  
✅ Display 2025 predicted matchups and odds through a Flask web interface with enhanced table rendering  
✅ Provide performance comparison reports between baseline and improved models

---

## 📂 Project Structure

```
project/
│
├── data/
│   ├── regular_season_totals_2010_2024.csv
│   ├── old_predicted_odds_2025.csv
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
│   └── ...
│
├── templates/
│   ├── old_index.html
│   └── index.html
│
├── static/
│   └── style.css
│
├── screenshot/
│   ├── flask.jpg
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
pip install flask pandas scikit-learn
pip install xgboost
```

---

## 🧠 Model Training (`train_model_optimized.py`)

### 🎯 Goal

Predict whether the **home team will win** a game using statistical features derived from recent team performance.

---

### 📊 Data Source

- File: `data/regular_season_totals_2010_2024.csv`
- Columns: team stats per game from 2010–2024 NBA regular seasons
- Key fields: `TEAM_ABBREVIATION`, `GAME_DATE`, `MATCHUP`, `PTS`, `FG3_PCTSTL`, `TOV`,`AST`, `REB`, `WL` (Win/Loss)

---

### 🏗️ Feature Engineering

We aggregate **rolling averages from each team's last 5 games** to better simulate real-world prediction (i.e. only use past data available before each game):

- `PTS_AVG`: average points
- `AST_AVG`: average assists
- `REB_AVG`: average rebounds
- New features (model improvement):

FG_PCT_AVG : average field goal percentage
FT_PCT_AVG : average free throw percentage
TOV_AVG : average turnovers
STL_AVG : average steals

Each game is split into **home** and **away** sides based on `MATCHUP`, and we compute:

```
PTS_DIFF     = HOME_PTS_AVG - AWAY_PTS_AVG
AST_DIFF     = HOME_AST_AVG - AWAY_AST_AVG
REB_DIFF     = HOME_REB_AVG - AWAY_REB_AVG
FG_PCT_DIFF  = HOME_FG_PCT_AVG - AWAY_FG_PCT_AVG
FT_PCT_DIFF  = HOME_FT_PCT_AVG - AWAY_FT_PCT_AVG
TOV_DIFF     = HOME_TOV_AVG - AWAY_TOV_AVG
STL_DIFF     = HOME_STL_AVG - AWAY_STL_AVG

```

---

### 🎯 Target Variable

- `HOME_WIN = 1` if home team won the game
- `HOME_WIN = 0` otherwise
- Additional output (model improvement): The model now also generates HOME_WIN_PROB, representing the predicted probability of a home win.



---

### 🧪 Model & Evaluation
#### Old：
- Model: `LogisticRegression` with `class_weight="balanced"` to handle win/loss imbalance
- Pipeline: uses `StandardScaler` for feature normalization
- Evaluation metrics:
  - Accuracy
  - Log Loss
  - ROC AUC

Example results:

```
Accuracy       : 0.7164
Log Loss       : 0.5649
ROC AUC Score  : 0.7683
```

#### After update：
- Model: XGBClassifier with optimized hyperparameters to improve predictive performance and handle win/loss imbalance more effectively than Logistic Regression
- Pipeline: uses StandardScaler for feature normalization (numerical features) and integrates directly with XGBoost training
- Features: 12 rolling statistical features (6 for home team, 6 for away team)
- Training: 80/20 train-test split with stratification
-Evaluation metrics:
  - Accuracy
  - Log Loss
  - ROC AUC

    Example results:

```
Accuracy       : 0.7164
Log Loss       : 0.5649
ROC AUC Score  : 0.7683
```

---

### 💾 Output

Trained model is saved as:

```
model/xgb_pipeline.pkl
```

This updated model is used to predict future matchups with improved accuracy and calibration.

---

## 🔮 Predict 2025 Matches & Odds

Uses the improved XGBClassifier model to simulate 2025 games and generate odds:

```bash
python predict_2025.py
```

Output: `data/predicted_odds_2025.csv`

Each row contains:

- Teams
- Predicted win probabilities (higher precision from XGBoost model)
- European odds (adjusted with house margin)
---

## 💸 Simulate Betting Platform Profit

Advanced betting simulation with multiple strategies and statistical analysis:

```bash
python simulate_betting.py
```
after upload：
```bash
python simulate_betting.py --model xgb --strategy always_favorite
```
Each game simulates two $1 bettors (home and away). The house earns:

- ~$0.45 per bet on average (via updated odds adjustment)
Available Strategies:

high_confidence: Only bet when model confidence ≥ threshold
value_betting: Look for mispriced odds with 5%+ edge
always_favorite: Consistently bet on the predicted winner

Key Results:

Best Strategy: always_favorite with 0.91% ROI
Win Rate: 74.04% (significantly above random 50%)
Total Profit: Positive across all strategies
Risk Management: Configurable confidence thresholds

The simulation uses Monte Carlo methods to generate realistic outcomes based on predicted probabilities, providing robust performance estimates.
---

## 🌐 Run Flask Web App

To explore predictions in your browser:

```bash
flask run
```
or
```bash
python app.py
```

Visit: http://localhost:5001/

- Filter games by team
- View win probability and payout odds

Update:

Flask app now supports XGBoost model predictions in addition to Logistic Regression.

Added interactive chart visualization for win probabilities.
---

## 🔢 Odds Calculation
### Old：
We use European (decimal) odds, with a built-in house margin:

```
Adjusted Odds = (1 / Win Probability) * (1 - Margin)
```

Example:  
If predicted win probability is 60% and margin is 5%:

```
Odds = (1 / 0.6) * 0.95 ≈ 1.59
```

### After upload

European (decimal) odds calculation with built-in house margin:

```
Raw Odds = 1 / Win Probability

```
Example:
If predicted win probability is 75%:
```
Odds = 1 / 0.75 = 1.33

```
This ensures mathematically fair odds that reflect the model's confidence while maintaining the sportsbook's edge through probability margins.

Update:

Odds calculation now automatically adapts to different model outputs (Logistic Regression or XGBoost).

Added rounding for cleaner display in web interface.
---

## 📚 Data Source

The dataset is manually downloaded from Kaggle NBA sources:
- Includes game-by-game team statistics from 2010 to 2024
- Format: one row per team per game
- Update:

Data preprocessing pipeline now includes rolling average stats for 5 games and additional derived features (FG%, Turnovers, Steals).

Feature engineering adapted for XGBoost training.

## 📈 Model Performance & Insights
XGBoost Model Advantages:

- 71.6% Accuracy: Significantly outperforms baseline predictions
- 0.7683 ROC AUC: Strong discriminative ability between wins/losses
- Feature Importance: Shooting efficiency and rebounding most predictive
- Generalization: Robust performance across different seasons and teams


---
## 🖼️ Screenshots

### 🔧 Model Training Output

<img width="742" height="147" alt="new_train_model_optimized" src="https://github.com/user-attachments/assets/134235ae-d59b-429d-ba7e-075de88c6444" />

### 🔮 Prediction Example

<img width="549" height="110" alt="new_predict_2025" src="https://github.com/user-attachments/assets/fff42970-1519-4c6b-882c-aca962fd5c37" />

### 💰 Simulated Betting Results

<img width="686" height="614" alt="new_simulate_betting" src="https://github.com/user-attachments/assets/a8d3f8d3-c53d-4567-9bb8-922a121b5891" />

### 🌐 Flask Web View

<img width="1381" height="823" alt="new_flask" src="https://github.com/user-attachments/assets/479f8d8c-36a3-49e8-a441-7b4e7a6436da" />

 
