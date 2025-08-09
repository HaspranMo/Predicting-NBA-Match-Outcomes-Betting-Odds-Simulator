
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from xgboost import XGBClassifier
import joblib

# 读取数据
df = pd.read_csv("data/regular_season_totals_2010_2024.csv")

# 构造新特征
df['FG%'] = df['FGM'] / df['FGA']
df['3P%'] = df['3PM'] / df['3PA']

# 构造 rolling 平均值（每队过去 5 场的平均表现）
rolling_stats = ['PTS', 'REB', 'AST', 'STL', 'TOV', 'FG%', '3P%']
for stat in rolling_stats:
    df[f"{stat}_rolling"] = df.groupby("Team")[stat].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())

# 创建比赛记录（主场队 vs 客场队）
games = df[df['Home'] == 1].copy()
games = games.rename(columns={col: f"{col}_home" for col in df.columns if col not in ['Season', 'Team', 'Opponent']})

away_df = df[df['Home'] == 0].copy()
away_df = away_df.rename(columns={col: f"{col}_away" for col in df.columns if col not in ['Season', 'Team', 'Opponent']})

matchup_stats = pd.merge(
    games,
    away_df,
    left_on=['Season', 'Team_home', 'Opponent_home'],
    right_on=['Season', 'Opponent_away', 'Team_away'],
    suffixes=('_home', '_away')
)

# 添加特征差值（主队减客队）
for stat in rolling_stats:
    matchup_stats[f"{stat}_diff_rolling"] = matchup_stats[f"{stat}_home_rolling"] - matchup_stats[f"{stat}_away_rolling"]

# 构造标签变量：主队是否赢了
matchup_stats['home_win'] = (matchup_stats['PTS_home'] > matchup_stats['PTS_away']).astype(int)

# 定义训练用的特征列
feature_columns = [f"{stat}_diff_rolling" for stat in rolling_stats]
X = matchup_stats[feature_columns]
y = matchup_stats['home_win']

# 分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# 构建 XGBoost 模型管道
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss'))
])

pipe.fit(X_train, y_train)

# 保存模型
joblib.dump(pipe, 'model/xgb_pipeline.pkl')

# 模型评估
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print("📊 Evaluation on Test Set:")
print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
print(f"Log Loss       : {log_loss(y_test, y_prob):.4f}")
print(f"ROC AUC Score  : {roc_auc_score(y_test, y_prob):.4f}")
