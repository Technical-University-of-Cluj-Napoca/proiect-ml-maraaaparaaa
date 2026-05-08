# Premier League Match Prediction — ML Project

A machine learning project that analyzes **4,788 Premier League matches** (2019–2024) to predict match outcomes and goals scored, with an interactive Streamlit dashboard.

---

## Overview

| | Problem | Target Variable | Best Model |
|---|---|---|---|
| Classification | Win / Draw / Loss | `result` | XGBoost |
| Regression | Goals Scored | `gf` | XGBoost |

**Dataset:** [Premier League Matches — Kaggle](https://www.kaggle.com/datasets/ajaxianazarenka/premier-league)  
**Seasons:** 2019–2024 | **Matches:** 4,788 | **Features:** 28 raw → 84 engineered

---

## Classification — Predict Match Result

**Goal:** Predict the outcome of a Premier League match (Win / Draw / Loss) from a team's perspective.

**Target:** `result` → W (2), D (1), L (0)

**Input features:** venue, xG, xGA, possession, shots, shots on target, goals conceded, formation, season, month.

**Key findings:**
- Classes are well balanced (W: ~40%, L: ~38%, D: ~22%) — no resampling needed
- Home advantage is a strong predictor: home teams score more and win more frequently
- `xg` correlates highly (~0.7) with actual goals, making it a top feature

**Models trained:** Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost, EBM

---

## Regression — Predict Goals Scored

**Goal:** Predict how many goals a team scores in a match.

**Target:** `gf` (Goals For) — integer values from 0 to 9, typical range 0–3

**Input features:** same as classification, minus the result label.

**Key findings:**
- Goals distribution is right-skewed; most matches end 1–2 goals
- `xg` and `sot` are the strongest predictors
- Home teams average more goals than away teams

**Models trained:** Linear Regression, Decision Tree, Random Forest, XGBoost, CatBoost

---

## Dataset — Column Reference

| Column | Full Name | Description |
|--------|-----------|-------------|
| `gf` | Goals For | Goals scored by the team |
| `ga` | Goals Against | Goals conceded |
| `xg` | Expected Goals | Shot-quality-based goal estimate |
| `xga` | Expected Goals Against | Opponent xG |
| `poss` | Possession | Ball possession % |
| `sh` | Shots | Total shots attempted |
| `sot` | Shots on Target | On-target shots |
| `dist` | Shot Distance | Average shot distance (yards) |
| `fk` | Free Kicks | Shots from free kicks |
| `pk` / `pkatt` | Penalties | Scored / attempted |
| `venue` | Venue | Home or Away |
| `result` | Result | W / D / L |
| `season` | Season | 2019–2024 |

---

## Streamlit App

An interactive dashboard (`app.py`) with three pages:

- **Home** — dataset overview, result and goals distributions
- **Classification** — model comparison, confusion matrix, learning curves, live prediction with SHAP explanation
- **Regression** — model comparison, actual vs predicted plot, learning curves, live prediction with SHAP explanation

### Run locally

```bash
git clone https://github.com/your-username/premier-league-ml.git
cd premier-league-ml
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

- **Python 3.10+**
- `pandas`, `numpy` — data processing
- `scikit-learn` — ML models, preprocessing, evaluation
- `xgboost`, `catboost`, `interpret` (EBM) — gradient boosting models
- `shap` — model explainability
- `matplotlib`, `seaborn` — visualizations
- `streamlit` — interactive dashboard
- `joblib` — model serialization

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
catboost
interpret
shap
streamlit
joblib
```

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost interpret shap streamlit joblib
```

---

## Data Preprocessing Steps

1. Drop irrelevant columns (`notes`, `match report`, `captain`, `referee`, `time`)
2. Drop `attendance` (~34% missing — COVID seasons)
3. Drop 2 rows where `dist` is missing
4. Encode categorical variables (one-hot / label encoding)
5. Feature engineering (new derived columns)
6. Feature scaling (StandardScaler)

---

## License

This project is for educational purposes. Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/ajaxianazarenka/premier-league).