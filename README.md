# NFL Predictor

A Python application that predicts both NFL against-the-spread (ATS) and over/under outcomes using the last 10 years of schedule and betting line data.

## Tech Stack

- `nflreadpy` for NFL schedule + spread data
- `pandas` for transformation and analysis
- `scikit-learn` for model training and prediction
- `seaborn` for visualizations

## What It Does

1. Downloads the last 10 years of NFL schedules, spreads, and totals.
2. Builds rolling team-form features.
3. Trains classification models to predict whether the home team covers and whether the total goes over.
4. Produces ATS and O/U picks with confidence scores for completed and upcoming games.
5. Saves evaluation charts and prediction CSV output.

## Setup

Python 3.10+ is required because `nflreadpy` does not support Python 3.9.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Optional arguments:

```bash
python main.py --season 2026 --week 1 --training-years 10
python main.py --season 2026 --training-years 10  # all weeks in season
```

## Outputs

- Season/week predictions CSV:
  - `reports/summaries/season_report_<season>.csv`
  - `reports/summaries/season_report_<season>_week_<week>.csv`
- Visualizations in `reports/figures/`:
  - `confusion_matrix.png`
  - `spread_cover_distribution.png`
  - `probability_calibration.png`
  - `feature_importance.png`
