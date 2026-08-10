# Weekly NFL Spread Predictor

A Python application that predicts weekly NFL against-the-spread (ATS) outcomes using the last 10 years of schedule and betting line data.

## Tech Stack

- `nflreadpy` for NFL schedule + spread data
- `pandas` for transformation and analysis
- `scikit-learn` for model training and prediction
- `seaborn` for visualizations

## What It Does

1. Downloads the last 10 years of NFL schedules and spreads.
2. Builds rolling team-form features.
3. Trains a classification model to predict whether the home team covers.
4. Produces weekly ATS picks and confidence scores.
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
```

## Outputs

- Weekly picks CSV: `reports/predictions/weekly_picks_<season>_week_<week>.csv`
- Visualizations in `reports/figures/`:
  - `confusion_matrix.png`
  - `spread_cover_distribution.png`
  - `probability_calibration.png`
  - `feature_importance.png`
