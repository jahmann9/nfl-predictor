from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import nflreadpy as nfl

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PredictorConfig
from .data import load_schedule_data, load_team_pbp_game_features
from .features import build_model_frame
from .model import fit_full_model, train_and_evaluate
from .visualize import ensure_dirs, plot_confusion, plot_cover_by_spread, plot_feature_importance, plot_probability_calibration


def _current_season_week() -> tuple[int, int]:
    try:
        return int(nfl.get_current_season()), int(nfl.get_current_week())
    except Exception:
        # Fallback in case upstream service call fails.
        now = pd.Timestamp.utcnow()
        fallback_season = int(now.year if now.month >= 8 else now.year - 1)
        return fallback_season, 1


def _pick_text(home_team: str, away_team: str, home_spread_line: float, prob_home_cover: float) -> str:
    if prob_home_cover >= 0.5:
        return f"{home_team} to cover ({home_spread_line:+.1f})"
    away_line = -home_spread_line
    return f"{away_team} to cover ({away_line:+.1f})"


def run() -> None:
    parser = argparse.ArgumentParser(description="Weekly NFL spread predictor")
    parser.add_argument("--season", type=int, default=None, help="Target prediction season")
    parser.add_argument("--week", type=int, default=None, help="Target prediction week")
    parser.add_argument("--training-years", type=int, default=10, help="How many historical seasons to use")
    parser.add_argument("--rolling-window", type=int, default=5, help="Rolling window for team-form features")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    args = parser.parse_args()

    current_season, current_week = _current_season_week()
    target_season = args.season or current_season
    target_week = args.week or current_week

    config = PredictorConfig(
        training_years=args.training_years,
        rolling_window_games=args.rolling_window,
    )

    start_season = target_season - config.training_years + 1
    print(f"Loading schedules from {start_season} to {target_season}...")

    schedule = load_schedule_data(start_season=start_season, end_season=target_season)
    team_game_features = load_team_pbp_game_features(start_season=start_season, end_season=target_season)
    model_df = build_model_frame(
        schedule,
        team_game_features=team_game_features,
        rolling_window_games=config.rolling_window_games,
    )

    train_df = model_df[
        model_df["season"].between(start_season, target_season)
        & model_df["home_covers"].notna()
        & model_df["spread_line"].notna()
    ].copy()

    if len(train_df) < 200:
        raise ValueError("Not enough historical games found to train a stable model.")

    available_numeric = [c for c in NUMERIC_FEATURES if c in train_df.columns]
    available_categorical = [c for c in CATEGORICAL_FEATURES if c in train_df.columns]

    artifacts = train_and_evaluate(
        train_df,
        numeric_features=available_numeric,
        categorical_features=available_categorical,
        random_state=config.random_state,
    )

    output_dir = Path(args.output_dir)
    fig_dir, pred_dir = ensure_dirs(output_dir)

    plot_confusion(artifacts.test_predictions, fig_dir)
    plot_cover_by_spread(artifacts.test_predictions, fig_dir)
    plot_probability_calibration(artifacts.test_predictions, fig_dir)
    plot_feature_importance(artifacts.feature_importance, fig_dir)

    full_model = fit_full_model(
        train_df,
        numeric_features=available_numeric,
        categorical_features=available_categorical,
        random_state=config.random_state,
    )

    weekly = model_df[
        (model_df["season"] == target_season)
        & (model_df["week"] == target_week)
        & (model_df["home_score"].isna() | model_df["away_score"].isna())
        & model_df["spread_line"].notna()
    ].copy()

    if weekly.empty:
        print(f"No upcoming games found for season={target_season}, week={target_week}.")
    else:
        weekly["away_spread_line"] = weekly["spread_line"]
        weekly["home_spread_line"] = -weekly["spread_line"]

        feature_cols = [*available_numeric, *available_categorical]
        probs = full_model.predict_proba(weekly[feature_cols])[:, 1]
        weekly["pred_prob_home_cover"] = probs
        weekly["pred_home_covers"] = (weekly["pred_prob_home_cover"] >= 0.5).astype(int)
        weekly["recommended_pick"] = weekly.apply(
            lambda row: _pick_text(
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_spread_line=float(row["home_spread_line"]),
                prob_home_cover=float(row["pred_prob_home_cover"]),
            ),
            axis=1,
        )

        picks = weekly[
            [
                "season",
                "week",
                "gameday",
                "away_team",
                "home_team",
                "away_spread_line",
                "home_spread_line",
                "pred_prob_home_cover",
                "pred_home_covers",
                "recommended_pick",
            ]
        ].sort_values("pred_prob_home_cover", ascending=False)

        out_file = pred_dir / f"weekly_picks_{target_season}_week_{target_week}.csv"
        picks.to_csv(out_file, index=False)
        print(f"Saved weekly predictions to: {out_file}")

    print("\nModel Evaluation")
    print("-" * 40)
    print(f"Train games: {int(artifacts.metrics['train_games'])}")
    print(f"Test games:  {int(artifacts.metrics['test_games'])}")
    print(f"Accuracy:    {artifacts.metrics['accuracy']:.3f}")
    if "roc_auc" in artifacts.metrics:
        print(f"ROC AUC:     {artifacts.metrics['roc_auc']:.3f}")

    print("\nSaved visualizations:")
    print(f"- {fig_dir / 'confusion_matrix.png'}")
    print(f"- {fig_dir / 'spread_cover_distribution.png'}")
    print(f"- {fig_dir / 'probability_calibration.png'}")
    print(f"- {fig_dir / 'feature_importance.png'}")
