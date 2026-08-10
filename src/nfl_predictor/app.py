from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import nflreadpy as nfl

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, PredictorConfig
from .data import load_schedule_data, load_team_pbp_game_features
from .features import build_model_frame
from .model import fit_full_model, train_and_evaluate
from .report import generate_weekly_html
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
    parser = argparse.ArgumentParser(description="NFL Predictor (Spread + O/U)")
    parser.add_argument("--season", type=int, default=None, help="Target prediction season")
    parser.add_argument("--week", type=int, default=None, help="Target prediction week (omit for all weeks)")
    parser.add_argument("--training-years", type=int, default=10, help="How many historical seasons to use")
    parser.add_argument("--rolling-window", type=int, default=5, help="Rolling window for team-form features")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    args = parser.parse_args()

    current_season, _ = _current_season_week()
    target_season = args.season or current_season
    target_week = args.week

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

    # For O/U model, also require total_goes_over to be available
    train_df_ou = train_df[train_df["total_goes_over"].notna()].copy() if "total_goes_over" in train_df.columns else train_df.copy()

    if len(train_df) < 200:
        raise ValueError("Not enough historical games found to train a stable model.")

    available_numeric = [c for c in NUMERIC_FEATURES if c in train_df.columns]
    available_categorical = [c for c in CATEGORICAL_FEATURES if c in train_df.columns]

    # Train spread model
    artifacts = train_and_evaluate(
        train_df,
        numeric_features=available_numeric,
        categorical_features=available_categorical,
        random_state=config.random_state,
        target="home_covers",
    )

    # Train O/U model if we have enough data
    artifacts_ou = None
    if "total_goes_over" in train_df_ou.columns and len(train_df_ou) >= 200:
        artifacts_ou = train_and_evaluate(
            train_df_ou,
            numeric_features=available_numeric,
            categorical_features=available_categorical,
            random_state=config.random_state,
            target="total_goes_over",
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
        target="home_covers",
    )

    full_model_ou = None
    if artifacts_ou is not None:
        full_model_ou = fit_full_model(
            train_df_ou,
            numeric_features=available_numeric,
            categorical_features=available_categorical,
            random_state=config.random_state,
            target="total_goes_over",
        )

    def _build_export_frame(frame: pd.DataFrame) -> pd.DataFrame:
        # Stat columns for the detail panel – include whatever is available.
        _detail_stat_cols = [
            "home_off_epa_last_n", "away_off_epa_last_n",
            "home_def_epa_allowed_last_n", "away_def_epa_allowed_last_n",
            "home_pace_plays_last_n", "away_pace_plays_last_n",
            "home_ats_rate_last_n", "away_ats_rate_last_n",
            "home_point_diff_last_n", "away_point_diff_last_n",
            "home_score_for_last_n", "away_score_for_last_n",
            "home_score_against_last_n", "away_score_against_last_n",
            "home_rest", "away_rest",
            "total_line",
        ]
        _base_cols = [
            "season", "week", "gameday",
            "away_team", "home_team",
            "away_score", "home_score",
            "away_spread_line", "home_spread_line",
            "pred_prob_home_cover", "pred_home_covers",
            "recommended_pick",
        ]
        _ou_cols = [
            "pred_prob_total_over", "pred_total_over", "ou_pick",
        ]
        _review_cols = [
            "actual_home_covers",
            "actual_result",
            "was_correct",
            "home_cover_margin",
            "actual_total_over",
            "was_correct_ou",
        ]

        cols = _base_cols + [c for c in _detail_stat_cols if c in frame.columns] + [
            c for c in _ou_cols if c in frame.columns
        ] + [
            c for c in _review_cols if c in frame.columns
        ]
        picks = frame[cols].copy()
        picks["pick_probability"] = picks["pred_prob_home_cover"].apply(lambda p: max(p, 1 - p))
        return picks.sort_values("pick_probability", ascending=False)

    feature_cols = [*available_numeric, *available_categorical]

    picks_scope = model_df[
        (model_df["season"] == target_season)
        & model_df["spread_line"].notna()
    ].copy()
    if target_week is not None:
        picks_scope = picks_scope[picks_scope["week"] == target_week].copy()

    if picks_scope.empty:
        scope = f"season={target_season}" + (f", week={target_week}" if target_week is not None else "")
        print(f"No games with spread lines found for {scope}.")
    else:
        picks_scope["away_spread_line"] = picks_scope["spread_line"]
        picks_scope["home_spread_line"] = -picks_scope["spread_line"]

        # Spread predictions
        probs = full_model.predict_proba(picks_scope[feature_cols])[:, 1]
        picks_scope["pred_prob_home_cover"] = probs
        picks_scope["pred_home_covers"] = (picks_scope["pred_prob_home_cover"] >= 0.5).astype(int)
        picks_scope["recommended_pick"] = picks_scope.apply(
            lambda row: _pick_text(
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_spread_line=float(row["home_spread_line"]),
                prob_home_cover=float(row["pred_prob_home_cover"]),
            ),
            axis=1,
        )

        # O/U predictions
        if full_model_ou is not None:
            ou_probs = full_model_ou.predict_proba(picks_scope[feature_cols])[:, 1]
            picks_scope["pred_prob_total_over"] = ou_probs
            picks_scope["pred_total_over"] = (picks_scope["pred_prob_total_over"] >= 0.5).astype(int)
            picks_scope["ou_pick"] = picks_scope.apply(
                lambda row: (
                    f"Over {float(row['total_line']):.1f}"
                    if float(row["pred_prob_total_over"]) >= 0.5
                    else f"Under {float(row['total_line']):.1f}"
                ),
                axis=1,
            )
        else:
            picks_scope["pred_prob_total_over"] = 0.5
            picks_scope["pred_total_over"] = 0
            picks_scope["ou_pick"] = ""

        # Completed games get outcomes; future games remain prediction-only.
        completed_mask = picks_scope["home_score"].notna() & picks_scope["away_score"].notna()
        picks_scope["actual_home_covers"] = np.where(
            completed_mask,
            (picks_scope["home_cover_margin"] > 0).astype(int),
            np.nan,
        )
        picks_scope["actual_result"] = ""
        completed_rows = picks_scope[completed_mask]
        if not completed_rows.empty:
            picks_scope.loc[completed_mask, "actual_result"] = completed_rows.apply(
                lambda row: (
                    f"{row['home_team']} covered ({float(row['home_spread_line']):+.1f})"
                    if float(row["home_cover_margin"]) > 0
                    else (
                        f"{row['away_team']} covered ({float(row['away_spread_line']):+.1f})"
                        if float(row["home_cover_margin"]) < 0
                        else "Push"
                    )
                ),
                axis=1,
            )

        if "total_points" in picks_scope.columns:
            picks_scope["actual_total_over"] = np.where(
                completed_mask,
                (picks_scope["total_points"] > picks_scope["total_line"]).astype(int),
                np.nan,
            )
            picks_scope["was_correct_ou"] = np.where(
                completed_mask,
                picks_scope["pred_total_over"] == picks_scope["actual_total_over"],
                np.nan,
            )
        else:
            picks_scope["actual_total_over"] = np.nan
            picks_scope["was_correct_ou"] = np.nan

        picks_scope["was_correct"] = np.where(
            completed_mask,
            picks_scope["pred_home_covers"] == picks_scope["actual_home_covers"],
            np.nan,
        )

        picks = _build_export_frame(picks_scope)

        if target_week is None:
            csv_name = f"season_report_{target_season}.csv"
            html_name = f"season_report_{target_season}.html"
        else:
            csv_name = f"season_report_{target_season}_week_{target_week}.csv"
            html_name = f"season_report_{target_season}_week_{target_week}.html"

        out_file = pred_dir / csv_name
        picks.to_csv(out_file, index=False)
        print(f"Saved predictions CSV to: {out_file}")

        html_file = pred_dir / html_name
        generate_weekly_html(
            picks=picks,
            season=target_season,
            week=target_week,
            model_metrics=artifacts.metrics,
            model_metrics_ou=artifacts_ou.metrics if artifacts_ou is not None else None,
            output_path=html_file,
        )

        completed_games = int(completed_mask.sum())
        upcoming_games = int((~completed_mask).sum())
        print(f"Games in report: {len(picks)} total ({completed_games} completed, {upcoming_games} upcoming)")
        if completed_games > 0:
            correct_rate = picks.loc[picks["was_correct"].notna(), "was_correct"].mean()
            print(f"Completed-game correctness: {correct_rate:.3f}")

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
