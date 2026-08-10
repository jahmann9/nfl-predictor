from __future__ import annotations

import numpy as np
import pandas as pd


def _build_long_games(schedule: pd.DataFrame, team_game_features: pd.DataFrame | None = None) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "game_id": schedule["game_id"],
            "season": schedule["season"],
            "week": schedule["week"],
            "gameday": schedule["gameday"],
            "team": schedule["home_team"],
            "opponent": schedule["away_team"],
            "is_home": 1,
            "score_for": schedule["home_score"],
            "score_against": schedule["away_score"],
            "spread_for_team": -schedule["spread_line"],
        }
    )

    away = pd.DataFrame(
        {
            "game_id": schedule["game_id"],
            "season": schedule["season"],
            "week": schedule["week"],
            "gameday": schedule["gameday"],
            "team": schedule["away_team"],
            "opponent": schedule["home_team"],
            "is_home": 0,
            "score_for": schedule["away_score"],
            "score_against": schedule["home_score"],
            "spread_for_team": schedule["spread_line"],
        }
    )

    long_games = pd.concat([home, away], ignore_index=True)

    if team_game_features is not None and not team_game_features.empty:
        keep_cols = [
            "game_id",
            "season",
            "week",
            "team",
            "offensive_epa_per_play",
            "defensive_epa_allowed_per_play",
            "offensive_plays",
        ]
        features = team_game_features.loc[:, [c for c in keep_cols if c in team_game_features.columns]].copy()
        long_games = long_games.merge(features, on=["game_id", "season", "week", "team"], how="left")

    long_games["point_diff"] = long_games["score_for"] - long_games["score_against"]
    long_games["ats_result"] = np.where(
        long_games["score_for"].notna() & long_games["score_against"].notna() & long_games["spread_for_team"].notna(),
        (long_games["score_for"] + long_games["spread_for_team"] > long_games["score_against"]).astype(float),
        np.nan,
    )
    return long_games


def _add_rolling_form(long_games: pd.DataFrame, rolling_window_games: int) -> pd.DataFrame:
    long_games = long_games.sort_values(["team", "gameday", "season", "week"]).copy()

    def rolling_shifted_mean(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(rolling_window_games, min_periods=1).mean()

    long_games["point_diff_last_n"] = (
        long_games.groupby("team", group_keys=False)["point_diff"].transform(rolling_shifted_mean)
    )
    long_games["ats_rate_last_n"] = (
        long_games.groupby("team", group_keys=False)["ats_result"].transform(rolling_shifted_mean)
    )
    long_games["score_for_last_n"] = (
        long_games.groupby("team", group_keys=False)["score_for"].transform(rolling_shifted_mean)
    )
    long_games["score_against_last_n"] = (
        long_games.groupby("team", group_keys=False)["score_against"].transform(rolling_shifted_mean)
    )
    long_games["off_epa_last_n"] = (
        long_games.groupby("team", group_keys=False)["offensive_epa_per_play"].transform(rolling_shifted_mean)
    )
    long_games["def_epa_allowed_last_n"] = (
        long_games.groupby("team", group_keys=False)["defensive_epa_allowed_per_play"].transform(rolling_shifted_mean)
    )
    long_games["pace_plays_last_n"] = (
        long_games.groupby("team", group_keys=False)["offensive_plays"].transform(rolling_shifted_mean)
    )
    return long_games


def build_model_frame(
    schedule: pd.DataFrame,
    team_game_features: pd.DataFrame | None = None,
    rolling_window_games: int = 5,
) -> pd.DataFrame:
    long_games = _build_long_games(schedule, team_game_features=team_game_features)
    long_games = _add_rolling_form(long_games, rolling_window_games=rolling_window_games)

    home = (
        long_games[long_games["is_home"] == 1]
        .loc[
            :,
            [
                "game_id",
                "point_diff_last_n",
                "ats_rate_last_n",
                "score_for_last_n",
                "score_against_last_n",
                "off_epa_last_n",
                "def_epa_allowed_last_n",
                "pace_plays_last_n",
            ],
        ]
        .rename(
            columns={
                "point_diff_last_n": "home_point_diff_last_n",
                "ats_rate_last_n": "home_ats_rate_last_n",
                "score_for_last_n": "home_score_for_last_n",
                "score_against_last_n": "home_score_against_last_n",
                "off_epa_last_n": "home_off_epa_last_n",
                "def_epa_allowed_last_n": "home_def_epa_allowed_last_n",
                "pace_plays_last_n": "home_pace_plays_last_n",
            }
        )
    )

    away = (
        long_games[long_games["is_home"] == 0]
        .loc[
            :,
            [
                "game_id",
                "point_diff_last_n",
                "ats_rate_last_n",
                "score_for_last_n",
                "score_against_last_n",
                "off_epa_last_n",
                "def_epa_allowed_last_n",
                "pace_plays_last_n",
            ],
        ]
        .rename(
            columns={
                "point_diff_last_n": "away_point_diff_last_n",
                "ats_rate_last_n": "away_ats_rate_last_n",
                "score_for_last_n": "away_score_for_last_n",
                "score_against_last_n": "away_score_against_last_n",
                "off_epa_last_n": "away_off_epa_last_n",
                "def_epa_allowed_last_n": "away_def_epa_allowed_last_n",
                "pace_plays_last_n": "away_pace_plays_last_n",
            }
        )
    )

    model_df = schedule.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")

    model_df["home_spread_line"] = -model_df["spread_line"]

    model_df["rest_diff"] = model_df["home_rest"] - model_df["away_rest"]
    model_df["point_diff_form_edge"] = model_df["home_point_diff_last_n"] - model_df["away_point_diff_last_n"]
    model_df["ats_form_edge"] = model_df["home_ats_rate_last_n"] - model_df["away_ats_rate_last_n"]
    model_df["off_epa_form_edge"] = model_df["home_off_epa_last_n"] - model_df["away_off_epa_last_n"]
    model_df["def_epa_form_edge"] = (
        model_df["away_def_epa_allowed_last_n"] - model_df["home_def_epa_allowed_last_n"]
    )
    model_df["pace_form_edge"] = model_df["home_pace_plays_last_n"] - model_df["away_pace_plays_last_n"]

    model_df["home_cover_margin"] = model_df["home_score"] + model_df["home_spread_line"] - model_df["away_score"]
    model_df["home_covers"] = np.where(
        model_df["home_cover_margin"].notna(),
        (model_df["home_cover_margin"] > 0).astype(int),
        np.nan,
    )

    return model_df
