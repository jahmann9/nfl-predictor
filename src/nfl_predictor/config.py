from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictorConfig:
    training_years: int = 10
    rolling_window_games: int = 5
    random_state: int = 42


NUMERIC_FEATURES = [
    "home_spread_line",
    "total_line",
    "home_rest",
    "away_rest",
    "rest_diff",
    "home_point_diff_last_n",
    "away_point_diff_last_n",
    "home_ats_rate_last_n",
    "away_ats_rate_last_n",
    "home_score_for_last_n",
    "away_score_for_last_n",
    "home_score_against_last_n",
    "away_score_against_last_n",
    "home_off_epa_last_n",
    "away_off_epa_last_n",
    "home_def_epa_allowed_last_n",
    "away_def_epa_allowed_last_n",
    "home_pace_plays_last_n",
    "away_pace_plays_last_n",
    "point_diff_form_edge",
    "ats_form_edge",
    "off_epa_form_edge",
    "def_epa_form_edge",
    "pace_form_edge",
]

CATEGORICAL_FEATURES = [
    "roof",
    "surface",
    "game_type",
]
