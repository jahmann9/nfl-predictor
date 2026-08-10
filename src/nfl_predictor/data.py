from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import nflreadpy as nfl


def _to_pandas(df: object) -> pd.DataFrame:
    """Convert nflreadpy output to pandas while supporting both Polars and pandas returns."""
    if isinstance(df, pd.DataFrame):
        return df.copy()

    if hasattr(df, "to_pandas"):
        return df.to_pandas()  # type: ignore[no-any-return]

    return pd.DataFrame(df)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_schedule_data(start_season: int, end_season: int) -> pd.DataFrame:
    seasons = list(range(start_season, end_season + 1))
    raw = nfl.load_schedules(seasons)
    schedule = _to_pandas(raw)

    schedule.columns = [str(c).lower() for c in schedule.columns]

    expected = [
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "spread_line",
        "total_line",
        "home_rest",
        "away_rest",
        "roof",
        "surface",
        "game_type",
    ]
    schedule = _ensure_columns(schedule, expected)

    for col in [
        "season",
        "week",
        "home_score",
        "away_score",
        "spread_line",
        "total_line",
        "home_rest",
        "away_rest",
    ]:
        schedule[col] = pd.to_numeric(schedule[col], errors="coerce")

    schedule["gameday"] = pd.to_datetime(schedule["gameday"], errors="coerce")
    schedule = schedule.sort_values(["gameday", "season", "week"], na_position="last").reset_index(drop=True)
    schedule["game_id"] = schedule["game_id"].fillna(
        schedule["season"].astype("Int64").astype(str)
        + "_"
        + schedule["week"].astype("Int64").astype(str)
        + "_"
        + schedule["away_team"].fillna("UNK").astype(str)
        + "_at_"
        + schedule["home_team"].fillna("UNK").astype(str)
    )

    return schedule


def load_team_pbp_game_features(start_season: int, end_season: int) -> pd.DataFrame:
    """Load and aggregate team-level game features from play-by-play.

    Pace is modeled as offensive plays per game as a practical pace proxy.
    """
    try:
        max_pbp_season = int(nfl.get_current_season())
    except Exception:
        max_pbp_season = end_season

    pbp_end_season = min(end_season, max_pbp_season)
    if start_season > pbp_end_season:
        return pd.DataFrame()

    seasons = list(range(start_season, pbp_end_season + 1))

    try:
        raw = nfl.load_pbp(seasons)
    except TypeError:
        raw = nfl.load_pbp()

    pbp = _to_pandas(raw)
    pbp.columns = [str(c).lower() for c in pbp.columns]

    expected = [
        "game_id",
        "season",
        "week",
        "posteam",
        "defteam",
        "epa",
        "play_type",
    ]
    pbp = _ensure_columns(pbp, expected)

    for col in ["season", "week", "epa"]:
        pbp[col] = pd.to_numeric(pbp[col], errors="coerce")

    pbp = pbp[pbp["season"].between(start_season, pbp_end_season)].copy()

    # Restrict to offensive snaps for cleaner EPA and pace metrics.
    if "play_type" in pbp.columns:
        pbp = pbp[pbp["play_type"].isin(["run", "pass"])].copy()

    off = (
        pbp.dropna(subset=["game_id", "posteam"])
        .groupby(["game_id", "season", "week", "posteam"], as_index=False)
        .agg(
            offensive_epa_per_play=("epa", "mean"),
            offensive_plays=("epa", "size"),
        )
        .rename(columns={"posteam": "team"})
    )

    defense = (
        pbp.dropna(subset=["game_id", "defteam"])
        .groupby(["game_id", "season", "week", "defteam"], as_index=False)
        .agg(defensive_epa_allowed_per_play=("epa", "mean"))
        .rename(columns={"defteam": "team"})
    )

    return off.merge(defense, on=["game_id", "season", "week", "team"], how="outer")
