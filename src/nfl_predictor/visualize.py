from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


sns.set_theme(style="whitegrid")


def ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    fig_dir = output_dir / "figures"
    pred_dir = output_dir / "predictions"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, pred_dir


def plot_confusion(test_predictions: pd.DataFrame, fig_dir: Path) -> None:
    cm = confusion_matrix(test_predictions["home_covers"], test_predictions["pred_home_covers"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Home Team Cover Prediction Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=180)
    plt.close()


def plot_cover_by_spread(test_predictions: pd.DataFrame, fig_dir: Path) -> None:
    plot_df = test_predictions.dropna(subset=["spread_line", "home_covers"]).copy()
    plot_df["home_covers"] = plot_df["home_covers"].astype(int)

    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=plot_df,
        x="spread_line",
        hue="home_covers",
        bins=20,
        multiple="stack",
        palette="Set2",
    )
    plt.title("Distribution of Cover Outcomes by Spread Line")
    plt.xlabel("Spread Line (Home Team)")
    plt.ylabel("Game Count")
    plt.tight_layout()
    plt.savefig(fig_dir / "spread_cover_distribution.png", dpi=180)
    plt.close()


def plot_probability_calibration(test_predictions: pd.DataFrame, fig_dir: Path) -> None:
    calib = test_predictions.dropna(subset=["pred_prob_home_covers", "home_covers"]).copy()
    calib["prob_bin"] = pd.cut(calib["pred_prob_home_covers"], bins=np.linspace(0, 1, 11), include_lowest=True)

    summary = (
        calib.groupby("prob_bin", observed=False)
        .agg(mean_pred_prob=("pred_prob_home_covers", "mean"), actual_cover_rate=("home_covers", "mean"), games=("home_covers", "size"))
        .reset_index(drop=True)
    )

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=summary, x="mean_pred_prob", y="actual_cover_rate", marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Predicted vs Actual Cover Rate (Calibration)")
    plt.xlabel("Mean Predicted Home Cover Probability")
    plt.ylabel("Actual Home Cover Rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "probability_calibration.png", dpi=180)
    plt.close()


def plot_feature_importance(feature_importance: pd.DataFrame, fig_dir: Path, top_n: int = 20) -> None:
    top = feature_importance.head(top_n).copy()
    top = top.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=top, x="importance", y="feature", orient="h", palette="viridis")
    plt.title(f"Top {top_n} Model Features")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance.png", dpi=180)
    plt.close()
