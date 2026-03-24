"""
Plotting utility for run_temporal_merging_experiments.py outputs.

Reads CSV files from:
  analysis/temporal_merging/<experiment_name>/

Generates figures under:
  analysis/temporal_merging/<experiment_name>/plots/
"""

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Path to experiment result directory (contains CSV outputs).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI.",
    )
    return p.parse_args()


def safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def numeric_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def plot_router_stats(summary_df: pd.DataFrame, out_dir: str, dpi: int):
    if summary_df.empty:
        return
    # plot key metrics only for readability
    candidate_metrics = [
        "router_conf_mean_mean",
        "vmem_abs_mean_mean",
        "vmem_var_t_mean_mean",
        "vmem_entropy_mean_mean",
        "spike_rate_mean_mean",
        "spike_entropy_mean_mean",
    ]
    metrics = [m for m in candidate_metrics if m in summary_df.columns]
    if not metrics:
        metrics = numeric_cols(summary_df, exclude=["block", "expert"])[:6]
    if not metrics:
        return

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        for expert_id in sorted(summary_df["expert"].unique().tolist()):
            sub = summary_df[summary_df["expert"] == expert_id].sort_values("block")
            ax.plot(
                sub["block"].values,
                sub[metric].values,
                marker="o",
                linewidth=1.8,
                label=f"Expert {expert_id}",
            )
        ax.set_title(f"Router stats per block: {metric}")
        ax.set_xlabel("Block")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"router_{metric}.png"), dpi=dpi)
        plt.close(fig)


def plot_t_sensitivity(df: pd.DataFrame, out_dir: str, dpi: int):
    if df.empty:
        return
    if "block" not in df.columns or "condition" not in df.columns:
        return
    pivot = df.pivot_table(
        index="block",
        columns="condition",
        values="acc_drop_vs_base",
        aggfunc="mean",
    ).sort_index()
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    xs = pivot.index.to_numpy()
    cols = list(pivot.columns)
    for i, c in enumerate(cols):
        ax.bar(xs + (i - (len(cols) - 1) / 2) * width, pivot[c].values, width=width, label=c)
    ax.set_title("T-sensitivity (accuracy drop vs base)")
    ax.set_xlabel("Block")
    ax.set_ylabel("Acc drop")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "t_sensitivity_acc_drop.png"), dpi=dpi)
    plt.close(fig)


def plot_counterfactual(df: pd.DataFrame, out_dir: str, dpi: int):
    if df.empty:
        return
    if "block" not in df.columns or "counterfactual_mode" not in df.columns:
        return
    pivot = df.pivot_table(
        index="block",
        columns="counterfactual_mode",
        values="acc_drop_vs_base",
        aggfunc="mean",
    ).sort_index()
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    xs = pivot.index.to_numpy()
    cols = list(pivot.columns)
    for i, c in enumerate(cols):
        ax.bar(xs + (i - (len(cols) - 1) / 2) * width, pivot[c].values, width=width, label=c)
    ax.set_title("Counterfactual routing (accuracy drop vs base)")
    ax.set_xlabel("Block")
    ax.set_ylabel("Acc drop")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "counterfactual_acc_drop.png"), dpi=dpi)
    plt.close(fig)


def plot_summary(summary_df: pd.DataFrame, out_dir: str, dpi: int):
    if summary_df.empty:
        return
    row = summary_df.iloc[0].to_dict()
    keys = ["base_acc_exp2", "base_acc_exp3"]
    vals = [row[k] for k in keys if k in row]
    if not vals:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(keys[: len(vals)], vals, color=["#4C72B0", "#55A868"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Baseline accuracies")
    ax.set_ylabel("Accuracy")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_baselines.png"), dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    result_dir = args.result_dir
    plot_dir = os.path.join(result_dir, "plots")
    ensure_dir(plot_dir)

    router_summary = safe_read_csv(os.path.join(result_dir, "router_stats_summary.csv"))
    t_sens = safe_read_csv(os.path.join(result_dir, "t_sensitivity.csv"))
    counter = safe_read_csv(os.path.join(result_dir, "counterfactual_routing.csv"))
    summary = safe_read_csv(os.path.join(result_dir, "summary_all_blocks.csv"))

    plot_router_stats(router_summary, plot_dir, args.dpi)
    plot_t_sensitivity(t_sens, plot_dir, args.dpi)
    plot_counterfactual(counter, plot_dir, args.dpi)
    plot_summary(summary, plot_dir, args.dpi)

    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
