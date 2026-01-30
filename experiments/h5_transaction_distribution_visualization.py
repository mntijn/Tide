#!/usr/bin/env python3
"""
H5: Transaction Distribution Visualization Experiment

Visualizes distributions of transaction types and amounts from a transactions
CSV file. Plots follow Edward Tufte's principles and ACM single-column sizing.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Style: ACM single-column + Tufte principles
# ---------------------------------------------------------------------------
def setup_style():
    """Configure matplotlib: ACM dimensions, Tufte minimalism."""
    col_width = 3.33  # ACM single-column inches
    golden = (1 + np.sqrt(5)) / 2

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.figsize": (col_width, col_width / golden),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Tufte: minimal non-data ink
        "axes.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    return col_width, col_width / golden


# Muted palette — prints well in greyscale, distinguishable for colour-blind
_BLUE = "#4878CF"
_GREY = "#555555"


def _save(fig, output_dir, name):
    """Save figure as PNG, then close."""
    path = os.path.join(output_dir, name + ".png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  → {path}")


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
def _dollar_fmt(x, _pos):
    if x >= 1_000_000:
        return f"${x / 1e6:.0f}M"
    if x >= 1_000:
        return f"${x / 1e3:.0f}K"
    return f"${x:.0f}"


def _count_fmt(x, _pos):
    if x >= 1_000_000:
        return f"{x / 1e6:.0f}M"
    if x >= 1_000:
        return f"{x / 1e3:.0f}K"
    if x >= 1:
        return f"{x:.0f}"
    return ""


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def visualize_transaction_distributions(transactions_csv_path, output_dir):
    print(f"Reading {transactions_csv_path}")
    try:
        df = pd.read_csv(transactions_csv_path)
    except FileNotFoundError:
        print(f"ERROR: file not found: {transactions_csv_path}")
        return

    if df.empty:
        print("WARNING: empty file, skipping.")
        return

    os.makedirs(output_dir, exist_ok=True)
    setup_style()

    # ── Plot 1: Transaction type counts (dot plot — Tufte preferred) ──────
    print("Plot 1: transaction type counts")
    type_counts = df["transaction_type"].value_counts().sort_values()

    fig, ax = plt.subplots()
    y_pos = np.arange(len(type_counts))
    ax.scatter(type_counts.values, y_pos, color=_BLUE, s=18, zorder=5,
               edgecolors="none")
    # Thin reference lines connecting label to dot (Tufte range frame idea)
    for y, val in zip(y_pos, type_counts.values):
        ax.plot([0, val], [y, y], color="#cccccc", linewidth=0.3, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(type_counts.index, fontsize=6.5)
    ax.set_xscale("log")
    ax.set_xlabel("Count (log scale)")
    # Place count labels to the right of dots, outside the data area
    x_max = type_counts.values.max()
    for y, val in zip(y_pos, type_counts.values):
        ax.text(val * 1.25, y, f"{val:,.0f}", va="center", fontsize=5.5,
                color=_GREY)
    ax.set_xlim(right=x_max * 3)  # room for labels
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)  # no y-tick marks
    fig.tight_layout()
    _save(fig, output_dir, "h5_transaction_type_distribution")

    # ── Plot 2: Amount distribution (log-log histogram) ───────────────────
    print("Plot 2: amount distribution")
    amounts = df["amount"].dropna()
    amounts = amounts[amounts > 0]

    log_bins = np.logspace(
        np.log10(max(1.0, amounts.min())),
        np.log10(amounts.max()),
        num=60,
    )

    fig, ax = plt.subplots()
    counts, edges, _ = ax.hist(
        amounts, bins=log_bins,
        color=_BLUE, edgecolor="white", linewidth=0.15,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Transaction amount (USD)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_dollar_fmt))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_count_fmt))

    # Reporting threshold — placed at top of plot outside data region
    threshold = 10_000
    ax.axvline(threshold, color="#999999", linewidth=0.4, linestyle=":",
               zorder=2)
    ax.text(threshold, ax.get_ylim()[1] * 1.4, "CTR\nthreshold",
            fontsize=5, color="#777777", ha="center", va="bottom")

    # Median — annotate in margin above the plot
    median_val = amounts.median()
    ax.axvline(median_val, color="#D95F02", linewidth=0.5, linestyle="--",
               alpha=0.6, zorder=3)
    ax.text(median_val, ax.get_ylim()[1] * 1.4,
            f"median\n${median_val:,.0f}",
            fontsize=5, color="#D95F02", ha="center", va="bottom")

    ax.set_clip_on(False)  # allow text above axes
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)  # room for annotations above
    _save(fig, output_dir, "h5_transaction_amount_distribution")

    # ── Plot 3: Amount by type (small multiples) ────────────────────────
    print("Plot 3: amount by type")
    tx_types = sorted(df["transaction_type"].unique())
    tx_types = [t for t in tx_types
                if (df.loc[df["transaction_type"] == t, "amount"] > 0).sum() >= 10]
    n = len(tx_types)
    if n == 0:
        print("  skipped (not enough types)")
        return

    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.33, 1.6 * nrows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    for idx, tx_type in enumerate(tx_types):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        subset = df.loc[df["transaction_type"] == tx_type, "amount"]
        subset = subset[subset > 0]

        ax.hist(subset, bins=log_bins, color=_BLUE,
                edgecolor="white", linewidth=0.15)
        ax.set_xscale("log")
        ax.set_yscale("log")

        # Panel title: clean, top-right, out of the data
        label = tx_type.replace("_", " ").title()
        ax.set_title(label, fontsize=7, loc="left", pad=3, color=_GREY)

        # Only label outer axes (Tufte: no redundant labels)
        if r == nrows - 1 or (r == nrows - 2 and c >= n - (nrows - 1) * ncols):
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(_dollar_fmt))
            ax.set_xlabel("")
        else:
            ax.xaxis.set_major_formatter(ticker.NullFormatter())

        if c == 0:
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(_count_fmt))
        else:
            ax.yaxis.set_major_formatter(ticker.NullFormatter())

        # Reduce tick clutter on log axes
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))

        # Median line per type
        med = subset.median()
        ax.axvline(med, color="#D95F02", linewidth=0.4, linestyle="--",
                   alpha=0.5, zorder=3)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.supxlabel("Transaction amount (USD)", fontsize=8, y=0.02)
    fig.supylabel("Frequency", fontsize=8, x=0.02)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    _save(fig, output_dir, "h5_amount_by_type")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_h5_experiment(transactions_filepath, output_plot_dir):
    print("=== H5: Transaction Distribution Visualization ===")
    visualize_transaction_distributions(transactions_filepath, output_plot_dir)
    print("=== H5 Experiment Finished ===")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H5: Transaction Distribution Visualization")
    parser.add_argument("-i", "--input-file",
                        default="generated_transactions.csv",
                        help="Path to the transactions CSV file.")
    parser.add_argument("-o", "--output-dir", default="plots",
                        help="Directory to save the plots.")
    args = parser.parse_args()
    success = run_h5_experiment(args.input_file, args.output_dir)
    sys.exit(0 if success else 1)
