#!/usr/bin/env python3
"""
H5: Transaction Distribution Visualization Experiment

This experiment visualizes the distributions of transaction types and amounts
from a given transactions file to analyze the characteristics of financial data.
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_transaction_distributions(transactions_csv_path, output_dir):
    """
    Reads transaction data and generates plots for type and amount distributions.
    """
    print(f"Reading transactions from: {transactions_csv_path}")
    try:
        df = pd.read_csv(transactions_csv_path)
    except FileNotFoundError:
        print(f"ERROR: Transactions file not found at {transactions_csv_path}")
        return

    if df.empty:
        print("WARNING: Transactions file is empty. Skipping visualization.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Apply Visualization Guidelines ---
    sns.set_context("notebook")
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    golden_ratio = 1.618
    figure_size = (8, 8 / golden_ratio)
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"

    # --- Plot 1: Transaction Type Distribution ---
    print("Generating transaction type distribution plot...")
    plt.figure(figsize=figure_size)
    ax1 = sns.countplot(y=df['transaction_type'],
                        order=df['transaction_type'].value_counts().index)
    ax1.set_title('Distribution of Transaction Types', fontsize=16)
    ax1.set_xlabel('Count', fontsize=12)
    ax1.set_ylabel('Transaction Type', fontsize=12)
    ax1.set_xscale('log')
    sns.despine()
    plt.tight_layout()
    type_plot_path = os.path.join(
        output_dir, 'h5_transaction_type_distribution.png')
    plt.savefig(type_plot_path)
    plt.close()
    print(f"Saved transaction type plot to: {type_plot_path}")

    # --- Plot 2: Transaction Amount Distribution ---
    print("Generating transaction amount distribution plot...")
    plt.figure(figsize=figure_size)
    ax2 = sns.histplot(df['amount'], bins=50, kde=True, log_scale=True)
    ax2.set_title(
        'Distribution of Transaction Amounts (Log Scale)', fontsize=16)
    ax2.set_xlabel('Amount (Log Scale)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    sns.despine()
    plt.tight_layout()
    amount_plot_path = os.path.join(
        output_dir, 'h5_transaction_amount_distribution.png')
    plt.savefig(amount_plot_path)
    plt.close()
    print(f"Saved transaction amount plot to: {amount_plot_path}")


def run_h5_experiment(transactions_filepath, output_plot_dir):
    """Run the H5 transaction distribution visualization experiment"""
    print("=== H5: Transaction Distribution Visualization ===")

    # Run visualization
    visualize_transaction_distributions(
        transactions_filepath, output_plot_dir)

    print("\n=== H5 Experiment Finished ===")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H5: Transaction Distribution Visualization Experiment")
    parser.add_argument(
        "-i", "--input-file",
        default="generated_transactions.csv",
        help="Path to the transactions CSV file."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="plots",
        help="Directory to save the plots."
    )
    args = parser.parse_args()

    success = run_h5_experiment(args.input_file, args.output_dir)
    sys.exit(0 if success else 1)
