#!/usr/bin/env python3
"""
Quick analysis: compute transaction type frequency shares from a generated
dataset and compare against FRPS-derived targets.
"""

import sys
import pandas as pd

FRPS_SHARES = {
    "payment":    0.912,   # cards + ACH debit + checks
    "transfer":   0.010,   # P2P ~2B of ~208B
    "withdrawal": 0.018,   # ATM 3.7B of 208B
    "deposit":    None,     # not in FRPS
    "salary":     0.020,   # payroll subset of ACH credit
}

def analyse(csv_path):
    df = pd.read_csv(csv_path, usecols=["transaction_type", "amount", "is_fraudulent"])
    # Normalise enum strings like "TransactionType.PAYMENT" -> "payment"
    df["tx_type"] = df["transaction_type"].str.replace("TransactionType.", "", regex=False).str.lower()

    total = len(df)
    legit = df[df["is_fraudulent"] == False]
    fraud = df[df["is_fraudulent"] == True]

    print(f"Total transactions: {total:,}")
    print(f"  Legitimate: {len(legit):,}")
    print(f"  Fraudulent: {len(fraud):,}")
    print(f"  Fraud ratio: {len(fraud)/total:.4%}\n")

    # --- Frequency shares (all transactions) ---
    counts = df["tx_type"].value_counts()
    print("=" * 65)
    print(f"{'Type':<14} {'Count':>10} {'Our Share':>10} {'FRPS Share':>11} {'Ratio':>8}")
    print("-" * 65)
    for tx_type in sorted(counts.index):
        n = counts[tx_type]
        share = n / total
        frps = FRPS_SHARES.get(tx_type)
        frps_str = f"{frps:.1%}" if frps is not None else "---"
        ratio_str = f"{share/frps:.1f}x" if frps is not None else "---"
        print(f"{tx_type:<14} {n:>10,} {share:>10.1%} {frps_str:>11} {ratio_str:>8}")
    print("-" * 65)
    print(f"{'TOTAL':<14} {total:>10,} {'100.0%':>10}\n")

    # --- Amount stats per type ---
    print("=" * 65)
    print(f"{'Type':<14} {'Median':>10} {'Mean':>10} {'Min':>10} {'Max':>12}")
    print("-" * 65)
    for tx_type in sorted(counts.index):
        subset = df.loc[df["tx_type"] == tx_type, "amount"]
        print(f"{tx_type:<14} ${subset.median():>9,.0f} ${subset.mean():>9,.0f} "
              f"${subset.min():>9,.0f} ${subset.max():>11,.0f}")
    print("-" * 65)

    # --- Legit-only frequency shares ---
    print(f"\nLegitimate-only shares:")
    legit_counts = legit["tx_type"].value_counts()
    legit_total = len(legit)
    print(f"{'Type':<14} {'Count':>10} {'Share':>10}")
    print("-" * 40)
    for tx_type in sorted(legit_counts.index):
        n = legit_counts[tx_type]
        print(f"{tx_type:<14} {n:>10,} {n/legit_total:>10.1%}")
    print("-" * 40)
    print(f"{'TOTAL':<14} {legit_total:>10,}\n")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "generated_transactions.csv"
    analyse(csv_path)
