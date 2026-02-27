#!/usr/bin/env python3
"""
Dataset Statistics Script for AMLbench

Computes comprehensive statistics for a generated dataset,
optimized for large datasets using vectorized operations and parallel processing.
"""

import os
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple


def load_data(nodes_path: str, transactions_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load nodes and transactions CSVs in parallel."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        nodes_future = executor.submit(pd.read_csv, nodes_path)
        tx_future = executor.submit(pd.read_csv, transactions_path)
        nodes_df = nodes_future.result()
        transactions_df = tx_future.result()
    return nodes_df, transactions_df


def compute_basic_stats(nodes_df: pd.DataFrame, tx_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic graph statistics (vectorized)."""
    active_nodes = pd.concat([tx_df['src'], tx_df['dest']]).nunique()
    return {
        'num_nodes_total': len(nodes_df),
        'num_nodes_active': active_nodes,
        'num_edges': len(tx_df),
    }


def compute_node_type_stats(nodes_df: pd.DataFrame) -> Dict[str, int]:
    """Count nodes by type."""
    return nodes_df['node_type'].value_counts().to_dict()


def compute_fraud_stats(nodes_df: pd.DataFrame, tx_df: pd.DataFrame) -> Dict[str, float]:
    """Compute fraud-related statistics (vectorized)."""
    fraud_nodes = nodes_df['is_fraudulent'].sum()
    total_nodes = len(nodes_df)
    fraud_edges = tx_df['is_fraudulent'].sum()
    total_edges = len(tx_df)
    legit_edges = total_edges - fraud_edges

    return {
        'fraud_node_count': int(fraud_nodes),
        'fraud_node_ratio': fraud_nodes / total_nodes if total_nodes > 0 else 0.0,
        'fraud_edge_count': int(fraud_edges),
        'fraud_edge_ratio': fraud_edges / total_edges if total_edges > 0 else 0.0,
        'class_imbalance_ratio': legit_edges / fraud_edges if fraud_edges > 0 else float('inf'),
    }


def compute_node_label_homophily(nodes_df: pd.DataFrame, tx_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute node-label homophily (vectorized).
    For each edge: do the two NODES it connects share the same fraud label?
    """
    # Create fraud lookup as a set for O(1) membership testing
    fraud_set = set(nodes_df.loc[nodes_df['is_fraudulent'], 'node_id'])

    # Vectorized: check if src/dest are in fraud_set
    src_fraud = tx_df['src'].isin(fraud_set).values
    dest_fraud = tx_df['dest'].isin(fraud_set).values

    # Same class = both fraud or both not fraud
    same_class = (src_fraud == dest_fraud).sum()
    total_tx = len(tx_df)

    # Fraud incident = at least one endpoint is fraud
    fraud_incident_mask = src_fraud | dest_fraud
    fraud_incident = fraud_incident_mask.sum()

    # Both fraud
    both_fraud = (src_fraud & dest_fraud).sum()

    return {
        'node_label_homophily': same_class / total_tx if total_tx > 0 else 0.0,
        'fraud_node_homophily': both_fraud / fraud_incident if fraud_incident > 0 else 0.0,
    }


def compute_edge_label_homophily(tx_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute edge-label homophily (vectorized with numpy).
    For each edge: what fraction of NEIGHBORING EDGES share the same label?
    """
    labels = tx_df['is_fraudulent'].astype(int).values
    src = tx_df['src'].values
    dest = tx_df['dest'].values

    # Build node -> (count_class_0, count_class_1, total_degree) using pandas groupby
    # This is much faster than Python dicts for large datasets
    edge_df = pd.DataFrame({'node': np.concatenate([src, dest]),
                            'label': np.concatenate([labels, labels])})

    # Count per node per class
    node_stats = edge_df.groupby(['node', 'label']).size().unstack(fill_value=0)
    if 0 not in node_stats.columns:
        node_stats[0] = 0
    if 1 not in node_stats.columns:
        node_stats[1] = 0
    node_stats['degree'] = node_stats[0] + node_stats[1]

    # Map back to edges
    src_class0 = node_stats[0].reindex(src, fill_value=0).values
    src_class1 = node_stats[1].reindex(src, fill_value=0).values
    src_deg = node_stats['degree'].reindex(src, fill_value=0).values

    dest_class0 = node_stats[0].reindex(dest, fill_value=0).values
    dest_class1 = node_stats[1].reindex(dest, fill_value=0).values
    dest_deg = node_stats['degree'].reindex(dest, fill_value=0).values

    # For each edge, compute same-class neighbors and total neighbors
    # same = node_class_counts[src][label] + node_class_counts[dest][label] - 2
    # total = degree[src] + degree[dest] - 2
    same_neighbors = np.where(
        labels == 0,
        src_class0 + dest_class0 - 2,
        src_class1 + dest_class1 - 2
    )
    total_neighbors = src_deg + dest_deg - 2

    # Aggregate by class
    class_same = {0: 0.0, 1: 0.0}
    class_total = {0: 0.0, 1: 0.0}

    mask_0 = labels == 0
    mask_1 = labels == 1

    class_same[0] = same_neighbors[mask_0].sum()
    class_same[1] = same_neighbors[mask_1].sum()
    class_total[0] = total_neighbors[mask_0].sum()
    class_total[1] = total_neighbors[mask_1].sum()

    result = {}
    for c in [0, 1]:
        result[f'homophily_class_{c}'] = (
            class_same[c] / class_total[c] if class_total[c] > 0 else 0.0
        )

    total_same = class_same[0] + class_same[1]
    total_all = class_total[0] + class_total[1]
    result['edge_label_homophily'] = total_same / total_all if total_all > 0 else 0.0

    return result


def compute_degree_stats(tx_df: pd.DataFrame) -> Dict[str, float]:
    """Compute degree statistics (vectorized)."""
    out_deg = tx_df['src'].value_counts()
    in_deg = tx_df['dest'].value_counts()

    # Combine into total degree
    all_nodes = pd.Index(tx_df['src']).union(pd.Index(tx_df['dest']))
    total_deg = out_deg.reindex(all_nodes, fill_value=0) + in_deg.reindex(all_nodes, fill_value=0)

    return {
        'degree_mean': total_deg.mean(),
        'degree_std': total_deg.std(),
        'degree_min': int(total_deg.min()),
        'degree_max': int(total_deg.max()),
        'degree_median': total_deg.median(),
        'out_degree_mean': out_deg.mean(),
        'in_degree_mean': in_deg.mean(),
    }


def compute_transaction_type_stats(tx_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute transaction type distribution (vectorized)."""
    type_counts = tx_df['transaction_type'].value_counts().to_dict()

    # Fraud rate per type using groupby
    fraud_agg = tx_df.groupby('transaction_type')['is_fraudulent'].agg(['sum', 'count'])
    fraud_rate_by_type = (fraud_agg['sum'] / fraud_agg['count']).to_dict()

    return {
        'transaction_type_counts': type_counts,
        'fraud_rate_by_type': fraud_rate_by_type,
    }


def compute_amount_stats(tx_df: pd.DataFrame) -> Dict[str, float]:
    """Compute transaction amount statistics (vectorized)."""
    amounts = tx_df['amount']

    stats = {
        'amount_mean': amounts.mean(),
        'amount_std': amounts.std(),
        'amount_min': amounts.min(),
        'amount_max': amounts.max(),
        'amount_median': amounts.median(),
        'amount_25th_percentile': amounts.quantile(0.25),
        'amount_75th_percentile': amounts.quantile(0.75),
        'amount_total': amounts.sum(),
    }

    # Fraud vs legit using boolean indexing
    fraud_mask = tx_df['is_fraudulent']
    if fraud_mask.any():
        stats['fraud_amount_mean'] = amounts[fraud_mask].mean()
        stats['fraud_amount_median'] = amounts[fraud_mask].median()
    if (~fraud_mask).any():
        stats['legit_amount_mean'] = amounts[~fraud_mask].mean()
        stats['legit_amount_median'] = amounts[~fraud_mask].median()

    return stats


def compute_temporal_stats(tx_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute temporal statistics."""
    timestamps = pd.to_datetime(tx_df['timestamp'], format='mixed')
    return {
        'time_span_start': str(timestamps.min()),
        'time_span_end': str(timestamps.max()),
        'time_span_days': (timestamps.max() - timestamps.min()).days,
    }


def compute_currency_stats(tx_df: pd.DataFrame) -> Dict[str, int]:
    """Compute currency distribution."""
    return tx_df['currency'].value_counts().to_dict()


def compute_country_stats(nodes_df: pd.DataFrame) -> Dict[str, int]:
    """Compute country distribution (top 20)."""
    return nodes_df['country_code'].value_counts().head(20).to_dict()


def compute_all_statistics(nodes_df: pd.DataFrame, tx_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all statistics in parallel where possible."""

    # Group 1: Stats that only need nodes_df
    # Group 2: Stats that only need tx_df
    # Group 3: Stats that need both

    results = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(compute_basic_stats, nodes_df, tx_df): 'basic',
            executor.submit(compute_node_type_stats, nodes_df): 'node_types',
            executor.submit(compute_fraud_stats, nodes_df, tx_df): 'fraud',
            executor.submit(compute_node_label_homophily, nodes_df, tx_df): 'node_homophily',
            executor.submit(compute_edge_label_homophily, tx_df): 'edge_homophily',
            executor.submit(compute_degree_stats, tx_df): 'degree',
            executor.submit(compute_transaction_type_stats, tx_df): 'transaction_types',
            executor.submit(compute_amount_stats, tx_df): 'amount',
            executor.submit(compute_temporal_stats, tx_df): 'temporal',
            executor.submit(compute_currency_stats, tx_df): 'currency',
            executor.submit(compute_country_stats, nodes_df): 'country',
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                print(f"Error computing {key}: {e}")
                results[key] = {}

    return results


def print_text_report(stats: Dict[str, Any]):
    """Print human-readable text report."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS REPORT")
    print("=" * 60)

    print("\n--- Basic Graph Statistics ---")
    b = stats['basic']
    print(f"  Total nodes:           {b['num_nodes_total']:,}")
    print(f"  Active nodes:          {b['num_nodes_active']:,}")
    print(f"  Edges (transactions):  {b['num_edges']:,}")

    print("\n--- Node Type Distribution ---")
    for node_type, count in stats['node_types'].items():
        print(f"  {node_type}: {count:,}")

    print("\n--- Fraud Statistics ---")
    f = stats['fraud']
    print(f"  Fraudulent nodes:      {f['fraud_node_count']:,} ({f['fraud_node_ratio']:.4%})")
    print(f"  Fraudulent edges:      {f['fraud_edge_count']:,} ({f['fraud_edge_ratio']:.4%})")
    print(f"  Class imbalance ratio: {f['class_imbalance_ratio']:.2f}:1 (legit:fraud)")

    print("\n--- Homophily Indices ---")
    nh = stats['node_homophily']
    eh = stats['edge_homophily']
    print("  Node-label homophily (do connected nodes share labels?):")
    print(f"    Overall:             {nh['node_label_homophily']:.6f}")
    print(f"    Fraud-class:         {nh['fraud_node_homophily']:.6f}")
    print("  Edge-label homophily (do neighboring edges share labels?):")
    print(f"    Overall:             {eh['edge_label_homophily']:.6f}")
    print(f"    Class 0 (legit):     {eh.get('homophily_class_0', 0):.6f}")
    print(f"    Class 1 (fraud):     {eh.get('homophily_class_1', 0):.6f}")

    print("\n--- Degree Statistics ---")
    d = stats['degree']
    print(f"  Mean degree:           {d['degree_mean']:.2f}")
    print(f"  Std degree:            {d['degree_std']:.2f}")
    print(f"  Min degree:            {d['degree_min']}")
    print(f"  Max degree:            {d['degree_max']}")
    print(f"  Median degree:         {d['degree_median']:.1f}")

    print("\n--- Transaction Amount Statistics ---")
    a = stats['amount']
    print(f"  Mean:                  ${a['amount_mean']:,.2f}")
    print(f"  Std:                   ${a['amount_std']:,.2f}")
    print(f"  Min:                   ${a['amount_min']:,.2f}")
    print(f"  Max:                   ${a['amount_max']:,.2f}")
    print(f"  Median:                ${a['amount_median']:,.2f}")
    print(f"  Total volume:          ${a['amount_total']:,.2f}")
    if 'fraud_amount_mean' in a:
        print(f"  Fraud mean:            ${a['fraud_amount_mean']:,.2f}")
        print(f"  Legit mean:            ${a['legit_amount_mean']:,.2f}")

    print("\n--- Transaction Type Distribution ---")
    tt = stats['transaction_types']
    for tx_type, count in tt['transaction_type_counts'].items():
        fraud_rate = tt['fraud_rate_by_type'].get(tx_type, 0)
        print(f"  {tx_type}: {count:,} ({fraud_rate:.2%} fraud)")

    print("\n--- Temporal Statistics ---")
    t = stats.get('temporal', {})
    print(f"  Start date:            {t.get('time_span_start', 'N/A')}")
    print(f"  End date:              {t.get('time_span_end', 'N/A')}")
    print(f"  Time span:             {t.get('time_span_days', 'N/A')} days")

    print("\n--- Currency Distribution ---")
    for currency, count in stats['currency'].items():
        print(f"  {currency}: {count:,}")

    print("\n--- Country Distribution (Top 20) ---")
    for country, count in list(stats['country'].items())[:20]:
        print(f"  {country}: {count:,}")

    print("\n" + "=" * 60)


def print_latex_table(stats: Dict[str, Any]):
    """Print LaTeX table for publication."""
    b = stats.get('basic', {})
    f = stats.get('fraud', {})
    eh = stats.get('edge_homophily', {})
    d = stats.get('degree', {})
    t = stats.get('temporal', {})

    print("\n% LaTeX table for dataset statistics")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Dataset Statistics}")
    print("\\label{tab:dataset_stats}")
    print("\\begin{tabular}{lr}")
    print("\\toprule")
    print("\\textbf{Metric} & \\textbf{Value} \\\\")
    print("\\midrule")
    print(f"Nodes & {b.get('num_nodes_active', 0):,} \\\\")
    print(f"Edges & {b.get('num_edges', 0):,} \\\\")
    print(f"Fraud ratio (edges) & {f.get('fraud_edge_ratio', 0):.4f} \\\\")
    print(f"Class imbalance & {f.get('class_imbalance_ratio', 0):.1f}:1 \\\\")
    print(f"Edge-label homophily & {eh.get('edge_label_homophily', 0):.4f} \\\\")
    print(f"Homophily (class 0) & {eh.get('homophily_class_0', 0):.4f} \\\\")
    print(f"Homophily (class 1) & {eh.get('homophily_class_1', 0):.4f} \\\\")
    print(f"Mean degree & {d.get('degree_mean', 0):.2f} \\\\")
    print(f"Time span (days) & {t.get('time_span_days', 'N/A')} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics for AMLbench")
    parser.add_argument("-d", "--data-dir", required=True,
                        help="Directory containing the CSV files")
    parser.add_argument("-n", "--nodes-file", default="generated_nodes.csv",
                        help="Nodes CSV filename")
    parser.add_argument("-t", "--transactions-file", default="generated_transactions.csv",
                        help="Transactions CSV filename")
    parser.add_argument("-o", "--output-format", choices=['text', 'latex'], default='text',
                        help="Output format (text or latex)")
    args = parser.parse_args()

    nodes_path = os.path.join(args.data_dir, args.nodes_file)
    transactions_path = os.path.join(args.data_dir, args.transactions_file)

    print(f"Loading data from {args.data_dir}...")
    nodes_df, tx_df = load_data(nodes_path, transactions_path)
    print(f"  Loaded {len(nodes_df):,} nodes and {len(tx_df):,} transactions")

    print("Computing statistics (parallel)...")
    all_stats = compute_all_statistics(nodes_df, tx_df)

    if args.output_format == 'latex':
        print_latex_table(all_stats)
    else:
        print_text_report(all_stats)

    return all_stats


if __name__ == "__main__":
    main()
