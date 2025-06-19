#!/usr/bin/env python3
"""
H3: Statistical Validation Experiment

Tests that the synthetic data generator consistently produces datasets
that match their specified configuration parameters. This experiment
runs the generator multiple times with identical configurations but
different random seeds to evaluate both accuracy and consistency.

Metrics collected over 10 runs:
- Actual individual and business node counts
- Measured transaction rate (average across all accounts)
- Percentage of background transactions outside specified amount ranges
- Actual count of generated fraudulent patterns
"""

import os
import sys
import csv
import json
import yaml
import tempfile
import datetime
import subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt

# Add the parent directory to path to import main
sys.path.append(str(Path(__file__).parent.parent))

# Constants for the three compliance-testing configurations
CONFIGURATIONS = [
    {
        'name': '1',  # Configuration label used in output and file names
        'individuals': 1000,
        'business_probability': 0.10,
        'transaction_rate': 0.5,
        'amount_range': [10.0, 500.0],
        'target_fraud_patterns': 25
    },
    {
        'name': '2',
        'individuals': 1500,
        'business_probability': 0.15,
        'transaction_rate': 1.0,
        'amount_range': [10.0, 500.0],
        'target_fraud_patterns': 30
    },
    {
        'name': '3',
        'individuals': 2000,
        'business_probability': 0.20,
        'transaction_rate': 1.5,
        'amount_range': [10.0, 500.0],
        'target_fraud_patterns': 35
    }
]


def _generate_pattern_frequency(total_patterns):
    """Return an integer distribution of pattern types that sums to ``total_patterns``.

    The proportions are based on the original 33-pattern breakdown (9/13/11)."""
    base_distribution = {
        'RapidFundMovement': 9,
        'FrontBusinessActivity': 13,
        'RepeatedOverseasTransfers': 11,
    }
    ratio_sum = sum(base_distribution.values())
    pattern_frequency = {}
    allocated = 0
    # Distribute patterns proportionally (rounded) and fix rounding issues at the end
    for i, (name, count) in enumerate(base_distribution.items(), 1):
        if i == len(base_distribution):
            pattern_frequency[name] = total_patterns - allocated
        else:
            share = round(total_patterns * (count / ratio_sum))
            pattern_frequency[name] = share
            allocated += share
    return pattern_frequency


def create_h3_statistical_config(base_params, seed):
    """Create configuration for the H3 statistical validation test."""
    # Build a configuration dictionary based on the selected base parameters
    amount_range = base_params['amount_range']
    pattern_frequency = _generate_pattern_frequency(
        base_params['target_fraud_patterns'])

    return {
        'graph_scale': {
            'individuals': base_params['individuals'],
            'institutions_per_country': 3,
            'individual_accounts_per_institution_range': [1, 3],
            'business_accounts_per_institution_range': [1, 6]
        },
        'transaction_rates': {
            'per_account_per_day': base_params['transaction_rate']
        },
        'time_span': {
            'start_date': '2023-01-01T00:00:00',
            'end_date': '2023-05-15T23:59:59'
        },
        'account_balance_range_normal': [1000.0, 50000.0],
        'backgroundPatterns': {
            'randomPayments': {
                'legit_rate_multiplier': [0.3, 0.8],
                'transaction_type_probabilities': {
                    'transfer': 0.4,
                    'payment': 0.3,
                    'deposit': 0.15,
                    'withdrawal': 0.15
                },
                'amount_ranges': {
                    'payment': amount_range,
                    'transfer': amount_range,
                    'cash_operations': amount_range
                }
            },
            'salaryPayments': {
                'payment_intervals': [14, 30],
                'salary_range': [2500.0, 7500.0],
                'salary_variation': 0.05,
                'preferred_payment_days': [1, 15, 30]
            }
        },
        'company_size_range': [1, 1000],
        'salary_payment_days': [24, 30, 1],
        'salary_amount_range': [2500.0, 7500.0],
        'random_seed': seed,
        'business_creation_date_range': [90, 5475],
        'random_business_probability': base_params['business_probability'],
        'pattern_frequency': {
            'random': False,
            **pattern_frequency
        },
        'reporting_threshold': 10000,
        'reporting_currency': 'EUR',
        'high_risk_business_probability': 0.05,
        'offshore_account_probability': 0.1,
        'amount_distribution_params': {
            'mean': 100,
            'variance': 50
        },
        'high_transaction_amount_ratio': 0.05,
        'low_transaction_amount_ratio': 0.1,
        'account_categories': ["Current", "Savings", "Loan"],
        'high_risk_config': {
            'countries_weight_factor': 1.0,
            'business_categories_weight_factor': 1.0,
            'company_size_thresholds': {
                'very_small_max': 5
            }
        },
        'risk_weights': {
            'base_individual': 0.05,
            'base_business': 0.10,
            'age_group': 0.15,
            'occupation': 0.12,
            'country': 0.2,
            'business_category': 0.25,
            'very_small_company': 0.10,
            'max_score': 1.0
        },
        'fraud_selection_config': {
            'min_risk_score_for_fraud_consideration': 0.50,
            'base_fraud_probability_if_considered': 0.10
        }
    }


def analyze_nodes(nodes_file):
    """Analyzes node counts from the generated nodes file."""
    results = {
        'individuals': 0,
        'businesses': 0,
        'issues': []
    }

    if not os.path.exists(nodes_file):
        results['issues'].append(f"Nodes file not found: {nodes_file}")
        return results

    try:
        with open(nodes_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_type = row.get('node_type', '')
                if node_type in ['NodeType.INDIVIDUAL', 'individual']:
                    results['individuals'] += 1
                elif node_type in ['NodeType.BUSINESS', 'business']:
                    results['businesses'] += 1
    except Exception as e:
        results['issues'].append(f"Error reading nodes file: {e}")

    return results


def analyze_transactions(config, edges_file, patterns_file):
    """Analyzes transaction rates and total transaction count."""
    results = {
        'rate': 0.0,
        'total_transactions': 0,
        'issues': []
    }

    if not os.path.exists(edges_file):
        results['issues'].append(f"Edges file not found: {edges_file}")
        return results

    pattern_transactions = set()
    if os.path.exists(patterns_file):
        try:
            with open(patterns_file, 'r') as f:
                data = json.load(f)
                for pattern in data.get('patterns', []):
                    for tx in pattern.get('transactions', []):
                        tx_id = tx.get('transaction_id')
                        if tx_id:
                            pattern_transactions.add(tx_id)
        except Exception as e:
            results['issues'].append(f"Error reading patterns file: {e}")

    source_account_transactions = defaultdict(int)
    total_transactions_count = 0

    start_date = datetime.datetime.fromisoformat(
        config['time_span']['start_date'])
    end_date = datetime.datetime.fromisoformat(config['time_span']['end_date'])
    total_days = (end_date - start_date).days

    try:
        with open(edges_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                edge_type = row.get('edge_type', '')
                if edge_type not in ['EdgeType.TRANSACTION', 'transaction']:
                    continue

                total_transactions_count += 1

                if row.get('transaction_id') in pattern_transactions:
                    continue

                source_account = row.get('src', '')
                if source_account:
                    source_account_transactions[source_account] += 1

    except Exception as e:
        results['issues'].append(f"Error reading edges file: {e}")
        return results

    results['total_transactions'] = total_transactions_count

    if source_account_transactions:
        total_bg_transactions = sum(source_account_transactions.values())
        num_active_accounts = len(source_account_transactions)
        actual_rate = total_bg_transactions / \
            (num_active_accounts * total_days) if num_active_accounts > 0 else 0
        results['rate'] = actual_rate
    else:
        results['issues'].append("No background transactions found")

    return results


def analyze_patterns(patterns_file):
    """Analyzes the number of generated fraudulent patterns."""
    results = {'pattern_count': 0, 'issues': []}
    if not os.path.exists(patterns_file):
        results['issues'].append(f"Patterns file not found: {patterns_file}")
        return results
    try:
        with open(patterns_file, 'r') as f:
            data = json.load(f)
            results['pattern_count'] = len(data.get('patterns', []))
    except Exception as e:
        results['issues'].append(f"Error reading patterns file: {e}")
    return results


def run_single_simulation(base_params, seed, temp_dir):
    """Runs a single data generation and analysis, and cleans up files."""
    cfg_label = base_params['name']
    print(f"--- Running simulation for config {cfg_label} | seed {seed} ---")
    config = create_h3_statistical_config(base_params, seed)

    prefix = f"cfg{cfg_label}_seed{seed}"
    config_file = os.path.join(temp_dir, f"h3_{prefix}.yaml")
    nodes_file = os.path.join(temp_dir, f"nodes_{prefix}.csv")
    edges_file = os.path.join(temp_dir, f"edges_{prefix}.csv")
    patterns_file = os.path.join(temp_dir, f"patterns_{prefix}.json")
    files_to_delete = [config_file, nodes_file, edges_file, patterns_file]

    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        main_script = str(Path(__file__).parent.parent / "main.py")
        cmd = [
            sys.executable, main_script,
            '--config', config_file,
            '--output-dir', temp_dir,
            '--nodes-file', os.path.basename(nodes_file),
            '--edges-file', os.path.basename(edges_file),
            '--patterns-file', os.path.basename(patterns_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"ERROR: Generation failed for config {cfg_label} | seed {seed} with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return {'issues': [f'Generation failed for config {cfg_label} | seed {seed}']}

        # Analyze generated files
        node_results = analyze_nodes(nodes_file)
        tx_results = analyze_transactions(config, edges_file, patterns_file)
        pattern_results = analyze_patterns(patterns_file)

        issues = node_results['issues'] + \
            tx_results['issues'] + pattern_results['issues']
        if issues:
            print(
                f"WARNING: Analysis issues for config {cfg_label} | seed {seed}: {issues}")

        return {
            'seed': seed,
            'individual_nodes': node_results['individuals'],
            'business_nodes': node_results['businesses'],
            'transaction_rate': tx_results['rate'],
            'total_transactions': tx_results['total_transactions'],
            'pattern_count': pattern_results['pattern_count'],
            'issues': issues
        }
    finally:
        # Clean up generated files to conserve memory.
        for f_path in files_to_delete:
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
            except OSError as e:
                print(
                    f"WARNING: Could not delete temporary file {f_path}: {e}")


def plot_compliance_results(per_run_df):
    """Create four compliance plots adhering to user-specified guidelines.

    1. Business nodes (boxplot with target line)
    2. Total transactions (boxplot)
    3. Transaction rate (line plot + target line)
    4. Fraud pattern count (line plot + target line)
    """

    # Set up styling for accessibility and clarity
    sns.set_context("notebook", font_scale=1.8)
    sns.set_style("ticks")
    sns.set_palette("colorblind", color_codes=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    # --- Helper to set y-limits with padding ---
    def set_padded_ylim(ax, values, padding_factor=0.1):
        min_val, max_val = np.min(values), np.max(values)
        padding = (max_val - min_val) * padding_factor
        ax.set_ylim(min_val - padding, max_val + padding)

    # --------------------------------------------------------------------------------
    # 1. Business Nodes boxplot
    ax = axes[0]
    sns.boxplot(data=per_run_df, x='config', y='business_nodes', ax=ax)
    target_lines = [c['individuals'] * c['business_probability']
                    for c in CONFIGURATIONS]
    for target in target_lines:
        ax.axhline(target, linestyle='--', linewidth=1, color='#004D40')
    set_padded_ylim(ax, np.concatenate(
        [per_run_df['business_nodes'], target_lines]))
    ax.set_title('Business Nodes per Configuration')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Business Node Count')
    sns.despine(ax=ax)

    # --------------------------------------------------------------------------------
    # 2. Total Transactions boxplot
    ax = axes[1]
    sns.boxplot(data=per_run_df, x='config', y='total_transactions', ax=ax)
    set_padded_ylim(ax, per_run_df['total_transactions'])
    ax.set_title('Total Transactions per Configuration')
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Total Transactions')
    sns.despine(ax=ax)

    # --------------------------------------------------------------------------------
    # 3. Transaction Rate line plot
    ax = axes[2]
    sns.lineplot(data=per_run_df, x='seed', y='transaction_rate', hue='config',
                 marker='o', ax=ax, legend=False)
    target_lines = [c['transaction_rate'] for c in CONFIGURATIONS]
    for target in target_lines:
        ax.axhline(target, linestyle='--', linewidth=1, color='grey')
    set_padded_ylim(ax, np.concatenate(
        [per_run_df['transaction_rate'], target_lines]))
    ax.set_title('Transaction Rate across Seeds')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Rate (tx/account/day)')
    sns.despine(ax=ax)

    # --------------------------------------------------------------------------------
    # 4. Fraud Pattern Count line plot
    ax = axes[3]
    sns.lineplot(data=per_run_df, x='seed', y='pattern_count', hue='config',
                 marker='s', ax=ax, legend=False)
    target_lines = [c['target_fraud_patterns'] for c in CONFIGURATIONS]
    for target in target_lines:
        ax.axhline(target, linestyle='--', linewidth=1, color='grey')
    set_padded_ylim(ax, np.concatenate(
        [per_run_df['pattern_count'], target_lines]))
    ax.set_title('Fraud Patterns Generated across Seeds')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Pattern Count')
    sns.despine(ax=ax)

    plt.tight_layout()
    plt.show()

    # Save figure
    if not os.path.exists('plots'):
        os.makedirs('plots')
    save_path = 'plots/h3_compliance_metrics.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")


def run_h3_statistical_validation():
    """Execute the H3 configuration compliance test across three configurations."""
    print("=== H3: Configuration Compliance Testing ===\n")

    num_runs = 10
    seeds = range(1, num_runs + 1)

    for cfg_idx, cfg in enumerate(CONFIGURATIONS, 1):
        print(
            f"\n--- Configuration {cfg_idx} (Individuals={cfg['individuals']}, TxRate={cfg['transaction_rate']}, FraudPatterns={cfg['target_fraud_patterns']}) ---")
        print(f"{'Seed':<5} | {'Individuals':<12} | {'Businesses':<12} | {'Tx Rate':<10} | {'Total Tx':<12} | {'Patterns':<10}")
        print("-" * 90)

        all_results = []
        per_run_records = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for seed in seeds:
                run_result = run_single_simulation(cfg, seed, temp_dir)
                if 'Generation failed' in run_result.get('issues', []):
                    print("\nAborting experiment due to critical generation failure.")
                    sys.exit(1)

                all_results.append(run_result)
                # Print per-run results
                print(
                    f"{run_result['seed']:<5} | "
                    f"{run_result['individual_nodes']:<12} | "
                    f"{run_result['business_nodes']:<12} | "
                    f"{run_result['transaction_rate']:<10.2f} | "
                    f"{run_result['total_transactions']:<12} | "
                    f"{run_result['pattern_count']:<10}"
                )

                # Collect per-run data for plotting
                per_run_records.append({
                    'config': cfg['name'],
                    'seed': seed,
                    'business_nodes': run_result['business_nodes'],
                    'total_transactions': run_result['total_transactions'],
                    'transaction_rate': run_result['transaction_rate'],
                    'pattern_count': run_result['pattern_count']
                })

        # Aggregate results for the current configuration
        metrics = {
            'individual_nodes': [r['individual_nodes'] for r in all_results],
            'business_nodes': [r['business_nodes'] for r in all_results],
            'transaction_rate': [r['transaction_rate'] for r in all_results],
            'total_transactions': [r['total_transactions'] for r in all_results],
            'pattern_count': [r['pattern_count'] for r in all_results]
        }

        target_businesses = cfg['individuals'] * cfg['business_probability']

        stats = {
            'Target Individuals': (cfg['individuals'], metrics['individual_nodes']),
            'Target Businesses': (target_businesses, metrics['business_nodes']),
            'Target Transaction Rate': (cfg['transaction_rate'], metrics['transaction_rate']),
            'Total Transactions': (None, metrics['total_transactions']),
            'Target Fraud Patterns': (cfg['target_fraud_patterns'], metrics['pattern_count'])
        }

        print("\nConfiguration Summary:")
        print(
            f"{'Metric':<30} {'Target':<10} {'Mean':<10} {'Std Dev':<10} {'Min':<10} {'Max':<10}")
        print("-" * 90)

        for name, (target, values) in stats.items():
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            target_str = str(round(target, 2)) if target is not None else "N/A"
            print(
                f"{name:<30} {target_str:<10} {mean:<10.2f} {std:<10.2f} {min_val:<10.0f} {max_val:<10.0f}")

        # Append per-config records to global list
        if 'global_per_run_records' not in locals():
            global_per_run_records = per_run_records
        else:
            global_per_run_records.extend(per_run_records)

    print("\n=== H3 Compliance Testing Complete ===")
    print("All configurations executed and summarised above. Review deviations between 'Target' and 'Mean' along with variability (Std Dev) to assess generator reliability.")

    # Generate and save visualizations using per-run data
    if 'global_per_run_records' in locals():
        per_run_df = pd.DataFrame(global_per_run_records)
        plot_compliance_results(per_run_df)

    return True


if __name__ == "__main__":
    success = run_h3_statistical_validation()
    sys.exit(0 if success else 1)
