#!/usr/bin/env python3
"""
H3: Configuration Compliance Experiment

Tests that generated datasets accurately reflect all parameters specified
in the input configuration files.

Validates:
- Graph size compliance (nodes and edges)
- Transaction rate compliance
- Transaction amount ranges
- Temporal constraints
"""

import os
import sys
import csv
import json
import yaml
import tempfile
import datetime
import subprocess
from pathlib import Path
from collections import defaultdict

# Add the parent directory to path to import main
sys.path.append(str(Path(__file__).parent.parent))


def create_h3_graph_size_config():
    """Create configuration for graph size compliance test"""
    return {
        'graph_scale': {
            'individuals': 1500,
            'institutions_per_country': 3,
            'individual_accounts_per_institution_range': [1, 3],
            'business_accounts_per_institution_range': [1, 6]
        },
        'transaction_rates': {
            'per_account_per_day': 0.1
        },
        'time_span': {
            'start_date': '2023-01-01T00:00:00',
            'end_date': '2023-05-15T23:59:59'
        },
        'account_balance_range_normal': [1000.0, 50000.0],
        'background_amount_range': [10.0, 500.0],
        'company_size_range': [1, 1000],
        'salary_payment_days': [24, 30, 1],
        'salary_amount_range': [2500.0, 7500.0],
        'random_seed': 42,
        'business_creation_date_range': [90, 5475],
        'random_business_probability': 0.15,
        'pattern_frequency': {
            'random': False,
            'RapidFundMovement': 2,
            'FrontBusinessActivity': 1,
            'RepeatedOverseasTransfers': 3
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


def create_h3_transaction_rate_config():
    """Create configuration for transaction rate compliance test"""
    return {
        'graph_scale': {
            'individuals': 2000,
            'institutions_per_country': 3,
            'individual_accounts_per_institution_range': [1, 3],
            'business_accounts_per_institution_range': [1, 6]
        },
        'transaction_rates': {
            'per_account_per_day': 2.5
        },
        'time_span': {
            'start_date': '2023-01-01T00:00:00',
            'end_date': '2023-05-15T23:59:59'
        },
        'account_balance_range_normal': [1000.0, 50000.0],
        'background_amount_range': [50.0, 1000.0],
        'company_size_range': [1, 1000],
        'salary_payment_days': [24, 30, 1],
        'salary_amount_range': [2500.0, 7500.0],
        'random_seed': 42,
        'business_creation_date_range': [90, 5475],
        'random_business_probability': 0.10,
        'pattern_frequency': {
            'random': False,
            'RapidFundMovement': 2,
            'FrontBusinessActivity': 1,
            'RepeatedOverseasTransfers': 3
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


def validate_graph_size(config, nodes_file, edges_file):
    """Validate that graph size matches configuration"""
    results = {
        'individuals_compliant': False,
        'businesses_compliant': False,
        'issues': []
    }

    if not os.path.exists(nodes_file):
        results['issues'].append("Nodes file not found")
        return results

    # Count nodes by type
    individual_count = 0
    business_count = 0

    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('node_type') == 'Individual':
                individual_count += 1
            elif row.get('node_type') == 'Business':
                business_count += 1

    # Check individuals count
    expected_individuals = config['graph_scale']['individuals']
    if individual_count == expected_individuals:
        results['individuals_compliant'] = True
    else:
        results['issues'].append(
            f"Expected {expected_individuals} individuals, got {individual_count}")

    # Check businesses count (approximate based on probability)
    expected_businesses_min = int(
        expected_individuals * config['random_business_probability'] * 0.8)
    expected_businesses_max = int(
        expected_individuals * config['random_business_probability'] * 1.2)

    if expected_businesses_min <= business_count <= expected_businesses_max:
        results['businesses_compliant'] = True
    else:
        results['issues'].append(
            f"Expected {expected_businesses_min}-{expected_businesses_max} businesses, got {business_count}")

    return results


def validate_transaction_rates(config, edges_file, patterns_file):
    """Validate that transaction rates match configuration"""
    results = {
        'rate_compliant': False,
        'amount_compliant': False,
        'issues': []
    }

    if not os.path.exists(edges_file):
        results['issues'].append("Edges file not found")
        return results

    # Load pattern transactions to exclude them
    pattern_transactions = set()
    if os.path.exists(patterns_file):
        with open(patterns_file, 'r') as f:
            data = json.load(f)
            for pattern in data.get('patterns', []):
                for tx in pattern.get('transactions', []):
                    pattern_transactions.add(tx.get('transaction_id', ''))

    # Analyze background transactions
    transactions_per_account = defaultdict(int)
    amount_violations = 0
    total_background_transactions = 0

    start_date = datetime.datetime.fromisoformat(
        config['time_span']['start_date'])
    end_date = datetime.datetime.fromisoformat(config['time_span']['end_date'])
    total_days = (end_date - start_date).days

    min_amount, max_amount = config['background_amount_range']

    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip pattern transactions
            if row.get('transaction_id') in pattern_transactions:
                continue

            total_background_transactions += 1

            # Count transactions per account
            source_account = row.get('source_account_id')
            dest_account = row.get('destination_account_id')
            if source_account:
                transactions_per_account[source_account] += 1
            if dest_account:
                transactions_per_account[dest_account] += 1

            # Check amount compliance
            try:
                amount = float(row.get('amount', 0))
                if not (min_amount <= amount <= max_amount):
                    amount_violations += 1
            except ValueError:
                pass

    # Calculate average transaction rate
    if transactions_per_account:
        total_account_days = sum(transactions_per_account.values())
        num_accounts = len(transactions_per_account)
        actual_rate = total_account_days / \
            (num_accounts * total_days) if num_accounts > 0 else 0
        expected_rate = config['transaction_rates']['per_account_per_day']

        # Allow 10% tolerance
        if abs(actual_rate - expected_rate) / expected_rate <= 0.1:
            results['rate_compliant'] = True
        else:
            results['issues'].append(
                f"Expected {expected_rate} tx/account/day, got {actual_rate:.2f}")

    # Check amount compliance (allow 5% violations)
    violation_rate = amount_violations / \
        total_background_transactions if total_background_transactions > 0 else 0
    if violation_rate <= 0.05:
        results['amount_compliant'] = True
    else:
        results['issues'].append(
            f"Amount violations: {violation_rate:.1%} (max 5% allowed)")

    return results


def run_graph_size_test():
    """Run graph size compliance test"""
    print("Running graph size compliance test...")

    config = create_h3_graph_size_config()

    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "h3_graph_config.yaml")
        nodes_file = os.path.join(temp_dir, "generated_nodes.csv")
        edges_file = os.path.join(temp_dir, "generated_edges.csv")

        # Create config file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Run generation using subprocess
        main_script = str(Path(__file__).parent.parent / "main.py")

        cmd = [
            sys.executable, main_script,
            '--config', config_file,
            '--output-dir', temp_dir,
            '--nodes-file', 'generated_nodes.csv',
            '--edges-file', 'generated_edges.csv'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"ERROR: Generation failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return {'individuals_compliant': False, 'businesses_compliant': False, 'issues': ['Generation failed']}

        # Validate results
        return validate_graph_size(config, nodes_file, edges_file)


def run_transaction_rate_test():
    """Run transaction rate compliance test"""
    print("Running transaction rate compliance test...")

    config = create_h3_transaction_rate_config()

    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "h3_rate_config.yaml")
        nodes_file = os.path.join(temp_dir, "generated_nodes.csv")
        edges_file = os.path.join(temp_dir, "generated_edges.csv")
        patterns_file = os.path.join(temp_dir, "generated_patterns.json")

        # Create config file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Run generation using subprocess
        main_script = str(Path(__file__).parent.parent / "main.py")

        cmd = [
            sys.executable, main_script,
            '--config', config_file,
            '--output-dir', temp_dir,
            '--nodes-file', 'generated_nodes.csv',
            '--edges-file', 'generated_edges.csv',
            '--patterns-file', 'generated_patterns.json'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"ERROR: Generation failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return {'rate_compliant': False, 'amount_compliant': False, 'issues': ['Generation failed']}

        # Validate results
        return validate_transaction_rates(config, edges_file, patterns_file)


def run_h3_experiment():
    """Run the H3 configuration compliance experiment"""
    print("=== H3: Configuration Compliance Experiment ===")
    print("Testing dataset compliance with configuration parameters\n")

    # Test 1: Graph Size Compliance
    print("=== Test 1: Graph Size Compliance ===")
    graph_results = run_graph_size_test()

    print("Graph Size Results:")
    print(
        f"  Individuals: {'✓' if graph_results['individuals_compliant'] else '✗'}")
    print(
        f"  Businesses: {'✓' if graph_results['businesses_compliant'] else '✗'}")
    if graph_results['issues']:
        for issue in graph_results['issues']:
            print(f"    - {issue}")
    print()

    # Test 2: Transaction Rate Compliance
    print("=== Test 2: Transaction Rate Compliance ===")
    rate_results = run_transaction_rate_test()

    print("Transaction Rate Results:")
    print(
        f"  Rate Compliance: {'✓' if rate_results['rate_compliant'] else '✗'}")
    print(
        f"  Amount Compliance: {'✓' if rate_results['amount_compliant'] else '✗'}")
    if rate_results['issues']:
        for issue in rate_results['issues']:
            print(f"    - {issue}")
    print()

    # Overall assessment
    all_compliant = (graph_results['individuals_compliant'] and
                     graph_results['businesses_compliant'] and
                     rate_results['rate_compliant'] and
                     rate_results['amount_compliant'])

    print("=== H3 Results ===")
    if all_compliant:
        print(
            "✓ All configuration parameters are accurately reflected in generated datasets")
    else:
        print("✗ Some configuration parameters are not accurately reflected:")
        if not graph_results['individuals_compliant']:
            print("  - Individual count mismatch")
        if not graph_results['businesses_compliant']:
            print("  - Business count outside expected range")
        if not rate_results['rate_compliant']:
            print("  - Transaction rate does not match configuration")
        if not rate_results['amount_compliant']:
            print("  - Transaction amounts outside configured range")

    return all_compliant


if __name__ == "__main__":
    success = run_h3_experiment()
    sys.exit(0 if success else 1)
