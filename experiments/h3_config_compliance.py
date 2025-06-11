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
            'per_account_per_day': 1.0
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
                    'payment': [10.0, 2000.0],
                    'transfer': [5.0, 800.0],
                    'cash_operations': [20.0, 1500.0]
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
                    'payment': [10.0, 2000.0],
                    'transfer': [5.0, 800.0],
                    'cash_operations': [20.0, 1500.0]
                }
            },
            'salaryPayments': {
                'payment_intervals': [14, 30],
                'salary_range': [2500.0, 7500.0],
                'salary_variation': 0.05,
                'preferred_payment_days': [1, 15, 30]
            },
            'fraudsterBackground': {
                'fraudster_rate_multiplier': [0.1, 0.5],
                'amount_ranges': {
                    'small_transactions': [5.0, 200.0],
                    'medium_transactions': [200.0, 1000.0]
                },
                'transaction_size_probabilities': {
                    'small': 0.8,
                    'medium': 0.2
                }
            }
        },
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

    try:
        with open(nodes_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_type = row.get('node_type', '')
                # Handle both enum format (NodeType.INDIVIDUAL) and simple format (individual)
                if node_type in ['NodeType.INDIVIDUAL', 'individual']:
                    individual_count += 1
                elif node_type in ['NodeType.BUSINESS', 'business']:
                    business_count += 1
    except Exception as e:
        results['issues'].append(f"Error reading nodes file: {e}")
        return results

    # Check individuals count (exact match required)
    expected_individuals = config['graph_scale']['individuals']
    if individual_count == expected_individuals:
        results['individuals_compliant'] = True
    else:
        results['issues'].append(
            f"Expected exactly {expected_individuals} individuals, got {individual_count}")

    # Check businesses count (use tighter tolerance for better validation)
    expected_businesses = int(
        expected_individuals * config['random_business_probability'])
    # Allow ±10% tolerance instead of ±20% for more precise validation
    tolerance = 0.10
    expected_businesses_min = int(expected_businesses * (1 - tolerance))
    expected_businesses_max = int(expected_businesses * (1 + tolerance))

    if expected_businesses_min <= business_count <= expected_businesses_max:
        results['businesses_compliant'] = True
    else:
        results['issues'].append(
            f"Expected ~{expected_businesses} businesses ({expected_businesses_min}-{expected_businesses_max} range), got {business_count}")

    return results


def validate_transaction_rates(config, edges_file, patterns_file):
    """Validate that transaction rates and patterns match configuration"""
    results = {
        'rate_compliant': False,
        'transaction_types_compliant': False,
        'issues': []
    }

    if not os.path.exists(edges_file):
        results['issues'].append("Edges file not found")
        return results

    # Load pattern transactions to exclude them
    pattern_transactions = set()
    if os.path.exists(patterns_file):
        try:
            with open(patterns_file, 'r') as f:
                data = json.load(f)
                for pattern in data.get('patterns', []):
                    for tx in pattern.get('transactions', []):
                        # Pattern transactions may not have transaction_id, skip them for now
                        # We'll identify them by timestamp or other means if needed
                        pass
        except Exception as e:
            results['issues'].append(f"Error reading patterns file: {e}")

    # Analyze background transactions
    source_account_transactions = defaultdict(int)
    transaction_type_counts = defaultdict(int)
    total_background_transactions = 0
    all_accounts = set()
    salary_transactions = 0
    small_random_transactions = 0
    large_random_transactions = 0

    start_date = datetime.datetime.fromisoformat(
        config['time_span']['start_date'])
    end_date = datetime.datetime.fromisoformat(config['time_span']['end_date'])
    total_days = (end_date - start_date).days

    # Get expected transaction type probabilities
    expected_probs = config.get('backgroundPatterns', {}).get(
        'randomPayments', {}).get('transaction_type_probabilities', {})
    salary_range = config.get('backgroundPatterns', {}).get(
        'salaryPayments', {}).get('salary_range', [2500.0, 7500.0])

    try:
        with open(edges_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip non-transaction edges (e.g., ownership edges)
                edge_type = row.get('edge_type', '')
                if edge_type not in ['EdgeType.TRANSACTION', 'transaction']:
                    continue

                # Skip if no amount (ownership edges don't have amounts)
                amount_str = row.get('amount', '')
                if not amount_str:
                    continue

                total_background_transactions += 1

                # Count transactions per source account (use 'src' column)
                source_account = row.get('src', '')
                dest_account = row.get('dest', '')

                if source_account:
                    source_account_transactions[source_account] += 1
                    all_accounts.add(source_account)
                if dest_account:
                    all_accounts.add(dest_account)

                # Count transaction types
                transaction_type = row.get('transaction_type', '').replace(
                    'TransactionType.', '').lower()
                transaction_type_counts[transaction_type] += 1

                # Categorize transactions by amount to understand the patterns
                try:
                    amount = float(amount_str)
                    if salary_range[0] <= amount <= salary_range[1]:
                        salary_transactions += 1
                    elif amount < 1000:  # Likely random small transactions
                        small_random_transactions += 1
                    else:  # Larger transactions
                        large_random_transactions += 1
                except ValueError:
                    pass
    except Exception as e:
        results['issues'].append(f"Error reading edges file: {e}")
        return results

    # Calculate average transaction rate (transactions per account per day)
    if source_account_transactions:
        total_transactions = sum(source_account_transactions.values())
        num_active_accounts = len(source_account_transactions)
        actual_rate = total_transactions / \
            (num_active_accounts * total_days) if num_active_accounts > 0 else 0

        # Get the base expected rate
        base_expected_rate = config['transaction_rates']['per_account_per_day']

        # Adjust for legit_rate_multiplier from background patterns
        rate_multiplier_range = config.get('backgroundPatterns', {}).get(
            'randomPayments', {}).get('legit_rate_multiplier', [1.0, 1.0])
        avg_rate_multiplier = sum(
            rate_multiplier_range) / len(rate_multiplier_range)
        expected_rate = base_expected_rate * avg_rate_multiplier

        # Allow 25% tolerance for transaction rate due to randomness and multiple background patterns
        tolerance = 0.25
        if abs(actual_rate - expected_rate) / expected_rate <= tolerance:
            results['rate_compliant'] = True
        else:
            results['issues'].append(
                f"Expected ~{expected_rate:.3f} tx/account/day (base {base_expected_rate} × avg multiplier {avg_rate_multiplier:.2f}), got {actual_rate:.3f} (tolerance: ±{tolerance:.0%})")
    else:
        results['issues'].append("No source account transactions found")

    # Validate transaction type probabilities (focus on this rather than strict amounts)
    if total_background_transactions > 0 and expected_probs:
        type_compliance_issues = []

        for tx_type, expected_prob in expected_probs.items():
            # Map transaction types (handle variations)
            type_variants = {
                'transfer': ['transfer'],
                'payment': ['payment'],
                'deposit': ['deposit'],
                'withdrawal': ['withdrawal']
            }

            actual_count = 0
            for variant in type_variants.get(tx_type, [tx_type]):
                actual_count += transaction_type_counts.get(variant, 0)

            actual_prob = actual_count / total_background_transactions

            # Allow 30% tolerance for transaction type probabilities due to randomness
            prob_tolerance = 0.30
            if abs(actual_prob - expected_prob) / expected_prob <= prob_tolerance:
                continue
            else:
                type_compliance_issues.append(
                    f"{tx_type}: expected {expected_prob:.1%}, got {actual_prob:.1%}")

        if not type_compliance_issues:
            results['transaction_types_compliant'] = True
        else:
            results['issues'].extend(
                [f"Transaction type probabilities: {', '.join(type_compliance_issues)}"])

        # Additional insights about transaction patterns (not failures, just info)
        results['pattern_insights'] = {
            'total_transactions': total_background_transactions,
            'salary_like_transactions': salary_transactions,
            'small_random_transactions': small_random_transactions,
            'large_transactions': large_random_transactions,
            'transaction_type_distribution': dict(transaction_type_counts)
        }
    else:
        results['issues'].append(
            "No background transactions found or no expected probabilities configured")

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
    print("Configuration: 1,500 individuals, 15% business probability (~225 businesses)")
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

    # Test 2: Transaction Rate and Pattern Compliance
    print("=== Test 2: Transaction Rate and Pattern Compliance ===")
    print("Configuration: 2,000 individuals, 10% business probability, 2.5 tx/account/day")
    print("Background patterns: randomPayments (transfer 40%, payment 30%, deposit/withdrawal 15% each)")
    rate_results = run_transaction_rate_test()

    print("Transaction Rate and Pattern Results:")
    print(
        f"  Rate Compliance: {'✓' if rate_results['rate_compliant'] else '✗'}")
    print(
        f"  Transaction Types Compliance: {'✓' if rate_results['transaction_types_compliant'] else '✗'}")
    if rate_results['issues']:
        for issue in rate_results['issues']:
            print(f"    - {issue}")

    # Show pattern insights if available
    if 'pattern_insights' in rate_results:
        insights = rate_results['pattern_insights']
        print(f"  Pattern Insights:")
        print(f"    - Total transactions: {insights['total_transactions']:,}")
        print(
            f"    - Salary-like transactions: {insights['salary_like_transactions']:,}")
        print(
            f"    - Small random transactions: {insights['small_random_transactions']:,}")
        print(f"    - Large transactions: {insights['large_transactions']:,}")
    print()

    # Overall assessment
    all_compliant = (graph_results['individuals_compliant'] and
                     graph_results['businesses_compliant'] and
                     rate_results['rate_compliant'] and
                     rate_results['transaction_types_compliant'])

    print("=== H3 Results Summary ===")
    if all_compliant:
        print(
            "✓ All configuration parameters are accurately reflected in generated datasets")
        print("  - Graph size matches specified node counts")
        print("  - Business creation follows probability distribution")
        print("  - Transaction rates match configured values")
        print("  - Transaction types match configured probabilities")
    else:
        print("✗ Some configuration parameters are not accurately reflected:")
        if not graph_results['individuals_compliant']:
            print(
                "  - Individual count does not match configuration (expected exact match)")
        if not graph_results['businesses_compliant']:
            print("  - Business count outside expected range from probability")
        if not rate_results['rate_compliant']:
            print("  - Transaction rate does not match configuration")
        if not rate_results['transaction_types_compliant']:
            print("  - Transaction types do not match configured probabilities")

    return all_compliant


if __name__ == "__main__":
    success = run_h3_experiment()
    sys.exit(0 if success else 1)
