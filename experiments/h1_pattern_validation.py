#!/usr/bin/env python3
"""
H1: Pattern Validation Experiment

Tests that Tide generates money laundering patterns that structurally and temporally
match their formal definitions as specified in the research.

Validates three aspects for each pattern:
1. Entity selection meets requirements
2. Transaction amounts and types are correct
3. Temporal patterns follow definitions
"""

import os
import sys
import json
import datetime
import tempfile
import shutil
import subprocess
import yaml
from pathlib import Path

# Add the parent directory to path to import main
sys.path.append(str(Path(__file__).parent.parent))


def create_h1_config():
    """Create configuration for H1 pattern validation test"""
    config = {
        'graph_scale': {
            'individuals': 2000,
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
        'pattern_frequency': {
            'random': False,
            'RapidFundMovement': 1,
            'FrontBusinessActivity': 0,
            'RepeatedOverseasTransfers': 0
        },
        'reporting_threshold': 10000,
        'reporting_currency': 'EUR',
        'high_risk_business_probability': 0.05,
        'random_business_probability': 0.10,
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
        },
        'pattern_config': {
            'rapidMovement': {
                'min_accounts_for_pattern': 2,
                'max_sender_entities': 5,
                'transaction_params': {
                    'inflows': {
                        'min_inflows': 5,
                        'max_inflows': 10,
                        'amount_range': [500, 5000]
                    },
                    'withdrawals': {
                        'min_withdrawals': 3,
                        'max_withdrawals': 8,
                        'amount_range': [100, 2000]
                    },
                    'inflow_to_withdrawal_delay': [1, 24]
                }
            }
        }
    }
    return config


def validate_rapid_fund_movement(pattern_data):
    """Validate RapidFundMovement pattern structure and timing"""
    results = []

    for pattern in pattern_data:
        if pattern['pattern_type'] != 'RapidFundMovement':
            continue

        validation = {
            'pattern_id': pattern['pattern_id'],
            'entity_validation': True,
            'transaction_validation': True,
            'temporal_validation': True,
            'issues': []
        }

        # 1. Entity validation: Should have 5-10 accounts + 1 central
        entities = pattern['entities']
        if len(entities) < 6 or len(entities) > 11:
            validation['entity_validation'] = False
            validation['issues'].append(
                f"Expected 6-11 entities, got {len(entities)}")

        # 2. Transaction validation: Check amounts
        transactions = pattern['transactions']
        for tx in transactions:
            if tx['transaction_type'] == 'inflow':
                if not (500 <= tx['amount'] <= 5000):
                    validation['transaction_validation'] = False
                    validation['issues'].append(
                        f"Inflow amount {tx['amount']} outside range 500-5000")
            elif tx['transaction_type'] == 'withdrawal':
                if not (100 <= tx['amount'] <= 2000):
                    validation['transaction_validation'] = False
                    validation['issues'].append(
                        f"Withdrawal amount {tx['amount']} outside range 100-2000")

        # 3. Temporal validation: Check timing constraints
        if transactions:
            start_time = min(datetime.datetime.fromisoformat(
                tx['timestamp'].replace('Z', '+00:00')) for tx in transactions)
            end_time = max(datetime.datetime.fromisoformat(
                tx['timestamp'].replace('Z', '+00:00')) for tx in transactions)
            duration_hours = (end_time - start_time).total_seconds() / 3600

            if duration_hours > 72:
                validation['temporal_validation'] = False
                validation['issues'].append(
                    f"Pattern duration {duration_hours:.1f}h exceeds 72h limit")

        results.append(validation)

    return results


def validate_front_business(pattern_data):
    """Validate FrontBusinessActivity pattern structure"""
    results = []

    for pattern in pattern_data:
        if pattern['pattern_type'] != 'FrontBusinessActivity':
            continue

        validation = {
            'pattern_id': pattern['pattern_id'],
            'entity_validation': True,
            'transaction_validation': True,
            'temporal_validation': True,
            'issues': []
        }

        # 1. Entity validation: Should have 3+ entities (simplified validation)
        entities = pattern['entities']
        # Note: entities is a list of entity IDs (strings), not objects
        # For pattern validation, we focus on the structural requirement of having multiple entities
        if len(entities) < 3:
            validation['entity_validation'] = False
            validation['issues'].append(
                f"Expected 3+ entities, got {len(entities)}")

        # 2. Transaction validation: Check deposit amounts
        transactions = pattern['transactions']
        for tx in transactions:
            if tx['transaction_type'] in ['deposit', 'transfer']:
                if not (10000 <= tx['amount'] <= 50000):
                    validation['transaction_validation'] = False
                    validation['issues'].append(
                        f"Deposit amount {tx['amount']} outside range 10000-50000")

        results.append(validation)

    return results


def validate_overseas_transfers(pattern_data):
    """Validate RepeatedOverseasTransfers pattern structure"""
    results = []

    for pattern in pattern_data:
        if pattern['pattern_type'] != 'RepeatedOverseasTransfers':
            continue

        validation = {
            'pattern_id': pattern['pattern_id'],
            'entity_validation': True,
            'transaction_validation': True,
            'temporal_validation': True,
            'issues': []
        }

        # 1. Entity validation: Should have 1 source + 2-5 overseas destinations
        entities = pattern['entities']
        if len(entities) < 3 or len(entities) > 6:
            validation['entity_validation'] = False
            validation['issues'].append(
                f"Expected 3-6 entities, got {len(entities)}")

        # 2. Transaction validation: Check transfer amounts
        transactions = pattern['transactions']
        for tx in transactions:
            if not (5000 <= tx['amount'] <= 20000):
                validation['transaction_validation'] = False
                validation['issues'].append(
                    f"Transfer amount {tx['amount']} outside range 5000-20000")

        # 3. Temporal validation: Check intervals
        if len(transactions) >= 2:
            times = [datetime.datetime.fromisoformat(
                tx['timestamp'].replace('Z', '+00:00')) for tx in transactions]
            times.sort()

            for i in range(1, len(times)):
                interval_days = (times[i] - times[i-1]).days
                if interval_days not in [7, 14, 30]:
                    validation['temporal_validation'] = False
                    validation['issues'].append(
                        f"Interval {interval_days} days not in [7, 14, 30]")
                    break

        results.append(validation)

    return results


def run_h1_experiment():
    """Run the H1 pattern validation experiment"""
    print("=== H1: Pattern Validation Experiment ===")
    print("Testing structural and temporal pattern compliance\n")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "h1_config.yaml")
        patterns_file = os.path.join(temp_dir, "generated_patterns.json")

        # Create config file
        config = create_h1_config()
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Run generation using subprocess
        print("Generating synthetic dataset...")
        main_script = str(Path(__file__).parent.parent / "main.py")

        cmd = [
            sys.executable, main_script,
            '--config', config_file,
            '--output-dir', temp_dir,
            '--patterns-file', 'generated_patterns.json'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Print the captured output
        if result.stdout:
            print("Generation output:")
            print(result.stdout)
        if result.stderr:
            print("Generation warnings/errors:")
            print(result.stderr)

        if result.returncode != 0:
            print(
                f"ERROR: Generation failed with return code {result.returncode}")
            return False

        # Load and validate patterns
        if not os.path.exists(patterns_file):
            print("ERROR: No patterns file generated!")
            return False

        with open(patterns_file, 'r') as f:
            data = json.load(f)

        patterns = data.get('patterns', [])
        print(f"Generated {len(patterns)} patterns for validation\n")

        # Validate each pattern type
        all_passed = True

        # RapidFundMovement validation
        rfm_results = validate_rapid_fund_movement(patterns)
        print("RapidFundMovement Patterns:")
        for result in rfm_results:
            status = "PASS" if all(
                [result['entity_validation'], result['transaction_validation'], result['temporal_validation']]) else "FAIL"
            print(f"  Pattern {result['pattern_id']}: {status}")
            if result['issues']:
                for issue in result['issues']:
                    print(f"    - {issue}")
                all_passed = False
        print()

        # FrontBusinessActivity validation
        fb_results = validate_front_business(patterns)
        print("FrontBusinessActivity Patterns:")
        for result in fb_results:
            status = "PASS" if all(
                [result['entity_validation'], result['transaction_validation'], result['temporal_validation']]) else "FAIL"
            print(f"  Pattern {result['pattern_id']}: {status}")
            if result['issues']:
                for issue in result['issues']:
                    print(f"    - {issue}")
                all_passed = False
        print()

        # RepeatedOverseasTransfers validation
        rot_results = validate_overseas_transfers(patterns)
        print("RepeatedOverseasTransfers Patterns:")
        for result in rot_results:
            status = "PASS" if all(
                [result['entity_validation'], result['transaction_validation'], result['temporal_validation']]) else "FAIL"
            print(f"  Pattern {result['pattern_id']}: {status}")
            if result['issues']:
                for issue in result['issues']:
                    print(f"    - {issue}")
                all_passed = False
        print()

        # Summary
        print("=== H1 Results ===")
        if all_passed:
            print("✓ All patterns validate correctly against formal definitions")
        else:
            print("✗ Some patterns failed validation")

        return all_passed


if __name__ == "__main__":
    success = run_h1_experiment()
    sys.exit(0 if success else 1)
