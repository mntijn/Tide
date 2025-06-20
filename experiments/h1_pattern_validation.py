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
            'RapidFundMovement': 3,
            'FrontBusinessActivity': 2,
            'RepeatedOverseasTransfers': 3,
            'UTurnTransactions': 3
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
                        'amount_range': [500, 6000]
                    },
                    'withdrawals': {
                        'min_withdrawals': 3,
                        'max_withdrawals': 8
                    },
                    'inflow_to_withdrawal_delay': [1, 24]
                },
                'validation_params': {
                    'outflow_ratio_range': [0.85, 0.95],
                    'max_inflow_phase_duration_hours': 24,
                    'min_phase_delay_hours': 1,
                    'max_total_duration_hours': 128
                }
            },
            'frontBusiness': {
                'min_entities': 3,
                'transaction_params': {
                    'deposit_amount_range': [15000, 75000],
                    'min_deposits': 5,
                    'max_deposits': 15
                }
            },
            'repeatedOverseas': {
                'min_overseas_entities': 2,
                'max_overseas_entities': 5,
                'transaction_params': {
                    'transfer_amount_range': [5000, 20000],
                    'min_transactions': 4,
                    'max_transactions': 12,
                    'transfer_interval_days': [7, 14, 30]
                }
            },
            'uTurnTransactions': {
                'min_intermediaries': 2,
                'max_intermediaries': 5,
                'transaction_params': {
                    'initial_amount_range': [10000, 100000],
                    'return_percentage_range': [0.7, 0.9],
                    'international_delay_days': [1, 5],
                    'time_variability': {
                        'business_hours': [9, 16],
                        'include_minutes': True,
                        'include_hours': True
                    }
                },
                'validation_params': {
                    'fee_percentage_range': [0.01, 0.03]
                }
            }
        }
    }
    return config


def validate_rapid_fund_movement(pattern_data, config):
    """Validate RapidFundMovement pattern structure and timing"""
    results = []

    # Extract validation parameters from config
    rapid_config = config.get('pattern_config', {}).get('rapidMovement', {})
    tx_params = rapid_config.get('transaction_params', {})
    inflow_params = tx_params.get('inflows', {})
    withdrawal_params = tx_params.get('withdrawals', {})
    validation_params = rapid_config.get('validation_params', {})

    # Get ranges from config with fallbacks
    inflow_amount_range = inflow_params.get('amount_range', [500, 6000])
    outflow_ratio_range = validation_params.get(
        'outflow_ratio_range', [0.85, 0.95])
    max_inflow_duration = validation_params.get(
        'max_inflow_phase_duration_hours', 24)
    min_phase_delay = validation_params.get('min_phase_delay_hours', 1)
    max_total_duration = validation_params.get('max_total_duration_hours', 72)

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

        # 2. Transaction validation: Check amounts using config values
        transactions = pattern['transactions']
        for tx in transactions:
            if tx['transaction_type'] == 'TransactionType.TRANSFER':
                if not (inflow_amount_range[0] <= tx['amount'] <= inflow_amount_range[1]):
                    validation['transaction_validation'] = False
                    validation['issues'].append(
                        f"Inflow amount {tx['amount']} outside range {inflow_amount_range[0]}-{inflow_amount_range[1]}")
            # Note: Withdrawal amounts are structured below reporting thresholds automatically

        # 3. Temporal validation: Check timing constraints and two-phase pattern
        if transactions:
            # Separate inflows and withdrawals
            inflows = [
                tx for tx in transactions if tx['transaction_type'] == 'TransactionType.TRANSFER']
            withdrawals = [
                tx for tx in transactions if tx['transaction_type'] == 'TransactionType.WITHDRAWAL']

            print(
                f"    Pattern {validation['pattern_id']}: {len(inflows)} inflows, {len(withdrawals)} withdrawals")

            if inflows and withdrawals:
                # Convert timestamps to datetime objects
                inflow_times = [datetime.datetime.fromisoformat(
                    tx['timestamp'].replace('Z', '+00:00')) for tx in inflows]
                withdrawal_times = [datetime.datetime.fromisoformat(
                    tx['timestamp'].replace('Z', '+00:00')) for tx in withdrawals]

                # Phase 1: Check inflow timing (using config value)
                inflow_start = min(inflow_times)
                inflow_end = max(inflow_times)
                inflow_duration_hours = (
                    inflow_end - inflow_start).total_seconds() / 3600

                print(
                    f"    Inflow phase: {inflow_duration_hours:.1f}h duration (limit: {max_inflow_duration}h)")

                if inflow_duration_hours > max_inflow_duration:
                    validation['temporal_validation'] = False
                    validation['issues'].append(
                        f"Inflow phase duration {inflow_duration_hours:.1f}h exceeds {max_inflow_duration}h limit")

                # Phase 2: Check that withdrawals come after inflows (with delay)
                earliest_withdrawal = min(withdrawal_times)
                latest_inflow = max(inflow_times)
                latest_withdrawal = max(withdrawal_times)

                if earliest_withdrawal <= latest_inflow:
                    validation['temporal_validation'] = False
                    validation['issues'].append(
                        "Withdrawals should occur after inflow phase completion")

                # Check delay between phases (using config value)
                phase_delay_hours = (
                    earliest_withdrawal - latest_inflow).total_seconds() / 3600
                print(
                    f"    Phase delay: {phase_delay_hours:.1f}h (minimum: {min_phase_delay}h)")

                if phase_delay_hours < min_phase_delay:
                    validation['temporal_validation'] = False
                    validation['issues'].append(
                        f"Phase delay {phase_delay_hours:.1f}h is less than minimum {min_phase_delay}h")

                # Check withdrawal phase duration
                withdrawal_duration_hours = (
                    latest_withdrawal - earliest_withdrawal).total_seconds() / 3600
                print(
                    f"    Withdrawal phase: {withdrawal_duration_hours:.1f}h duration")

                # Check total pattern duration
                total_pattern_duration_hours = (
                    latest_withdrawal - inflow_start).total_seconds() / 3600
                print(
                    f"    Total pattern: {total_pattern_duration_hours:.1f}h duration (limit: {max_total_duration}h)")

                # Check amount relationship: outflow should be within configured ratio
                total_inflow = sum(tx['amount'] for tx in inflows)
                total_outflow = sum(tx['amount'] for tx in withdrawals)

                print(
                    f"    Amounts: inflow={total_inflow:.2f}, outflow={total_outflow:.2f}")

                if total_inflow > 0:
                    outflow_ratio = total_outflow / total_inflow
                    print(
                        f"    Outflow ratio: {outflow_ratio:.3f} (expected: {outflow_ratio_range[0]}-{outflow_ratio_range[1]})")

                    if not (outflow_ratio_range[0] <= outflow_ratio <= outflow_ratio_range[1]):
                        validation['temporal_validation'] = False
                        validation['issues'].append(
                            f"Outflow ratio {outflow_ratio:.2f} not in range {outflow_ratio_range[0]}-{outflow_ratio_range[1]}")

            # Overall duration check (using config value)
            start_time = min(datetime.datetime.fromisoformat(
                tx['timestamp'].replace('Z', '+00:00')) for tx in transactions)
            end_time = max(datetime.datetime.fromisoformat(
                tx['timestamp'].replace('Z', '+00:00')) for tx in transactions)
            duration_hours = (end_time - start_time).total_seconds() / 3600

            if duration_hours > max_total_duration:
                validation['temporal_validation'] = False
                validation['issues'].append(
                    f"Pattern duration {duration_hours:.1f}h exceeds {max_total_duration}h limit")

        results.append(validation)

    return results


def validate_front_business(pattern_data, config):
    """Validate FrontBusinessActivity pattern structure"""
    results = []

    front_config = config.get('pattern_config', {}).get('frontBusiness', {})
    tx_params = front_config.get('transaction_params', {})
    deposit_amount_range = tx_params.get(
        'deposit_amount_range', [15000, 75000])
    min_entities = front_config.get('min_entities', 3)

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

        # 1. Entity validation
        entities = pattern['entities']
        if len(entities) < min_entities:
            validation['entity_validation'] = False
            validation['issues'].append(
                f"Expected {min_entities}+ entities, got {len(entities)}")

        transactions = pattern['transactions']
        if not transactions:
            validation['transaction_validation'] = False
            validation['issues'].append("No transactions found in pattern")
            results.append(validation)
            continue

        deposits = [
            tx for tx in transactions if tx['transaction_type'] == 'TransactionType.DEPOSIT']
        transfers = [
            tx for tx in transactions if tx['transaction_type'] == 'TransactionType.TRANSFER']

        # 2. Transaction validation
        if not deposits:
            validation['transaction_validation'] = False
            validation['issues'].append("No deposit transactions found.")
        if not transfers:
            validation['transaction_validation'] = False
            validation['issues'].append("No transfer transactions found.")

        for tx in deposits:
            if not (deposit_amount_range[0] <= tx['amount'] <= deposit_amount_range[1]):
                validation['transaction_validation'] = False
                validation['issues'].append(
                    f"Deposit amount {tx['amount']} outside range {deposit_amount_range[0]}-{deposit_amount_range[1]}")

        # 3. Temporal and Amount validation
        if len(deposits) != len(transfers):
            validation['transaction_validation'] = False
            validation['issues'].append(
                f"Mismatch between number of deposits ({len(deposits)}) and transfers ({len(transfers)}).")

        if deposits and transfers:
            # Sort both by time to attempt to match pairs
            deposits.sort(key=lambda x: x['timestamp'])
            transfers.sort(key=lambda x: x['timestamp'])

            total_deposited = sum(tx['amount'] for tx in deposits)
            total_transferred = sum(tx['amount'] for tx in transfers)

            for i in range(len(deposits)):
                deposit = deposits[i]
                transfer = transfers[i]

                deposit_time = datetime.datetime.fromisoformat(
                    deposit['timestamp'].replace('Z', '+00:00'))
                transfer_time = datetime.datetime.fromisoformat(
                    transfer['timestamp'].replace('Z', '+00:00'))

                # Check for pairing: transfer must follow deposit
                if transfer_time < deposit_time:
                    validation['temporal_validation'] = False
                    validation['issues'].append(
                        f"Pair {i+1}: Transfer happens before its corresponding deposit.")

                # Check delay for each pair
                delay_hours = (transfer_time -
                               deposit_time).total_seconds() / 3600
                if not (0.5 <= delay_hours <= 6.0):
                    validation['temporal_validation'] = False
                    validation['issues'].append(
                        f"Pair {i+1}: Delay between deposit and transfer is {delay_hours:.2f}h, expected 0.5-6h.")

                # Check for transfer amount ratio for each pair
                if deposit['amount'] > 0:
                    ratio = transfer['amount'] / deposit['amount']
                    if not (0.80 <= ratio <= 1.0):
                        validation['transaction_validation'] = False
                        validation['issues'].append(
                            f"Pair {i+1}: Transfer amount ({transfer['amount']:.2f}) is {ratio:.2f} of deposit ({deposit['amount']:.2f}), expected 0.8-1.0.")
                else:
                    validation['transaction_validation'] = False
                    validation['issues'].append(
                        f"Pair {i+1}: Deposit amount is zero.")

        results.append(validation)

    return results


def validate_overseas_transfers(pattern_data, config):
    """Validate RepeatedOverseasTransfers pattern structure"""
    results = []

    # Extract validation parameters from config
    overseas_config = config.get(
        'pattern_config', {}).get('repeatedOverseas', {})
    tx_params = overseas_config.get('transaction_params', {})
    transfer_amount_range = tx_params.get(
        'transfer_amount_range', [5000, 20000])
    interval_days = tx_params.get('transfer_interval_days', [7, 14, 30])
    min_destinations = overseas_config.get('min_overseas_entities', 2)
    max_destinations = overseas_config.get('max_overseas_entities', 5)

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

        # 1. Entity validation: Should have 1 source + 2-5 overseas destinations (using config values)
        entities = pattern['entities']
        expected_min = min_destinations + 1  # +1 for source
        expected_max = max_destinations + 1  # +1 for source
        if len(entities) < expected_min or len(entities) > expected_max:
            validation['entity_validation'] = False
            validation['issues'].append(
                f"Expected {expected_min}-{expected_max} entities, got {len(entities)}")

        # 2. Transaction validation: Check transfer amounts (using config values)
        transactions = pattern['transactions']
        for tx in transactions:
            if not (transfer_amount_range[0] <= tx['amount'] <= transfer_amount_range[1]):
                validation['transaction_validation'] = False
                validation['issues'].append(
                    f"Transfer amount {tx['amount']} outside range {transfer_amount_range[0]}-{transfer_amount_range[1]}")

        # 3. Temporal validation: Check intervals
        if len(transactions) >= 2:
            times = [datetime.datetime.fromisoformat(
                tx['timestamp'].replace('Z', '+00:00')) for tx in transactions]
            times.sort()

            # Check if this looks like a periodic pattern (consistent intervals)
            intervals = [
                (times[i] - times[i-1]).days for i in range(1, len(times))]
            unique_intervals = set(intervals)

            # If most intervals are consistent and match config, it's likely periodic
            is_likely_periodic = len(unique_intervals) <= 2 and any(
                interval in interval_days for interval in unique_intervals)

            if is_likely_periodic:
                # For periodic patterns, validate strict adherence to configured intervals
                for i, interval_days_actual in enumerate(intervals):
                    if interval_days_actual not in interval_days:
                        validation['temporal_validation'] = False
                        validation['issues'].append(
                            f"Periodic pattern interval {interval_days_actual} days not in {interval_days}")
                        break
            else:
                # For high_frequency patterns, just check that intervals are reasonable (not too large)
                max_reasonable_interval = max(
                    interval_days) * 2  # Allow some flexibility
                for i, interval_days_actual in enumerate(intervals):
                    if interval_days_actual > max_reasonable_interval:
                        validation['temporal_validation'] = False
                        validation['issues'].append(
                            f"High frequency pattern has too large interval: {interval_days_actual} days")
                        break

        results.append(validation)

    return results


def validate_u_turn_transactions(pattern_data, config):
    """Validate UTurnTransactions pattern structure and timing"""
    results = []

    # Extract validation parameters from config
    u_turn_config = config.get('pattern_config', {}).get(
        'uTurnTransactions', {})
    tx_params = u_turn_config.get('transaction_params', {})
    validation_params = u_turn_config.get('validation_params', {})

    min_intermediaries = u_turn_config.get('min_intermediaries', 2)
    max_intermediaries = u_turn_config.get('max_intermediaries', 5)

    delay_days_range = tx_params.get('international_delay_days', [1, 5])
    fee_range = validation_params.get('fee_percentage_range', [0.01, 0.03])
    return_percentage_range = tx_params.get(
        'return_percentage_range', [0.7, 0.9])

    for pattern in pattern_data:
        if pattern['pattern_type'] != 'UTurnTransactions':
            continue

        validation = {
            'pattern_id': pattern['pattern_id'],
            'entity_validation': True,
            'transaction_validation': True,
            'temporal_validation': True,
            'issues': []
        }

        # 1. Entity validation
        entities = pattern['entities']
        min_entities = min_intermediaries + 2
        max_entities = max_intermediaries + 2
        if not (min_entities <= len(entities) <= max_entities):
            validation['entity_validation'] = False
            validation['issues'].append(
                f"Expected {min_entities}-{max_entities} entities, got {len(entities)}")

        transactions = pattern['transactions']
        expected_min_tx = min_intermediaries + 1  # +1 for return tx
        if not transactions or len(transactions) < expected_min_tx:
            if not transactions:
                validation['issues'].append(
                    "No transactions found in pattern.")
            else:
                validation['issues'].append(
                    f"Not enough transactions. Expected at least {expected_min_tx}, found {len(transactions)}.")
            validation['transaction_validation'] = False
            results.append(validation)
            continue

        try:
            transactions.sort(key=lambda tx: tx['timestamp'])
        except KeyError:
            validation['transaction_validation'] = False
            validation['issues'].append(
                "Transactions in pattern data are missing 'timestamp'.")
            results.append(validation)
            continue

        # Check for source/destination IDs first. If not present, can't validate path and amounts between steps.
        has_src_dest_ids = all(
            'src' in tx and 'dest' in tx for tx in transactions)
        if not has_src_dest_ids:
            validation['transaction_validation'] = False
            validation['issues'].append(
                "Transactions are missing source_id/destination_id, cannot validate path.")
            results.append(validation)
            continue

        # 2. Transaction and Temporal validation
        for i in range(1, len(transactions)):
            tx_prev = transactions[i-1]
            tx_curr = transactions[i]

            # Path continuity
            if tx_prev['dest'] != tx_curr['src']:
                validation['transaction_validation'] = False
                validation['issues'].append(
                    f"Transaction path broken at hop {i}: {tx_prev['dest']} -> {tx_curr['src']}")

            # Time delay
            t_prev = datetime.datetime.fromisoformat(
                tx_prev['timestamp'].replace('Z', '+00:00'))
            t_curr = datetime.datetime.fromisoformat(
                tx_curr['timestamp'].replace('Z', '+00:00'))
            delay_days = (t_curr - t_prev).total_seconds() / (24 * 3600)
            if not (delay_days_range[0] <= delay_days <= delay_days_range[1]):
                validation['temporal_validation'] = False
                validation['issues'].append(
                    f"Delay of {delay_days:.1f} days at hop {i} is outside range {delay_days_range[0]}-{delay_days_range[1]}")

            # Amount change
            if tx_prev['amount'] <= 0:
                validation['transaction_validation'] = False
                validation['issues'].append(
                    f"Transaction amount for previous hop {i-1} is non-positive.")
                continue

            ratio = tx_curr['amount'] / tx_prev['amount']

            if i < len(transactions) - 1:  # Intermediary hops
                fee_percentage = 1 - ratio
                if not (fee_range[0] <= fee_percentage <= fee_range[1]):
                    validation['transaction_validation'] = False
                    validation['issues'].append(
                        f"Transaction fee at hop {i} is {fee_percentage:.3f}, outside range {fee_range[0]}-{fee_range[1]}")
            else:  # Final return transaction
                return_percentage = ratio
                if not (return_percentage_range[0] <= return_percentage <= return_percentage_range[1]):
                    validation['transaction_validation'] = False
                    validation['issues'].append(
                        f"Return amount percentage is {return_percentage:.3f}, outside range {return_percentage_range[0]}-{return_percentage_range[1]}")

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

        result = subprocess.run(cmd, capture_output=False, text=True)

        # Print the captured output
        # if result.stdout:
        #     print("Generation output:")
        #     print(result.stdout)
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
        rfm_results = validate_rapid_fund_movement(patterns, config)
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
        fb_results = validate_front_business(patterns, config)
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
        rot_results = validate_overseas_transfers(patterns, config)
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

        # UTurnTransactions validation
        utt_results = validate_u_turn_transactions(patterns, config)
        print("UTurnTransactions Patterns:")
        for result in utt_results:
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
