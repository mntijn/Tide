#!/usr/bin/env python3
"""
H4: Reproducibility Experiment

Tests that identical configurations with the same random seed produce
identical datasets.

Validates:
- Multiple runs with same seed produce identical outputs
- Different seeds produce different outputs
- All output files are identical (nodes, edges)
"""

import os
import sys
import yaml
import tempfile
import hashlib
import subprocess
from pathlib import Path

# Add the parent directory to path to import main
sys.path.append(str(Path(__file__).parent.parent))


def create_h4_config(seed):
    """Create configuration for reproducibility test"""
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
        'random_seed': seed,
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
                }
            },
            'frontBusiness': {
                'min_entities': 3,
                'min_accounts_for_front_business': 2,
                'num_front_business_accounts_to_use': 3,
                'min_overseas_destination_accounts': 2,
                'max_overseas_destination_accounts_for_front': 4,
                'transaction_params': {
                    'min_deposit_cycles': 5,
                    'max_deposit_cycles': 15,
                    'deposit_amount_range': [15000, 75000],
                    'deposits_per_cycle': [1, 3]
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
                }
            }
        }
    }


def calculate_file_hash(filepath):
    """Calculate SHA-256 hash of a file"""
    if not os.path.exists(filepath):
        return None

    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def read_file_lines(filepath, max_lines=None):
    """Read file lines for comparison"""
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if max_lines:
        return lines[:max_lines]
    return lines


def show_diff_snippet(lines1, lines2, label1="Run 1", label2="Run 2", max_diffs=5):
    """Show first few differences between two file contents"""
    diffs_found = 0
    print(f"\n    First {max_diffs} differences:")

    max_len = max(len(lines1), len(lines2))
    for i in range(max_len):
        line1 = lines1[i].strip() if i < len(lines1) else "<missing>"
        line2 = lines2[i].strip() if i < len(lines2) else "<missing>"

        if line1 != line2:
            diffs_found += 1
            print(f"      Line {i+1}:")
            print(f"        {label1}: {line1[:80]}{'...' if len(line1) > 80 else ''}")
            print(f"        {label2}: {line2[:80]}{'...' if len(line2) > 80 else ''}")

            if diffs_found >= max_diffs:
                remaining = sum(1 for j in range(i+1, max_len)
                               if (lines1[j].strip() if j < len(lines1) else "") !=
                                  (lines2[j].strip() if j < len(lines2) else ""))
                if remaining > 0:
                    print(f"      ... and {remaining} more differences")
                break

    if diffs_found == 0:
        print("      No line-by-line differences found (may be whitespace/ordering)")


def run_generation(config, run_id):
    """Run generation and return file hashes and content"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, f"h4_config_{run_id}.yaml")
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
            print(f"ERROR: Generation failed for run {run_id}")
            print(f"STDERR: {result.stderr}")
            return {
                'nodes': None,
                'edges': None,
                'nodes_content': [],
                'edges_content': [],
            }

        # Calculate hashes and read content
        return {
            'nodes': calculate_file_hash(nodes_file),
            'edges': calculate_file_hash(edges_file),
            'nodes_content': read_file_lines(nodes_file),
            'edges_content': read_file_lines(edges_file),
        }


def test_same_seed_reproducibility(seed, num_runs=3):
    """Test that multiple runs with same seed produce identical results"""
    print(f"Testing reproducibility with seed {seed} ({num_runs} runs)...")

    config = create_h4_config(seed)
    all_results = []

    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...")
        result = run_generation(config, f"{seed}_{i}")
        all_results.append(result)

    # Check if all hashes are identical
    results = {
        'nodes_identical': True,
        'edges_identical': True,
        'all_identical': True,
        'hashes': all_results
    }

    if not all_results:
        results['all_identical'] = False
        return results

    reference = all_results[0]

    for i, run_result in enumerate(all_results[1:], 1):
        if run_result['nodes'] != reference['nodes']:
            results['nodes_identical'] = False
            print(f"    ✗ Nodes differ in run {i+1}")
            show_diff_snippet(
                reference['nodes_content'],
                run_result['nodes_content'],
                "Run 1", f"Run {i+1}"
            )

        if run_result['edges'] != reference['edges']:
            results['edges_identical'] = False
            print(f"    ✗ Edges differ in run {i+1}")
            show_diff_snippet(
                reference['edges_content'],
                run_result['edges_content'],
                "Run 1", f"Run {i+1}"
            )

    results['all_identical'] = (results['nodes_identical'] and
                                results['edges_identical'])

    if results['all_identical']:
        print(f"    ✓ All runs identical")

    return results


def test_different_seed_variation(seed1, seed2):
    """Test that different seeds produce different results"""
    print(f"Testing seed variation (seed {seed1} vs seed {seed2})...")

    config1 = create_h4_config(seed1)
    config2 = create_h4_config(seed2)

    hashes1 = run_generation(config1, f"var_{seed1}")
    hashes2 = run_generation(config2, f"var_{seed2}")

    results = {
        'nodes_different': hashes1['nodes'] != hashes2['nodes'],
        'edges_different': hashes1['edges'] != hashes2['edges'],
        'any_different': False
    }

    results['any_different'] = (results['nodes_different'] or
                                results['edges_different'])

    if results['any_different']:
        print(f"    ✓ Seeds produce different outputs")
        if results['nodes_different']:
            print(f"      - Nodes differ")
        if results['edges_different']:
            print(f"      - Edges differ")
    else:
        print(f"    ✗ Seeds produce identical outputs (unexpected)")

    return results


def run_h4_experiment():
    """Run the H4 reproducibility experiment"""
    print("=== H4: Reproducibility Experiment ===")
    print("Testing deterministic output with random seeds\n")

    # Test 1: Same seed reproducibility
    print("=== Test 1: Same Seed Reproducibility ===")
    same_seed_results = test_same_seed_reproducibility(42, num_runs=3)

    print("Same Seed Results:")
    print(f"  Nodes: {'✓' if same_seed_results['nodes_identical'] else '✗'}")
    print(f"  Edges: {'✓' if same_seed_results['edges_identical'] else '✗'}")
    print()

    # Print hashes for seed 42
    print("Hashes for seed 42:")
    for i, hashes in enumerate(same_seed_results['hashes']):
        print(f"  Run {i+1}:")
        print(f"    Nodes:    {hashes['nodes']}")
        print(f"    Edges:    {hashes['edges']}")
    print()

    # Test 2: Different seed variation
    print("=== Test 2: Different Seed Variation ===")
    different_seed_results = test_different_seed_variation(42, 123)

    print("Different Seed Results:")
    print(
        f"  Nodes: {'✓' if different_seed_results['nodes_different'] else '✗'}")
    print(
        f"  Edges: {'✓' if different_seed_results['edges_different'] else '✗'}")
    print()

    # Test 3: Additional verification with different seed
    print("=== Test 3: Additional Same Seed Verification ===")
    additional_same_seed_results = test_same_seed_reproducibility(
        123, num_runs=3)

    print("Additional Same Seed Results:")
    print(
        f"  Nodes: {'✓' if additional_same_seed_results['nodes_identical'] else '✗'}")
    print(
        f"  Edges: {'✓' if additional_same_seed_results['edges_identical'] else '✗'}")
    print()

    # Print hashes for seed 123
    print("Hashes for seed 123:")
    for i, hashes in enumerate(additional_same_seed_results['hashes']):
        print(f"  Run {i+1}:")
        print(f"    Nodes:    {hashes['nodes']}")
        print(f"    Edges:    {hashes['edges']}")
    print()

    # Overall assessment
    all_reproducible = (same_seed_results['all_identical'] and
                        additional_same_seed_results['all_identical'])

    seeds_produce_variation = different_seed_results['any_different']

    print("=== H4 Results ===")
    if all_reproducible and seeds_produce_variation:
        print("✓ Tide produces fully reproducible outputs with same seeds")
        print("✓ Different seeds produce different outputs as expected")
        print("✓ Deterministic behavior confirmed")
    else:
        print("✗ Reproducibility issues detected:")
        if not all_reproducible:
            print("  - Same seeds do not produce identical outputs")
        if not seeds_produce_variation:
            print("  - Different seeds produce identical outputs (unexpected)")

    # Print hash summary for debugging
    if not all_reproducible:
        print("\nHash Summary for Debugging:")
        print("Seed 42 runs:")
        for i, hashes in enumerate(same_seed_results['hashes']):
            print(f"  Run {i+1}:")
            print(f"    Nodes: {hashes['nodes'][:16]}...")
            print(f"    Edges: {hashes['edges'][:16]}...")

    return all_reproducible and seeds_produce_variation


if __name__ == "__main__":
    success = run_h4_experiment()
    sys.exit(0 if success else 1)
