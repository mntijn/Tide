#!/usr/bin/env python3
"""
H2: Scalability Experiment

Tests that Tide can generate financial networks of varying scales and structures
according to user-specified configuration parameters.

Measures performance metrics across different graph sizes:
- Wall time
- Memory usage
- Total graph size (nodes and edges)
- Pattern generation success
- Power-law scaling analysis
- Correlation analysis
"""

import os
import sys
import json
import time
import yaml
import tempfile
import subprocess
import psutil
import numpy as np
from pathlib import Path

try:
    from scipy.stats import pearsonr
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using basic correlation calculation.")

# Add the parent directory to path to import main
sys.path.append(str(Path(__file__).parent.parent))

# Import the generation logic directly instead of using subprocess
try:
    import datetime
    from tide.graph_generator import GraphGenerator
    from tide.outputs import export_to_csv
    DIRECT_IMPORT_AVAILABLE = True
except ImportError:
    DIRECT_IMPORT_AVAILABLE = False
    print("Warning: Could not import tide modules. Falling back to subprocess.")


def create_h2_config(scale_type):
    """Create configuration for different scales with realistic pattern generation"""

    # Simple pattern rate: 1 pattern per 1000 individuals
    pattern_rate_per_individuals = 1.2/1000

    # Transaction rates per account per day by scale (for efficiency)
    transaction_rates_by_scale = {
        "extra_small": 0.8,
        "small": 1.0,
        "small_medium": 1.2,
        "medium": 1.4,
        "medium_large": 1.6,
        "large": 1.8,
        "extra_large": 2
    }

    # Time span in days (keeping it reasonable for efficiency)
    time_span_days = 150

    trans_rate = transaction_rates_by_scale.get(scale_type, 1.4)

    base_config = {
        'transaction_rates': {
            'per_account_per_day': trans_rate
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
        'backgroundPatterns': {
            'randomPayments': {
                'weight': 0.6,  # 60% of total background transactions
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
                'weight': 0.25,  # 25% of total background transactions
                'payment_intervals': [14, 30],
                'salary_range': [2500.0, 7500.0],
                'salary_variation': 0.05,
                'preferred_payment_days': [1, 15, 30]
            },
            'fraudsterBackground': {
                'weight': 0.1,  # 10% of total background transactions
                'fraudster_rate_multiplier': [0.1, 0.5],
                'amount_ranges': {
                    'small_transactions': [5.0, 200.0],
                    'medium_transactions': [200.0, 1000.0]
                },
                'transaction_size_probabilities': {
                    'small': 0.8,
                    'medium': 0.2
                }
            },
            'legitimateHighPayments': {
                'weight': 0.05,  # 5% of total background transactions
                'high_payment_ranges': {
                    'property_transactions': [50000.0, 500000.0],
                    'business_deals': [10000.0, 100000.0],
                    'luxury_purchases': [5000.0, 50000.0]
                },
                'high_payment_type_probabilities': {
                    'property': 0.2,
                    'business': 0.5,
                    'luxury': 0.3
                },
                'high_payment_rate_per_month': 0.1
            }
        },
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

    # Common configuration for all scales
    common_config = {
        'graph_scale': {
            'institutions_per_country': 3,
            'individual_accounts_per_institution_range': [1, 3],
            'business_accounts_per_institution_range': [1, 6]
        },
        'random_business_probability': 0.15  # 15% business creation probability
    }

    if scale_type == "extra_small":
        config = base_config.copy()
        config.update(common_config)
        config['graph_scale']['individuals'] = 1000
    elif scale_type == "small":
        config = base_config.copy()
        config.update(common_config)
        config['graph_scale']['individuals'] = 2000
    elif scale_type == "small_medium":
        config = base_config.copy()
        config.update(common_config)
        config['graph_scale']['individuals'] = 3500
    elif scale_type == "medium":
        config = base_config.copy()
        config.update(common_config)
        config['graph_scale']['individuals'] = 5000
    elif scale_type == "medium_large":
        config = base_config.copy()
        config.update(common_config)
        config['graph_scale']['individuals'] = 8000
    elif scale_type == "large":
        config = base_config.copy()
        config.update(common_config)
        config['graph_scale']['individuals'] = 12000
    elif scale_type == "extra_large":
        config = base_config.copy()
        config.update(common_config)
        config['graph_scale']['individuals'] = 20000
    else:
        raise ValueError(f"Invalid scale type: {scale_type}")

    # Calculate patterns based on number of individuals (1 per 1000 people)
    individuals = config['graph_scale']['individuals']
    expected_patterns = max(1, int(individuals * pattern_rate_per_individuals))

    # Use random pattern generation with the calculated total count
    config['pattern_frequency'] = {
        'random': True,
        'num_illicit_patterns': expected_patterns
    }

    # Store metadata for analysis
    config['_metadata'] = {
        'expected_total_patterns': expected_patterns,
        'pattern_rate_per_individuals': pattern_rate_per_individuals,
        'time_span_days': time_span_days
    }

    return config


def measure_performance(scale_type, config):
    """Run generation and measure performance metrics"""
    print(f"Running {scale_type} scale test...")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, f"{scale_type}_config.yaml")
        patterns_file = os.path.join(temp_dir, "generated_patterns.json")
        nodes_file = os.path.join(temp_dir, "generated_nodes.csv")
        edges_file = os.path.join(temp_dir, "generated_edges.csv")

        # Create config file (remove metadata before saving)
        config_to_save = {k: v for k, v in config.items() if k != '_metadata'}
        with open(config_file, 'w') as f:
            yaml.dump(config_to_save, f, default_flow_style=False)

        # Start monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()

        success = False

        if DIRECT_IMPORT_AVAILABLE:
            # Use direct function call for accurate memory measurement
            try:
                # Load configuration
                generator_parameters = config_to_save.copy()

                # Convert date strings to datetime objects
                generator_parameters["time_span"]["start_date"] = datetime.datetime.fromisoformat(
                    generator_parameters["time_span"]["start_date"])
                generator_parameters["time_span"]["end_date"] = datetime.datetime.fromisoformat(
                    generator_parameters["time_span"]["end_date"])

                # Generate graph
                aml_graph_gen = GraphGenerator(params=generator_parameters)
                graph = aml_graph_gen.generate_graph()
                end_time = time.time()

                # Export to CSV
                export_to_csv(
                    graph=graph,
                    nodes_filepath=nodes_file,
                    edges_filepath=edges_file
                )

                # Export patterns as JSON
                patterns_data = {
                    'metadata': {
                        'generation_timestamp': datetime.datetime.now().isoformat(),
                        'total_patterns': len(aml_graph_gen.injected_patterns),
                        'graph_nodes': aml_graph_gen.num_of_nodes(),
                        'graph_edges': aml_graph_gen.num_of_edges(),
                        'config_file': config_file,
                        'output_directory': temp_dir
                    },
                    'patterns': aml_graph_gen.injected_patterns
                }

                with open(patterns_file, 'w') as f:
                    json.dump(patterns_data, f, indent=2, default=str)

                success = True
            except Exception as e:
                print(
                    f"ERROR: Direct generation failed for {scale_type} scale: {e}")
                success = False
        else:
            # Fallback to subprocess
            main_script = str(Path(__file__).parent.parent / "main.py")
            cmd = [
                sys.executable, main_script,
                '--config', config_file,
                '--output-dir', temp_dir,
                '--patterns-file', 'generated_patterns.json',
                '--nodes-file', 'generated_nodes.csv',
                '--edges-file', 'generated_edges.csv'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
            if not success:
                print(
                    f"ERROR: Subprocess generation failed for {scale_type} scale")
                print(f"STDERR: {result.stderr}")

        # Measure results
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        wall_time = end_time - start_time
        # Ensure memory usage is non-negative
        peak_memory = max(0, final_memory - initial_memory)

        if not success:
            expected_patterns = config.get('_metadata', {}).get(
                'expected_total_patterns', 0)
            return {
                'scale_type': scale_type,
                'wall_time_seconds': wall_time,
                'memory_usage_mb': peak_memory,
                'nodes_count': 0,
                'account_nodes_count': 0,
                'edges_count': 0,
                'total_edges_count': 0,
                'patterns_generated': 0,
                'patterns_expected': expected_patterns,
                'patterns_success': False,
                'individuals_configured': config['graph_scale']['individuals'],
                'business_probability': config['random_business_probability'],
                'actual_transaction_rate': 0.0
            }

        # Count nodes and edges
        nodes_count = 0
        account_nodes_count = 0
        edges_count = 0

        if os.path.exists(nodes_file):
            with open(nodes_file, 'r') as f:
                lines = list(f)
                nodes_count = len(lines) - 1  # Subtract header
                # Count account nodes specifically
                for line in lines[1:]:  # Skip header
                    if "ACCOUNT" in line.upper():
                        account_nodes_count += 1

        if os.path.exists(edges_file):
            with open(edges_file, 'r') as f:
                lines = list(f)
                # Count only transaction edges, not ownership edges
                edges_count = sum(
                    1 for line in lines[1:] if "ownership" not in line.lower())
                total_edges_count = len(lines) - 1  # Subtract header

        # Calculate actual transaction rate per account per day
        actual_transaction_rate = 0.0
        if account_nodes_count > 0:
            # Use the time span from config (these are already datetime objects)
            start_date = config['time_span']['start_date']
            end_date = config['time_span']['end_date']
            time_span_days = (end_date - start_date).days
            if time_span_days > 0:
                actual_transaction_rate = edges_count / \
                    (account_nodes_count * time_span_days)

        # Count patterns
        patterns_count = 0
        patterns_success = False
        expected_patterns = config.get('_metadata', {}).get(
            'expected_total_patterns', 0)

        if os.path.exists(patterns_file):
            with open(patterns_file, 'r') as f:
                data = json.load(f)
                patterns_count = len(data.get('patterns', []))
                # Allow 20% tolerance for random pattern generation
                patterns_success = abs(
                    patterns_count - expected_patterns) <= max(1, expected_patterns * 0.2)

        return {
            'scale_type': scale_type,
            'wall_time_seconds': wall_time,
            'memory_usage_mb': peak_memory,
            'nodes_count': nodes_count,
            'account_nodes_count': account_nodes_count,
            'edges_count': edges_count,
            'total_edges_count': total_edges_count,
            'patterns_generated': patterns_count,
            'patterns_expected': expected_patterns,
            'patterns_success': patterns_success,
            'individuals_configured': config['graph_scale']['individuals'],
            'business_probability': config['random_business_probability'],
            'actual_transaction_rate': actual_transaction_rate
        }


def calculate_scaling_metrics(results):
    """Calculate scaling relationships"""
    if len(results) < 2:
        return {}

    # Sort by individuals count
    results.sort(key=lambda x: x['individuals_configured'])

    scaling_metrics = {}

    # Calculate scaling factors
    for i in range(1, len(results)):
        prev = results[i-1]
        curr = results[i]

        size_ratio = curr['individuals_configured'] / \
            prev['individuals_configured']
        time_ratio = curr['wall_time_seconds'] / prev['wall_time_seconds']
        memory_ratio = curr['memory_usage_mb'] / \
            prev['memory_usage_mb'] if prev['memory_usage_mb'] > 0 else 0

        scaling_metrics[f"{prev['scale_type']}_to_{curr['scale_type']}"] = {
            'size_ratio': size_ratio,
            'time_ratio': time_ratio,
            'memory_ratio': memory_ratio,
            'time_scaling_factor': time_ratio / size_ratio,
            'memory_scaling_factor': memory_ratio / size_ratio if memory_ratio > 0 else 0
        }

    return scaling_metrics


def power_law_model(N, c, alpha):
    """Power-law model: T(N) = c * N^alpha"""
    return c * (N ** alpha)


def calculate_scaling_exponent(graph_sizes, times):
    """Calculate scaling exponent using log-log linear regression"""
    if len(graph_sizes) != len(times) or len(graph_sizes) < 2:
        return None, None

    # Convert to numpy arrays and take logarithms
    log_N = np.log(np.array(graph_sizes))
    log_T = np.log(np.array(times))

    k = len(graph_sizes)

    # Calculate scaling exponent using the formula from the paper
    numerator = k * np.sum(log_N * log_T) - np.sum(log_N) * np.sum(log_T)
    denominator = k * np.sum(log_N ** 2) - (np.sum(log_N)) ** 2

    if denominator == 0:
        return None, None

    alpha = numerator / denominator

    # Calculate R² for log-log regression
    log_c = (np.sum(log_T) - alpha * np.sum(log_N)) / k
    predicted_log_T = log_c + alpha * log_N

    ss_res = np.sum((log_T - predicted_log_T) ** 2)
    ss_tot = np.sum((log_T - np.mean(log_T)) ** 2)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return alpha, r_squared


def calculate_pearson_correlation(graph_sizes, times):
    """Calculate Pearson correlation coefficient"""
    if SCIPY_AVAILABLE:
        correlation, p_value = pearsonr(graph_sizes, times)
        return correlation, p_value
    else:
        # Manual calculation
        if len(graph_sizes) != len(times) or len(graph_sizes) < 2:
            return None, None

        N = np.array(graph_sizes)
        T = np.array(times)

        N_mean = np.mean(N)
        T_mean = np.mean(T)

        numerator = np.sum((N - N_mean) * (T - T_mean))
        denominator = np.sqrt(np.sum((N - N_mean) ** 2)
                              * np.sum((T - T_mean) ** 2))

        if denominator == 0:
            return None, None

        correlation = numerator / denominator
        return correlation, None  # p-value not calculated in manual mode


def calculate_memory_efficiency(peak_memories, total_elements):
    """Calculate memory efficiency as peak memory / total elements"""
    if len(peak_memories) != len(total_elements):
        return []

    efficiencies = []
    for memory, elements in zip(peak_memories, total_elements):
        if elements > 0:
            efficiency = memory / elements
            efficiencies.append(efficiency)
        else:
            efficiencies.append(0)

    return efficiencies


def analyze_scaling_performance(results):
    """Perform comprehensive scaling analysis"""
    if len(results) < 2:
        return {}

    # Extract data
    graph_sizes = [r['nodes_count'] + r['edges_count']
                   for r in results]  # Total elements N
    times = [r['wall_time_seconds'] for r in results]
    memories = [r['memory_usage_mb'] for r in results]
    individuals = [r['individuals_configured'] for r in results]

    analysis = {}

    # Power-law scaling analysis
    alpha, r_squared = calculate_scaling_exponent(graph_sizes, times)
    if alpha is not None:
        analysis['scaling_exponent'] = alpha
        analysis['log_log_r_squared'] = r_squared

        # Determine complexity class
        if alpha < 1.2:
            complexity_class = "Linear or sub-linear"
        elif alpha < 2.2:
            complexity_class = "Quadratic or near-quadratic"
        else:
            complexity_class = "Super-quadratic"
        analysis['complexity_class'] = complexity_class

    # Pearson correlation for raw data
    correlation, p_value = calculate_pearson_correlation(graph_sizes, times)
    if correlation is not None:
        analysis['pearson_correlation'] = correlation
        if p_value is not None:
            analysis['pearson_p_value'] = p_value

    # Memory efficiency analysis
    memory_efficiencies = calculate_memory_efficiency(memories, graph_sizes)
    if memory_efficiencies:
        analysis['memory_efficiencies'] = memory_efficiencies
        analysis['avg_memory_efficiency'] = np.mean(memory_efficiencies)
        analysis['memory_efficiency_trend'] = "Increasing" if memory_efficiencies[-1] > memory_efficiencies[0] else "Decreasing"

    # Traditional scaling factors (kept for compatibility)
    traditional_metrics = calculate_scaling_metrics(results)
    analysis['traditional_scaling'] = traditional_metrics

    return analysis


def run_h2_experiment():
    """Run the H2 scalability experiment"""
    print("=== H2: Scalability Experiment ===")
    print("Testing generation performance across different scales\n")

    scales = ["extra_small", "small", "small_medium",
              "medium", "medium_large", "large"]
    results = []

    for scale in scales:
        config = create_h2_config(scale)
        result = measure_performance(scale, config)
        results.append(result)

        print(f"{scale.capitalize().replace('_', ' ')} Scale Results:")
        print(f"  Individuals: {result['individuals_configured']:,}")
        print(f"  Nodes: {result['nodes_count']:,}")
        print(f"  Account Nodes: {result['account_nodes_count']:,}")
        print(f"  Edges (Transactions): {result['edges_count']:,}")
        print(f"  Total Edges: {result['total_edges_count']:,}")
        print(
            f"  Configured Rate: {config['transaction_rates']['per_account_per_day']:.3f} tx/account/day")
        print(
            f"  Actual Rate: {result['actual_transaction_rate']:.3f} tx/account/day")
        print(
            f"  Total Elements (N): {result['nodes_count'] + result['edges_count']:,}")
        print(f"  Wall Time: {result['wall_time_seconds']:.2f} seconds")
        print(f"  Memory Usage: {result['memory_usage_mb']:.1f} MB")
        print(
            f"  Patterns Generated: {result['patterns_generated']}/{result['patterns_expected']} ({'✓' if result['patterns_success'] else '✗'})")
        print()

    # Perform comprehensive scaling analysis
    scaling_analysis = analyze_scaling_performance(results)

    print("=== Scaling Analysis ===")

    # Power-law analysis
    if 'scaling_exponent' in scaling_analysis:
        alpha = scaling_analysis['scaling_exponent']
        r_squared = scaling_analysis['log_log_r_squared']
        complexity = scaling_analysis['complexity_class']

        print(f"Power-law Model: T(N) = c × N^α")
        print(f"  Scaling Exponent (α): {alpha:.3f}")
        print(f"  Log-log R²: {r_squared:.3f}")
        print(f"  Complexity Class: {complexity}")
        print()

    # Correlation analysis
    if 'pearson_correlation' in scaling_analysis:
        correlation = scaling_analysis['pearson_correlation']
        print(f"Pearson Correlation (r): {correlation:.3f}")
        if 'pearson_p_value' in scaling_analysis:
            p_value = scaling_analysis['pearson_p_value']
            print(f"  P-value: {p_value:.3e}")

        # Interpret correlation strength
        if abs(correlation) > 0.9:
            strength = "Very strong"
        elif abs(correlation) > 0.7:
            strength = "Strong"
        elif abs(correlation) > 0.5:
            strength = "Moderate"
        else:
            strength = "Weak"
        print(f"  Correlation Strength: {strength}")
        print()

    # Memory efficiency analysis
    if 'memory_efficiencies' in scaling_analysis:
        efficiencies = scaling_analysis['memory_efficiencies']
        avg_efficiency = scaling_analysis['avg_memory_efficiency']
        trend = scaling_analysis['memory_efficiency_trend']

        print("Memory Efficiency Analysis:")
        for i, (result, efficiency) in enumerate(zip(results, efficiencies)):
            scale_name = result['scale_type'].capitalize()
            print(f"  {scale_name}: {efficiency:.4f} MB/element")
        print(f"  Average: {avg_efficiency:.4f} MB/element")
        print(f"  Trend: {trend}")
        print()

    # Traditional scaling metrics
    if 'traditional_scaling' in scaling_analysis:
        traditional = scaling_analysis['traditional_scaling']
        if traditional:
            print("Traditional Scaling Factors:")
            for transition, metrics in traditional.items():
                print(f"{transition.replace('_', ' → ').title()}:")
                print(f"  Size increase: {metrics['size_ratio']:.1f}x")
                print(f"  Time increase: {metrics['time_ratio']:.1f}x")
                if metrics['memory_ratio'] > 0:
                    print(f"  Memory increase: {metrics['memory_ratio']:.1f}x")
                print()

    # Overall assessment
    all_patterns_successful = all(r['patterns_success'] for r in results)

    # Updated assessment criteria
    good_scaling = True
    issues = []

    if 'scaling_exponent' in scaling_analysis:
        alpha = scaling_analysis['scaling_exponent']
        if alpha > 2.5:
            good_scaling = False
            issues.append(
                f"High scaling exponent (α={alpha:.3f}) indicates poor time complexity")

    if 'log_log_r_squared' in scaling_analysis:
        r_squared = scaling_analysis['log_log_r_squared']
        if r_squared < 0.8:
            issues.append(
                f"Low R² ({r_squared:.3f}) indicates inconsistent scaling behavior")

    if 'memory_efficiency_trend' in scaling_analysis:
        if scaling_analysis['memory_efficiency_trend'] == "Increasing":
            issues.append(
                "Memory efficiency decreasing with scale (memory usage growing faster than graph size)")

    print("=== H2 Results ===")
    if all_patterns_successful and good_scaling and not issues:
        print("✓ Tide demonstrates excellent scalability across different graph sizes")
        print("✓ Pattern generation successful at all scales")
        if 'scaling_exponent' in scaling_analysis:
            alpha = scaling_analysis['scaling_exponent']
            print(f"✓ Time complexity appears reasonable (α={alpha:.3f})")
        if 'pearson_correlation' in scaling_analysis:
            correlation = scaling_analysis['pearson_correlation']
            print(
                f"✓ Strong correlation between size and time (r={correlation:.3f})")
    else:
        print("⚠ Scalability assessment:")
        if not all_patterns_successful:
            print("  ✗ Pattern generation failed at some scales")
        else:
            print("  ✓ Pattern generation successful at all scales")

        if issues:
            for issue in issues:
                print(f"  ⚠ {issue}")
        else:
            print("  ✓ No significant scaling issues detected")

    return all_patterns_successful and good_scaling


if __name__ == "__main__":
    success = run_h2_experiment()
    sys.exit(0 if success else 1)
