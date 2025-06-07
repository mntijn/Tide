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


def create_h2_config(scale_type):
    """Create configuration for different scales"""
    base_config = {
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

    # Common configuration for all scales
    common_config = {
        'graph_scale': {
            'institutions_per_country': 3,
            'individual_accounts_per_institution_range': [1, 3],
            'business_accounts_per_institution_range': [1, 6]
        },
        'random_business_probability': 0.15,  # 15% business creation probability
        'pattern_frequency': {
            'random': False,
            'RapidFundMovement': 3,
            'FrontBusinessActivity': 2,
            'RepeatedOverseasTransfers': 3
        }
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

    return config


def measure_performance(scale_type, config):
    """Run generation and measure performance metrics"""
    print(f"Running {scale_type} scale test...")

    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, f"{scale_type}_config.yaml")
        patterns_file = os.path.join(temp_dir, "generated_patterns.json")
        nodes_file = os.path.join(temp_dir, "generated_nodes.csv")
        edges_file = os.path.join(temp_dir, "generated_edges.csv")

        # Create config file
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Start monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()

        # Run generation using subprocess
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

        # Measure results
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        wall_time = end_time - start_time
        peak_memory = final_memory - initial_memory

        if result.returncode != 0:
            print(f"ERROR: Generation failed for {scale_type} scale")
            print(f"STDERR: {result.stderr}")
            return {
                'scale_type': scale_type,
                'wall_time_seconds': wall_time,
                'memory_usage_mb': peak_memory,
                'nodes_count': 0,
                'edges_count': 0,
                'patterns_generated': 0,
                'patterns_success': False,
                'individuals_configured': config['graph_scale']['individuals'],
                'business_probability': config['random_business_probability']
            }

        # Count nodes and edges
        nodes_count = 0
        edges_count = 0

        if os.path.exists(nodes_file):
            with open(nodes_file, 'r') as f:
                nodes_count = sum(1 for line in f) - 1  # Subtract header

        if os.path.exists(edges_file):
            with open(edges_file, 'r') as f:
                edges_count = sum(1 for line in f) - 1  # Subtract header

        # Count patterns
        patterns_count = 0
        patterns_success = False
        if os.path.exists(patterns_file):
            with open(patterns_file, 'r') as f:
                data = json.load(f)
                patterns_count = len(data.get('patterns', []))
                expected_patterns = (config['pattern_frequency']['RapidFundMovement'] +
                                     config['pattern_frequency']['FrontBusinessActivity'] +
                                     config['pattern_frequency']['RepeatedOverseasTransfers'])
                patterns_success = patterns_count == expected_patterns

        return {
            'scale_type': scale_type,
            'wall_time_seconds': wall_time,
            'memory_usage_mb': peak_memory,
            'nodes_count': nodes_count,
            'edges_count': edges_count,
            'patterns_generated': patterns_count,
            'patterns_success': patterns_success,
            'individuals_configured': config['graph_scale']['individuals'],
            'business_probability': config['random_business_probability']
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
              "medium", "medium_large", "large", "extra_large"]
    results = []

    for scale in scales:
        config = create_h2_config(scale)
        result = measure_performance(scale, config)
        results.append(result)

        print(f"{scale.capitalize().replace('_', ' ')} Scale Results:")
        print(f"  Individuals: {result['individuals_configured']:,}")
        print(f"  Nodes: {result['nodes_count']:,}")
        print(f"  Edges: {result['edges_count']:,}")
        print(
            f"  Total Elements (N): {result['nodes_count'] + result['edges_count']:,}")
        print(f"  Wall Time: {result['wall_time_seconds']:.2f} seconds")
        print(f"  Memory Usage: {result['memory_usage_mb']:.1f} MB")
        print(
            f"  Patterns Generated: {result['patterns_generated']} ({'✓' if result['patterns_success'] else '✗'})")
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
