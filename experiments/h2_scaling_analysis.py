#!/usr/bin/env python3
"""
H2 Scalability Analysis Script

Analyzes scaling performance data from H2 experiment results.
Calculates Pearson correlation, power-law scaling exponent, and generates visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from scipy.stats import pearsonr
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using manual calculations.")

# Apply visualization guidelines
golden_ratio = 1.618
figure_size = (7, 7 / golden_ratio)

# Configure matplotlib and seaborn for accessibility and aesthetics
sns.set_context("notebook")
sns.set_style("ticks")
sns.set_palette("colorblind", color_codes=True)
plt.rcParams["figure.facecolor"] = "none"
plt.rcParams["axes.facecolor"] = "none"
plt.rcParams["figure.figsize"] = figure_size


def power_law_model(N, c, alpha):
    """Power-law model: T(N) = c * N^alpha"""
    return c * (N ** alpha)


def calculate_scaling_exponent(graph_sizes, times):
    """Calculate scaling exponent using log-log linear regression"""
    if len(graph_sizes) != len(times) or len(graph_sizes) < 2:
        return None, None, None

    # Convert to numpy arrays and take logarithms
    log_N = np.log(np.array(graph_sizes))
    log_T = np.log(np.array(times))

    k = len(graph_sizes)

    # Calculate scaling exponent using linear regression on log-log data
    numerator = k * np.sum(log_N * log_T) - np.sum(log_N) * np.sum(log_T)
    denominator = k * np.sum(log_N ** 2) - (np.sum(log_N)) ** 2

    if denominator == 0:
        return None, None, None

    alpha = numerator / denominator
    log_c = (np.sum(log_T) - alpha * np.sum(log_N)) / k
    c = np.exp(log_c)

    # Calculate R² for log-log regression
    predicted_log_T = log_c + alpha * log_N
    ss_res = np.sum((log_T - predicted_log_T) ** 2)
    ss_tot = np.sum((log_T - np.mean(log_T)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return alpha, c, r_squared


def calculate_pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient"""
    if SCIPY_AVAILABLE:
        correlation, p_value = pearsonr(x, y)
        return correlation, p_value
    else:
        # Manual calculation
        if len(x) != len(y) or len(x) < 2:
            return None, None

        x_arr = np.array(x)
        y_arr = np.array(y)

        x_mean = np.mean(x_arr)
        y_mean = np.mean(y_arr)

        numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
        denominator = np.sqrt(np.sum((x_arr - x_mean) ** 2)
                              * np.sum((y_arr - y_mean) ** 2))

        if denominator == 0:
            return None, None

        correlation = numerator / denominator
        return correlation, None  # p-value not calculated in manual mode


def create_scaling_plots(configurations, total_elements, times, alpha=None, c=None, r_squared=None):
    """Create scaling analysis plots following accessibility guidelines"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('H2 Scalability Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Linear scale
    ax1.scatter(total_elements, times, s=60, alpha=0.8,
                color='#D81B60', edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Total Elements (N)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Scaling Performance (Linear Scale)')
    ax1.grid(True, alpha=0.3)

    # Add configuration labels
    for i, (x, y, config) in enumerate(zip(total_elements, times, configurations)):
        ax1.annotate(config, (x, y), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    # Plot 2: Log-log scale with power-law fit
    ax2.loglog(total_elements, times, 'o', markersize=8,
               color='#1E88E5', markeredgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Total Elements (N)')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Log-Log Scale with Power-Law Fit')
    ax2.grid(True, alpha=0.3)

    # Add power-law fit if available
    if alpha is not None and c is not None:
        N_fit = np.logspace(np.log10(min(total_elements)),
                            np.log10(max(total_elements)), 100)
        T_fit = power_law_model(N_fit, c, alpha)
        ax2.loglog(N_fit, T_fit, '--', color='#FFC107', linewidth=2,
                   label=f'T = {c:.2e} × N^{alpha:.3f}\nR² = {r_squared:.3f}')
        ax2.legend()

    # Plot 3: Scaling ratios
    if len(total_elements) > 1:
        size_ratios = [total_elements[i] / total_elements[i-1]
                       for i in range(1, len(total_elements))]
        time_ratios = [times[i] / times[i-1] for i in range(1, len(times))]
        transition_labels = [
            f"{configurations[i-1]}\n→\n{configurations[i]}" for i in range(1, len(configurations))]

        x_pos = np.arange(len(transition_labels))
        width = 0.35

        bars1 = ax3.bar(x_pos - width/2, size_ratios, width,
                        label='Size Ratio', color='#004D40', alpha=0.8)
        bars2 = ax3.bar(x_pos + width/2, time_ratios, width,
                        label='Time Ratio', color='#D81B60', alpha=0.8)

        ax3.set_xlabel('Scale Transition')
        ax3.set_ylabel('Ratio')
        ax3.set_title('Scaling Ratios Between Consecutive Tests')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(transition_labels, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}x', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}x', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # Plot 4: Time complexity per element
    time_per_element = np.array(times) / np.array(total_elements)
    ax4.plot(total_elements, time_per_element * 1e6, 'o-',
             markersize=6, color='#1E88E5', linewidth=2)
    ax4.set_xlabel('Total Elements (N)')
    ax4.set_ylabel('Time per Element (microseconds)')
    ax4.set_title('Time Complexity per Element')
    ax4.grid(True, alpha=0.3)

    # Add configuration labels
    for i, (x, y, config) in enumerate(zip(total_elements, time_per_element * 1e6, configurations)):
        ax4.annotate(config, (x, y), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    plt.tight_layout()
    return fig


def analyze_h2_results():
    """Analyze H2 scalability results from the provided table"""

    print("=== H2 Scalability Analysis ===")
    print("Analyzing results from the experiment table\n")

    # Data from the table
    configurations = ["Test-1k", "Test-2k",
                      "Test-3.5k", "Test-5k", "Test-8k", "Test-12k"]
    total_nodes = [4730, 9460, 16581, 23662, 37231, 56169]
    total_transactions = [375930, 940743, 1997464, 3327628, 5992458, 10202036]
    total_elements = [377004, 950203, 2014045, 3351290,
                      6029689, 10289250]  # N = nodes + transactions
    times = [5.74, 14.27, 35.85, 98.89, 406.68, 2336.81]  # seconds

    # Print raw data
    print("Raw Data:")
    for i, config in enumerate(configurations):
        print(f"  {config}: N={total_elements[i]:,}, T={times[i]:.2f}s")
    print()

    # Calculate power-law scaling exponent
    alpha, c, r_squared = calculate_scaling_exponent(total_elements, times)

    print("=== Power-Law Scaling Analysis ===")
    if alpha is not None:
        print(f"Power-law Model: T(N) = c × N^α")
        print(f"  Scaling Exponent (α): {alpha:.4f}")
        print(f"  Coefficient (c): {c:.4e}")
        print(f"  Log-log R²: {r_squared:.4f}")

        # Determine complexity class
        if alpha < 1.2:
            complexity_class = "Linear or sub-linear"
        elif alpha < 2.2:
            complexity_class = "Quadratic or near-quadratic"
        else:
            complexity_class = "Super-quadratic"
        print(f"  Complexity Class: {complexity_class}")
        print()
    else:
        print("  Could not calculate scaling exponent")
        print()

    # Calculate Pearson correlation
    correlation, p_value = calculate_pearson_correlation(total_elements, times)

    print("=== Correlation Analysis ===")
    if correlation is not None:
        print(f"Pearson Correlation (r): {correlation:.4f}")
        if p_value is not None:
            print(f"  P-value: {p_value:.4e}")

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
    else:
        print("  Could not calculate correlation")
        print()

    # Calculate scaling efficiency metrics
    print("=== Scaling Efficiency Analysis ===")

    # Time per element analysis
    time_per_element = np.array(times) / np.array(total_elements)
    print("Time per Element (microseconds):")
    for i, (config, tpe) in enumerate(zip(configurations, time_per_element)):
        print(f"  {config}: {tpe * 1e6:.2f} μs/element")

    efficiency_trend = "Increasing" if time_per_element[-1] > time_per_element[0] else "Decreasing"
    print(f"  Trend: {efficiency_trend} (efficiency {'decreasing' if efficiency_trend == 'Increasing' else 'improving'} with scale)")
    print()

    # Scaling ratios between consecutive tests
    if len(total_elements) > 1:
        print("Scaling Ratios (consecutive tests):")
        for i in range(1, len(configurations)):
            size_ratio = total_elements[i] / total_elements[i-1]
            time_ratio = times[i] / times[i-1]
            scaling_factor = time_ratio / size_ratio

            print(f"  {configurations[i-1]} → {configurations[i]}:")
            print(f"    Size increase: {size_ratio:.2f}x")
            print(f"    Time increase: {time_ratio:.2f}x")
            print(f"    Scaling factor: {scaling_factor:.2f}")
        print()

    # Overall assessment
    print("=== Assessment ===")

    issues = []
    good_scaling = True

    if alpha is not None:
        if alpha > 2.5:
            good_scaling = False
            issues.append(
                f"High scaling exponent (α={alpha:.3f}) indicates poor time complexity")
        elif alpha > 2.0:
            issues.append(
                f"Scaling exponent (α={alpha:.3f}) suggests quadratic or worse complexity")

    if r_squared is not None and r_squared < 0.8:
        issues.append(
            f"Low R² ({r_squared:.3f}) indicates inconsistent scaling behavior")

    if efficiency_trend == "Increasing":
        issues.append(
            "Time per element increasing with scale (efficiency decreasing)")

    if correlation is not None and abs(correlation) < 0.7:
        issues.append(
            f"Weak correlation (r={correlation:.3f}) between size and time")

    if not issues:
        print("✓ Excellent scaling performance observed")
        if alpha is not None:
            print(f"✓ Reasonable time complexity (α={alpha:.3f})")
        if correlation is not None:
            print(
                f"✓ Strong correlation between size and time (r={correlation:.3f})")
    else:
        print("⚠ Scaling analysis results:")
        for issue in issues:
            print(f"  • {issue}")

    # Create visualizations
    print("\n=== Generating Visualizations ===")
    fig = create_scaling_plots(
        configurations, total_elements, times, alpha, c, r_squared)

    # Save the plot
    output_file = Path(__file__).parent / "h2_scaling_analysis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Scaling analysis plot saved to: {output_file}")

    # Show the plot
    plt.show()

    return {
        'scaling_exponent': alpha,
        'coefficient': c,
        'log_log_r_squared': r_squared,
        'pearson_correlation': correlation,
        'pearson_p_value': p_value,
        'time_per_element': time_per_element,
        'efficiency_trend': efficiency_trend,
        'good_scaling': good_scaling,
        'issues': issues
    }


if __name__ == "__main__":
    results = analyze_h2_results()
