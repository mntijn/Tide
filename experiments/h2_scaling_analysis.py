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

# Apply Tufte-inspired visualization guidelines
golden_ratio = 1.618
figure_size = (7, 7 / golden_ratio)

# Configure matplotlib and seaborn for accessibility and a clean, Tufte-like aesthetic
sns.set_context("notebook")
sns.set_style("ticks")  # "ticks" style is a good base for Tufte
sns.set_palette("colorblind", color_codes=True)

# Set white background and other Tufte-friendly parameters inspired by the article
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.figsize"] = figure_size
plt.rcParams["axes.labelcolor"] = "black"
plt.rcParams["xtick.color"] = "black"
plt.rcParams["ytick.color"] = "black"
plt.rcParams["text.color"] = "black"
# From the article: increase font sizes for legibility
plt.rcParams["font.size"] = 20
plt.rcParams["axes.labelsize"] = 22
plt.rcParams["axes.titlesize"] = 24
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 18


def power_law_model(N, c, alpha):
    """Power-law model: T(N) = c * N^alpha"""
    return c * (N ** alpha)


def calculate_scaling_exponent(graph_sizes, values):
    """Calculate scaling exponent using log-log linear regression"""
    if len(graph_sizes) != len(values) or len(graph_sizes) < 2:
        return None, None, None

    # Convert to numpy arrays and take logarithms
    log_N = np.log(np.array(graph_sizes))
    log_T = np.log(np.array(values))

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


def create_time_scaling_plot(configurations, total_elements, times, time_fit_params):
    """Create Tufte-inspired time scaling analysis plot (log-log and per-element only)."""
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('H2 Time Scalability Analysis', fontsize=26,
                 fontweight='bold', color='black', y=1.02)

    # Plot 1: Time Log-log scale
    ax2.loglog(total_elements, times, 'o', markersize=8,
               color='#D81B60', alpha=0.7, mec='w')
    ax2.set_xlabel('Total Elements (N)')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Time Scaling (Log-Log Scale)')
    ax2.set_xlim(left=min(total_elements) * 0.9,
                 right=max(total_elements) * 1.1)
    ax2.set_ylim(bottom=min(times) * 0.9, top=max(times) * 1.5)
    sns.despine(ax=ax2)
    for i, (x, y, config) in enumerate(zip(total_elements, times, configurations)):
        ax2.annotate(config, (x, y), xytext=(
            8, -4), textcoords='offset points', fontsize=14, ha='left', va='center')
    if time_fit_params:
        alpha, c, r_squared = time_fit_params
        N_fit = np.logspace(np.log10(min(total_elements)),
                            np.log10(max(total_elements)), 100)
        T_fit = power_law_model(N_fit, c, alpha)
        ax2.loglog(N_fit, T_fit, '--', color='#004D40', linewidth=2.5)
        fit_text = f'T ≈ {c:.2e} * N^{alpha:.3f} (R²={r_squared:.3f})'
        ax2.text(0.95, 0.05, fit_text, transform=ax2.transAxes, fontsize=16,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.7, ec='none'))

        # Add O(N) reference line, anchored to the first data point
        c_linear = times[0] / total_elements[0]
        T_linear = c_linear * N_fit
        ax2.loglog(N_fit, T_linear, ':', color='grey', linewidth=2.0)
        # Position label on the visible part of the line
        ax2.text(N_fit[-5], T_linear[-5], ' O(N)', fontsize=14, color='grey',
                 ha='left', va='center')

    # Plot 2: Time per element
    time_per_element = np.array(times) / np.array(total_elements)
    ax3.plot(total_elements, time_per_element * 1e6, 'o-', markersize=7, color='#D81B60',
             linewidth=1.5, alpha=0.7, mfc='w', mec='#D81B60')
    ax3.set_xlabel('Total Elements (N)')
    ax3.set_ylabel('Time per Element (μs)')
    ax3.set_title('Cost per Element (Time)')
    ax3.set_xlim(left=0, right=max(total_elements) * 1.05)
    ax3.set_ylim(bottom=0, top=max(time_per_element * 1e6) * 1.05)
    sns.despine(ax=ax3)
    for i, (x, y, config) in enumerate(zip(total_elements, time_per_element * 1e6, configurations)):
        ax3.annotate(config, (x, y), xytext=(
            0, -10), textcoords='offset points', fontsize=14, ha='center', va='top')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def create_memory_scaling_plot(configurations, total_elements, memory_usages, memory_fit_params):
    """Create a Tufte-inspired memory scaling analysis plot on a log-log scale."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle('H2 Peak Memory Scalability',
                 fontsize=30, fontweight='bold', color='black', y=1.02)

    # Plot 1: Memory Log-log scale
    ax.loglog(total_elements, memory_usages, 'o', markersize=8,
              color='#1E88E5', alpha=0.7, mec='w')
    ax.set_xlabel('Total Graph Size (N)', fontsize=26)
    ax.set_ylabel('Peak Memory (MB)', fontsize=26)
    ax.set_title('Peak Memory vs. Graph Size (Log-Log Scale)', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.set_xlim(left=min(total_elements) * 0.9,
                right=max(total_elements) * 1.1)
    ax.set_ylim(bottom=min(memory_usages) * 0.9, top=max(memory_usages) * 1.5)
    sns.despine(ax=ax)
    for i, (x, y, config) in enumerate(zip(total_elements, memory_usages, configurations)):
        ax.annotate(config, (x, y), xytext=(
            8, -4), textcoords='offset points', fontsize=18, ha='left', va='center')
    if memory_fit_params:
        alpha_mem, c_mem, r_squared_mem = memory_fit_params
        N_fit = np.logspace(np.log10(min(total_elements)),
                            np.log10(max(total_elements)), 100)
        Mem_fit = power_law_model(N_fit, c_mem, alpha_mem)
        ax.loglog(N_fit, Mem_fit, '--', color='#004D40', linewidth=2.5)
        fit_text = f'M ≈ {c_mem:.2e} * N^{alpha_mem:.3f} (R²={r_squared_mem:.3f})'
        ax.text(0.95, 0.05, fit_text, transform=ax.transAxes, fontsize=20,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.7, ec='none'))

        # Add O(N) reference line, anchored to the first data point
        c_mem_linear = memory_usages[0] / total_elements[0]
        Mem_linear = c_mem_linear * N_fit
        ax.loglog(N_fit, Mem_linear, ':', color='grey', linewidth=2.0)
        # Position label on the visible part of the line
        y_top = max(memory_usages) * 1.5
        visible_indices = np.where(Mem_linear <= y_top)[0]
        label_index = visible_indices[-1] if len(
            visible_indices) > 0 else len(Mem_linear) - 1
        ax.text(N_fit[label_index], Mem_linear[label_index], ' O(N)', fontsize=18, color='grey',
                ha='left', va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def analyze_h2_results():
    """Analyze H2 scalability results from the provided table"""

    print("=== H2 Scalability Analysis ===")
    print("Analyzing results from the experiment table\n")

    # Data from the table
    configurations = ["Test-1k", "Test-2k",
                      "Test-3.5k", "Test-5k", "Test-8k", "Test-12k"]
    total_elements = [377003, 950195, 2014045, 3351314, 6029762, 10245234]
    times = [1.84, 4.86, 11.26, 24.80, 79.23, 241.18]  # seconds
    memory_usages = [459.7, 711.0, 1027.2, 1143.3, 1375.9, 2292.0]  # MB

    # Print raw data
    print("Raw Data:")
    for i, config in enumerate(configurations):
        print(
            f"  {config}: N={total_elements[i]:,}, T={times[i]:.2f}s, Mem={memory_usages[i]:.1f}MB")
    print()

    # Calculate power-law scaling exponent for time
    alpha, c, r_squared = calculate_scaling_exponent(total_elements, times)

    print("=== Time Scaling Analysis (Power-Law) ===")
    if alpha is not None:
        print(f"Power-law Model: T(N) = c × N^α")
        print(f"  Scaling Exponent (α): {alpha:.4f}")
        print(f"  Coefficient (c): {c:.4e}")
        print(f"  Log-log R²: {r_squared:.4f}")

        # Determine complexity class
        if alpha < 1.2:
            complexity_class = "Linear or sub-linear"
        elif alpha < 1.8:
            complexity_class = "Near-linear"
        elif alpha < 2.2:
            complexity_class = "Quadratic or near-quadratic"
        else:
            complexity_class = "Super-quadratic"
        print(f"  Time Complexity Class: {complexity_class}")
        print()
    else:
        print("  Could not calculate time scaling exponent")
        print()

    # Calculate power-law scaling exponent for memory
    alpha_mem, c_mem, r_squared_mem = calculate_scaling_exponent(
        total_elements, memory_usages)

    print("=== Memory Scaling Analysis (Power-Law) ===")
    if alpha_mem is not None:
        print(f"Power-law Model: M(N) = c × N^α")
        print(f"  Scaling Exponent (α_mem): {alpha_mem:.4f}")
        print(f"  Coefficient (c_mem): {c_mem:.4e}")
        print(f"  Log-log R²: {r_squared_mem:.4f}")

        # Determine complexity class
        if alpha_mem < 0.8:
            complexity_class_mem = "Sub-linear"
        elif alpha_mem < 1.2:
            complexity_class_mem = "Linear"
        else:
            complexity_class_mem = "Super-linear"
        print(f"  Memory Complexity Class: {complexity_class_mem}")
        if r_squared_mem < 0.9:
            print(
                "  Note: Low R² value suggests the power-law model may not be a good fit for memory.")
        print()
    else:
        print("  Could not calculate memory scaling exponent")
        print()

    # Calculate Pearson correlation for time
    correlation, p_value = calculate_pearson_correlation(total_elements, times)

    print("=== Time Correlation Analysis ===")
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
        print("  Could not calculate correlation for time")
        print()

    # Calculate Pearson correlation for memory
    correlation_mem, p_value_mem = calculate_pearson_correlation(
        total_elements, memory_usages)
    print("=== Memory Correlation Analysis ===")
    if correlation_mem is not None:
        print(f"Pearson Correlation (r): {correlation_mem:.4f}")
        if p_value_mem is not None:
            print(f"  P-value: {p_value_mem:.4e}")

        if abs(correlation_mem) > 0.9:
            strength_mem = "Very strong"
        elif abs(correlation_mem) > 0.7:
            strength_mem = "Strong"
        else:
            strength_mem = "Weak"
        print(f"  Correlation Strength: {strength_mem}")
        print()
    else:
        print("  Could not calculate correlation for memory")
        print()

    # Calculate scaling efficiency metrics
    print("=== Scaling Efficiency Analysis ===")

    # Time per element analysis
    time_per_element = np.array(times) / np.array(total_elements)
    print("Time per Element (microseconds):")
    for i, (config, tpe) in enumerate(zip(configurations, time_per_element)):
        print(f"  {config}: {tpe * 1e6:.2f} μs/element")

    efficiency_trend_time = "Increasing" if time_per_element[-1] > time_per_element[0] else "Decreasing"
    print(f"  Trend: {efficiency_trend_time} (time efficiency {'decreasing' if efficiency_trend_time == 'Increasing' else 'improving'} with scale)")
    print()

    # Memory per element analysis
    mem_per_element_all = (np.array(memory_usages) * 1024 *
                           1024) / np.array(total_elements)
    print("Memory per Element (bytes):")
    for i, (config, mpe) in enumerate(zip(configurations, mem_per_element_all)):
        print(f"  {config}: {mpe:.2f} bytes/element")

    # We check the trend on the clean data
    efficiency_trend_mem = "Increasing" if mem_per_element_all[-1] > mem_per_element_all[0] else "Decreasing"
    print(
        f"  Trend: {efficiency_trend_mem} (memory efficiency {'decreasing' if efficiency_trend_mem == 'Increasing' else 'improving'} with scale)")
    print()

    # Scaling ratios between consecutive tests
    if len(total_elements) > 1:
        print("Scaling Ratios (consecutive tests):")
        for i in range(1, len(configurations)):
            size_ratio = total_elements[i] / total_elements[i-1]
            time_ratio = times[i] / times[i-1]
            mem_ratio = memory_usages[i] / memory_usages[i-1]

            print(f"  {configurations[i-1]} → {configurations[i]}:")
            print(f"    Size increase: {size_ratio:.2f}x")
            print(f"    Time increase: {time_ratio:.2f}x")
            print(f"    Memory increase: {mem_ratio:.2f}x")
        print()

    # Overall assessment
    print("=== Assessment ===")

    issues = []
    good_scaling = True

    if alpha is not None:
        if alpha > 1.8:
            good_scaling = False
            issues.append(
                f"High time scaling exponent (α={alpha:.3f}) indicates poor time complexity")
        elif alpha > 1.5:
            issues.append(
                f"Time scaling exponent (α={alpha:.3f}) suggests super-linear complexity")

    if efficiency_trend_time == "Increasing":
        issues.append(
            "Time per element increasing with scale (time efficiency decreasing)")

    if alpha_mem is not None and alpha_mem > 1.2:
        issues.append(
            f"Memory usage scaling super-linearly (α_mem={alpha_mem:.3f})")

    if r_squared is not None and r_squared < 0.95:
        issues.append(
            f"Low R² for time ({r_squared:.3f}) indicates inconsistent scaling behavior")

    if r_squared_mem is not None and r_squared_mem < 0.9:
        issues.append(
            f"Very low R² for memory ({r_squared_mem:.3f}) suggests unpredictable memory scaling")

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

    # Time scaling plot
    time_fig = create_time_scaling_plot(
        configurations, total_elements, times, (alpha, c, r_squared)
    )
    time_output_file = Path(__file__).parent / "h2_time_scaling_analysis.png"
    time_fig.savefig(time_output_file, dpi=300,
                     bbox_inches='tight', facecolor='white')
    print(f"Time scaling analysis plot saved to: {time_output_file}")

    # Memory scaling plot
    mem_fig = create_memory_scaling_plot(
        configurations, total_elements, memory_usages, (
            alpha_mem, c_mem, r_squared_mem)
    )
    mem_output_file = Path(__file__).parent / "h2_memory_scaling_analysis.png"
    mem_fig.savefig(mem_output_file, dpi=300,
                    bbox_inches='tight', facecolor='white')
    print(f"Memory scaling analysis plot saved to: {mem_output_file}")

    # Show the plot
    plt.show()

    return {
        'scaling_exponent_time': alpha,
        'coefficient_time': c,
        'log_log_r_squared_time': r_squared,
        'pearson_correlation_time': correlation,
        'pearson_p_value_time': p_value,
        'scaling_exponent_mem': alpha_mem,
        'coefficient_mem': c_mem,
        'log_log_r_squared_mem': r_squared_mem,
        'pearson_correlation_mem': correlation_mem,
        'pearson_p_value_mem': p_value_mem,
        'time_per_element': time_per_element,
        'mem_per_element': mem_per_element_all,
        'good_scaling': good_scaling,
        'issues': issues
    }


if __name__ == "__main__":
    results = analyze_h2_results()
