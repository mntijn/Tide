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
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 12


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
    """Create Tufte-inspired scaling analysis plots based on web article"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('H2 Scalability Analysis', fontsize=20,
                 fontweight='bold', color='black')

    # Plot 1: Linear scale (Tufte style)
    ax1.scatter(total_elements, times, s=80, alpha=0.8,
                color='#D81B60', ec='w', lw=0.5)
    ax1.set_xlabel('Total Elements (N)')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Scaling Performance (Linear Scale)')
    sns.despine(ax=ax1, trim=True)

    # Direct labeling for each point
    for i, (x, y, config) in enumerate(zip(total_elements, times, configurations)):
        ax1.annotate(config, (x, y), xytext=(
            8, 0), textcoords='offset points', fontsize=10, ha='left', va='center')

    # Plot 2: Log-log scale with power-law fit (Tufte style)
    ax2.loglog(total_elements, times, 'o', markersize=8,
               color='#D81B60', alpha=0.7, mec='w')
    ax2.set_xlabel('Total Elements (N)')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Log-Log Scale with Power-Law Fit')
    sns.despine(ax=ax2, trim=True)

    # Add power-law fit with direct labeling instead of a legend
    if alpha is not None and c is not None:
        N_fit = np.logspace(np.log10(min(total_elements)),
                            np.log10(max(total_elements)), 100)
        T_fit = power_law_model(N_fit, c, alpha)
        ax2.loglog(N_fit, T_fit, '--', color='#004D40', linewidth=2.5)
        fit_text = f'T ≈ {c:.2e} * N^{alpha:.3f} (R²={r_squared:.3f})'
        # Place text in a less obtrusive position
        ax2.text(0.95, 0.05, fit_text, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.7, ec='none'))

    # Plot 3: Scaling ratios (Tufte style)
    if len(total_elements) > 1:
        size_ratios = [total_elements[i] / total_elements[i-1]
                       for i in range(1, len(total_elements))]
        time_ratios = [times[i] / times[i-1] for i in range(1, len(times))]
        transition_labels = [f"{c1} → {c2}" for c1, c2 in zip(
            configurations[:-1], configurations[1:])]

        y_pos = np.arange(len(transition_labels))

        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(transition_labels)
        ax3.set_xlabel('Ratio Multiplier')
        ax3.set_title('Scaling Ratios (Time vs. Size)')
        sns.despine(ax=ax3, left=True, bottom=True)
        ax3.tick_params(axis='x', which='both',
                        bottom=False, labelbottom=False)

        # Create "lollipop" or "dot" plots instead of bars for a cleaner look
        # Size Ratios
        ax3.hlines(y=y_pos, xmin=1, xmax=size_ratios,
                   color='#004D40', alpha=0.4, linewidth=2)
        ax3.scatter(size_ratios, y_pos, color='#004D40', s=100,
                    label='Size Ratio', alpha=0.8, zorder=3)
        # Time Ratios
        ax3.hlines(y=y_pos, xmin=1, xmax=time_ratios,
                   color='#D81B60', alpha=0.4, linewidth=2)
        ax3.scatter(time_ratios, y_pos, color='#D81B60', s=100,
                    label='Time Ratio', alpha=0.8, zorder=3)
        ax3.axvline(x=1, color='grey', linestyle='--', linewidth=1, zorder=0)

        # Direct labeling, no legend
        for i, (sr, tr) in enumerate(zip(size_ratios, time_ratios)):
            ax3.text(sr + 0.1, i, f'{sr:.1f}x', color='#004D40',
                     ha='left', va='center', fontsize=10)
            ax3.text(tr + 0.1, i, f'{tr:.1f}x', color='#D81B60',
                     ha='left', va='center', fontsize=10)

    # Plot 4: Time complexity per element (Tufte style)
    time_per_element = np.array(times) / np.array(total_elements)
    ax4.plot(total_elements, time_per_element * 1e6, 'o-', markersize=7,
             color='#1E88E5', linewidth=1.5, alpha=0.7, mfc='w', mec='#1E88E5')
    ax4.set_xlabel('Total Elements (N)')
    ax4.set_ylabel('Time per Element (μs)')
    ax4.set_title('Cost per Element at Scale')
    sns.despine(ax=ax4, trim=True)

    # Direct labeling for each point
    for i, (x, y, config) in enumerate(zip(total_elements, time_per_element * 1e6, configurations)):
        ax4.annotate(config, (x, y), xytext=(
            0, -10), textcoords='offset points', fontsize=10, ha='center', va='top')

    # Adjust layout to make room for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
