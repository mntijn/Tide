#!/usr/bin/env python3
"""
H2 Scalability Analysis Script for Aggregated Results

Analyzes scaling performance data from H2 experiment results, including mean and standard deviation
from multiple runs. Calculates Pearson correlation, power-law scaling exponent, and generates
visualizations with error bars.
"""

import re
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


def parse_aggregated_results(file_path):
    """Parse the aggregated results file."""
    with open(file_path, 'r') as f:
        content = f.read()

    results = []
    # Find all headers and then split the content by those headers.
    # This is more robust for headers with spaces.
    headers = re.findall(r'.+? Scale Aggregated Results:', content)
    sections = re.split(r'.+? Scale Aggregated Results:', content)

    if not headers or len(sections) < 2:
        return []

    # The first element of sections is the content before the first header, so we skip it.
    for header, section_content in zip(headers, sections[1:]):
        name = header.replace(" Scale Aggregated Results:", "").strip()
        section = header + section_content

        def extract_value(pattern):
            match = re.search(pattern, section)
            if match:
                # Remove commas for float conversion
                mean_str = match.group(1).replace(',', '')
                std_str = match.group(2).replace(',', '')
                # Handle cases where std might be missing or non-numeric
                try:
                    mean = float(mean_str)
                    std = float(std_str)
                    return mean, std
                except (ValueError, IndexError):
                    return (float(mean_str) if mean_str else None), 0.0
            return None, None

        total_elements_mean, total_elements_std = extract_value(
            r"Total Elements \(N\):\s+([\d,.]+)\s+\(±([\d,.]+)\)")
        wall_time_mean, wall_time_std = extract_value(
            r"Wall Time:\s+([\d.]+)s\s+\(±([\d.]+)s\)")
        memory_mean, memory_std = extract_value(
            r"Memory Usage:\s+([\d,.]+)MB\s+\(±([\d,.]+)MB\)")

        if total_elements_mean is not None and wall_time_mean is not None and memory_mean is not None:
            results.append({
                "config": name,
                "total_elements_mean": total_elements_mean,
                "total_elements_std": total_elements_std,
                "time_mean": wall_time_mean,
                "time_std": wall_time_std,
                "memory_mean": memory_mean,
                "memory_std": memory_std,
            })
    return results


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


def create_time_scaling_plot(configurations, total_elements, times, time_stds, time_fit_params):
    """Create Tufte-inspired time scaling analysis plot (log-log and per-element only)."""
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('H2 Time Scalability Analysis (Aggregated)', fontsize=26,
                 fontweight='bold', color='black', y=1.02)

    # Plot 1: Time Log-log scale
    ax2.errorbar(total_elements, times, yerr=time_stds, fmt='o', markersize=8,
                 color='#D81B60', alpha=0.7, capsize=5, mec='w')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Total Elements (N)')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Time Scaling (Log-Log Scale)')
    ax2.set_xlim(left=min(total_elements) * 0.9,
                 right=max(total_elements) * 1.1)
    ax2.set_ylim(bottom=min(times) * 0.5, top=max(times) * 2.0)
    sns.despine(ax=ax2)
    for i, (x, y, config) in enumerate(zip(total_elements, times, configurations)):
        ax2.annotate(config, (x, y), xytext=(
            8, -4), textcoords='offset points', fontsize=14, ha='left', va='center')

    if time_fit_params:
        alpha, c, r_squared = time_fit_params
        N_fit = np.logspace(np.log10(min(total_elements)),
                            np.log10(max(total_elements)), 100)
        T_fit = power_law_model(N_fit, c, alpha)
        ax2.plot(N_fit, T_fit, '--', color='#004D40', linewidth=2.5)
        fit_text = f'T ≈ {c:.2e} * N^{alpha:.3f} (R²={r_squared:.3f})'
        ax2.text(0.95, 0.05, fit_text, transform=ax2.transAxes, fontsize=16,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.7, ec='none'))

        # Add O(N) reference line, anchored to the first data point
        c_linear = times[0] / total_elements[0]
        T_linear = c_linear * N_fit
        ax2.plot(N_fit, T_linear, ':', color='grey', linewidth=2.0)
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
    ax3.set_ylim(bottom=0, top=max(time_per_element * 1e6) * 1.1)
    sns.despine(ax=ax3)
    for i, (x, y, config) in enumerate(zip(total_elements, time_per_element * 1e6, configurations)):
        ax3.annotate(config, (x, y), xytext=(
            0, -10), textcoords='offset points', fontsize=14, ha='center', va='top')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def create_memory_scaling_plot(configurations, total_elements, memory_usages, memory_stds, total_elements_stds, memory_fit_params):
    """Create a Tufte-inspired memory scaling analysis plot (log-log and per-element)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('H2 Peak Memory Scalability (Aggregated)',
                 fontsize=26, fontweight='bold', color='black', y=1.02)

    # Plot 1: Memory Log-log scale
    ax1.errorbar(total_elements, memory_usages, yerr=memory_stds, fmt='o', markersize=8,
                 color='#1E88E5', alpha=0.7, capsize=5, mec='w')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Total Elements (N)')
    ax1.set_ylabel('Peak Memory (MB)')
    ax1.set_title('Peak Memory vs. Graph Size (Log-Log)')
    ax1.set_xlim(left=min(total_elements) * 0.9,
                 right=max(total_elements) * 1.1)
    ax1.set_ylim(bottom=min(memory_usages) * 0.5,
                 top=max(memory_usages) * 2.0)
    sns.despine(ax=ax1)

    for i, (x, y, config) in enumerate(zip(total_elements, memory_usages, configurations)):
        ax1.annotate(config, (x, y), xytext=(
            8, -4), textcoords='offset points', fontsize=14, ha='left', va='center')

    if memory_fit_params:
        alpha_mem, c_mem, r_squared_mem = memory_fit_params
        N_fit = np.logspace(np.log10(min(total_elements)),
                            np.log10(max(total_elements)), 100)
        Mem_fit = power_law_model(N_fit, c_mem, alpha_mem)
        ax1.plot(N_fit, Mem_fit, '--', color='#004D40', linewidth=2.5)
        fit_text = f'M ≈ {c_mem:.2e} * N^{alpha_mem:.3f} (R²={r_squared_mem:.3f})'
        ax1.text(0.95, 0.05, fit_text, transform=ax1.transAxes, fontsize=16,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.7, ec='none'))

        # Add O(N) reference line, anchored to the first data point
        c_mem_linear = memory_usages[0] / total_elements[0]
        Mem_linear = c_mem_linear * N_fit
        ax1.plot(N_fit, Mem_linear, ':', color='grey', linewidth=2.0)

        y_top = ax1.get_ylim()[1]
        visible_indices = np.where(Mem_linear <= y_top)[0]
        if len(visible_indices) > 0:
            label_index = visible_indices[-1]
            ax1.text(N_fit[label_index], Mem_linear[label_index], ' O(N)', fontsize=14, color='grey',
                     ha='left', va='bottom')

    # Plot 2: Memory per element with error propagation
    memory_usages_arr = np.array(memory_usages)
    total_elements_arr = np.array(total_elements)
    memory_stds_arr = np.array(memory_stds)
    total_elements_stds_arr = np.array(total_elements_stds)

    memory_per_element = memory_usages_arr / total_elements_arr

    # Propagate error: y = M/N -> y_std = y * sqrt((M_std/M)^2 + (N_std/N)^2)
    relative_mem_std_sq = (memory_stds_arr / memory_usages_arr)**2
    relative_n_std_sq = (total_elements_stds_arr / total_elements_arr)**2
    memory_per_element_std = memory_per_element * \
        np.sqrt(relative_mem_std_sq + relative_n_std_sq)

    # Convert to KB for better readability
    memory_per_element_kb = memory_per_element * 1024
    memory_per_element_std_kb = memory_per_element_std * 1024

    ax2.errorbar(total_elements, memory_per_element_kb, yerr=memory_per_element_std_kb,
                 fmt='o-', markersize=7, color='#1E88E5',
                 linewidth=1.5, alpha=0.7, mfc='w', mec='#1E88E5', capsize=5)
    ax2.set_xlabel('Total Elements (N)')
    ax2.set_ylabel('Memory per Element (KB)')
    ax2.set_title('Cost per Element (Memory)')
    ax2.set_xlim(left=0, right=max(total_elements) * 1.05)
    ax2.set_ylim(bottom=0, top=max(memory_per_element_kb) * 1.2)
    sns.despine(ax=ax2)
    for i, (x, y, config) in enumerate(zip(total_elements, memory_per_element_kb, configurations)):
        ax2.annotate(config, (x, y), xytext=(
            0, -10), textcoords='offset points', fontsize=14, ha='center', va='top')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def create_throughput_plot(configurations, total_elements, throughputs, throughput_stds):
    """Create a Tufte-inspired throughput analysis plot."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.suptitle('H2 Throughput Analysis (Aggregated)',
                 fontsize=26, fontweight='bold', color='black', y=1.02)

    # Convert throughput to kElements/s for better axis readability
    throughputs_k = throughputs / 1000
    throughput_stds_k = throughput_stds / 1000

    ax.errorbar(total_elements, throughputs_k, yerr=throughput_stds_k,
                fmt='o-', markersize=8, color='#FFC107',
                linewidth=2, alpha=0.8, mfc='w', mec='#FFC107', capsize=5)

    ax.set_xlabel('Total Elements (N)')
    ax.set_ylabel('Throughput (kElements/s)')
    ax.set_title('Generator Throughput vs. Graph Size')
    ax.set_xlim(left=0, right=max(total_elements) * 1.05)
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax)

    # Annotate points
    for i, (x, y, config) in enumerate(zip(total_elements, throughputs_k, configurations)):
        ax.annotate(config, (x, y), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def analyze_h2_results(results_file):
    """Analyze H2 scalability results from the provided aggregated results file"""

    print("=== H2 Aggregated Scalability Analysis ===")
    print(f"Analyzing results from: {results_file}\n")

    # Data from the file
    parsed_data = parse_aggregated_results(results_file)
    if not parsed_data:
        print("Could not parse data from the results file. Exiting.")
        return

    configurations = [d['config'] for d in parsed_data]
    total_elements = [d['total_elements_mean'] for d in parsed_data]
    total_elements_stds = [d['total_elements_std'] for d in parsed_data]
    times = [d['time_mean'] for d in parsed_data]
    time_stds = [d['time_std'] for d in parsed_data]
    memory_usages = [d['memory_mean'] for d in parsed_data]
    memory_stds = [d['memory_std'] for d in parsed_data]

    # Print raw data
    print("Parsed Mean Data:")
    for i, config in enumerate(configurations):
        print(
            f"  {config}: N={total_elements[i]:,.0f}, T={times[i]:.2f}s (±{time_stds[i]:.2f}), Mem={memory_usages[i]:.1f}MB (±{memory_stds[i]:.1f})")
    print()

    # --- Throughput Analysis ---
    print("=== Throughput Analysis ===")
    total_elements_arr = np.array(total_elements)
    times_arr = np.array(times)
    throughputs = total_elements_arr / times_arr

    # Propagate error for throughput: T = N/t -> T_std = T * sqrt((N_std/N)^2 + (t_std/t)^2)
    relative_n_std_sq = (np.array(total_elements_stds) / total_elements_arr)**2
    relative_t_std_sq = (np.array(time_stds) / times_arr)**2
    throughput_stds = throughputs * \
        np.sqrt(relative_n_std_sq + relative_t_std_sq)

    for i, config in enumerate(configurations):
        print(
            f"  {config}: {throughputs[i]/1000:.1f} kElements/s (±{throughput_stds[i]/1000:.1f} kElements/s)")
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

    # Create visualizations
    print("\n=== Generating Visualizations ===")

    # Time scaling plot
    time_fig = create_time_scaling_plot(
        configurations, total_elements, times, time_stds, (alpha, c, r_squared)
    )
    time_output_file = Path(__file__).parent / \
        "h2_time_scaling_analysis_aggregated.png"
    time_fig.savefig(time_output_file, dpi=300,
                     bbox_inches='tight', facecolor='white')
    print(f"Time scaling analysis plot saved to: {time_output_file}")

    # Memory scaling plot
    mem_fig = create_memory_scaling_plot(
        configurations, total_elements, memory_usages, memory_stds, total_elements_stds, (
            alpha_mem, c_mem, r_squared_mem)
    )
    mem_output_file = Path(__file__).parent / \
        "h2_memory_scaling_analysis_aggregated.png"
    mem_fig.savefig(mem_output_file, dpi=300,
                    bbox_inches='tight', facecolor='white')
    print(f"Memory scaling analysis plot saved to: {mem_output_file}")

    # Throughput plot
    throughput_fig = create_throughput_plot(
        configurations, total_elements, throughputs, throughput_stds
    )
    throughput_output_file = Path(__file__).parent / \
        "h2_throughput_analysis_aggregated.png"
    throughput_fig.savefig(throughput_output_file, dpi=300,
                           bbox_inches='tight', facecolor='white')
    print(f"Throughput analysis plot saved to: {throughput_output_file}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # The script expects the path to the results file as an argument,
    # but defaults to 'new_Results.txt' in the project root for convenience.
    import sys
    if len(sys.argv) > 1:
        results_file_path = Path(sys.argv[1])
    else:
        results_file_path = Path(
            __file__).parent.parent / "new_results_final.txt"

    if not results_file_path.is_file():
        print(f"Error: Results file not found at '{results_file_path}'")
        sys.exit(1)

    analyze_h2_results(results_file_path)
