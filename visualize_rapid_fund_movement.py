#!/usr/bin/env python3
"""
RapidFundMovement Pattern Visualization

Creates two visualizations for the RapidFundMovement pattern:
1. Network topology showing entities, locations, and transaction flows
2. Timeline showing temporal pattern of transactions
"""

from experiments.h1_pattern_validation import create_h1_config
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import subprocess
import tempfile
import datetime
import json
import sys
import os

# Add the parent directory to path to import main
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))


def set_visualization_style():
    """Applies Tufte's principles and accessibility guidelines for visualizations."""
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
    })
    return sns.color_palette("colorblind")


def generate_pattern_data():
    """Generate pattern data for visualization"""
    print("Generating RapidFundMovement pattern data...")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "viz_config.yaml")
        patterns_file = os.path.join(temp_dir, "generated_patterns.json")
        nodes_file = os.path.join(temp_dir, "generated_nodes.csv")
        edges_file = os.path.join(temp_dir, "generated_edges.csv")

        # Create config file
        config = create_h1_config()
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Run generation using subprocess
        main_script = str(Path(__file__).parent / "main.py")
        cmd = [
            sys.executable, main_script,
            '--config', config_file,
            '--output-dir', temp_dir,
            '--patterns-file', 'generated_patterns.json'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"ERROR: Generation failed with return code {result.returncode}")
            print(result.stderr)
            return None, None, None

        # Load pattern data
        with open(patterns_file, 'r') as f:
            pattern_data = json.load(f)

        # Load node and edge data
        nodes_df = pd.read_csv(nodes_file)
        edges_df = pd.read_csv(edges_file)

        return pattern_data, nodes_df, edges_df


def create_network_visualization(pattern_data, nodes_df, edges_df, colors):
    """Create network topology visualization"""
    # Focus on RapidFundMovement patterns
    rfm_patterns = [p for p in pattern_data['patterns']
                    if p['pattern_type'] == 'RapidFundMovement']

    if not rfm_patterns:
        print("No RapidFundMovement patterns found!")
        return

    # Use the first pattern for visualization
    pattern = rfm_patterns[0]
    print(f"Visualizing pattern: {pattern['pattern_id']}")

    # Create a single, clear network diagram
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.suptitle(f'RapidFundMovement Pattern Structure: {pattern["pattern_id"]}',
                 fontsize=20, fontweight='bold')

    # Extract pattern entities and transactions
    pattern_entities = set(pattern['entities'])
    pattern_transactions = pattern['transactions']

    # Get entity information
    entity_info = {}
    for _, node in nodes_df.iterrows():
        if node['node_id'] in pattern_entities or node['node_id'] == 'account_9459':
            entity_info[node['node_id']] = {
                'type': node['node_type'].replace('NodeType.', ''),
                'country': node.get('country_code', 'Unknown'),
                'currency': node.get('currency', 'Unknown')
            }

    # Organize entities by role
    individuals = []
    individual_accounts = []
    overseas_accounts = []
    cash_account = None

    for entity_id in pattern_entities:
        if entity_id == 'account_9459':
            cash_account = entity_id
            continue

        info = entity_info.get(entity_id, {})
        if info.get('type') == 'INDIVIDUAL':
            individuals.append(entity_id)
        elif info.get('type') == 'ACCOUNT':
            # Determine if it's an overseas account based on transaction patterns
            is_overseas = any(tx['src'] == entity_id and tx['transaction_type'] == 'TransactionType.TRANSFER'
                              for tx in pattern_transactions)
            if is_overseas:
                overseas_accounts.append(entity_id)
            else:
                individual_accounts.append(entity_id)

    # Create manual positioning for clear flow visualization
    pos = {}

    # Position overseas accounts on the left (senders)
    y_start = len(overseas_accounts) / 2
    for i, account in enumerate(overseas_accounts):
        pos[account] = (-3, y_start - i - 1)

    # Position individual accounts in the center
    y_start = len(individual_accounts) / 2
    for i, account in enumerate(individual_accounts):
        pos[account] = (0, y_start - i - 1)

    # Position individuals near their accounts
    for individual in individuals:
        # Find connected accounts
        connected_accounts = [acc for acc in individual_accounts
                              if any(tx['src'] == acc or tx['dest'] == acc for tx in pattern_transactions)]
        if connected_accounts:
            # Position near the first connected account
            acc_pos = pos[connected_accounts[0]]
            pos[individual] = (acc_pos[0] + 1.5, acc_pos[1])
        else:
            pos[individual] = (1.5, 0)

    # Position cash account on the right
    if cash_account:
        pos[cash_account] = (3, 0)

    # Create the graph
    G = nx.DiGraph()

    # Add all entities as nodes
    all_entities = overseas_accounts + individual_accounts + individuals
    if cash_account:
        all_entities.append(cash_account)

    for entity in all_entities:
        G.add_node(entity)

    # Process transactions and add edges with detailed information
    edges_info = {}
    for tx in pattern_transactions:
        src, dest = tx['src'], tx['dest']
        if src in G.nodes and dest in G.nodes:
            edge_key = (src, dest)
            if edge_key not in edges_info:
                edges_info[edge_key] = {
                    'transactions': [],
                    'total_amount': 0,
                    'count': 0,
                    'type': tx['transaction_type']
                }

            edges_info[edge_key]['transactions'].append(tx)
            edges_info[edge_key]['total_amount'] += tx['amount']
            edges_info[edge_key]['count'] += 1

    # Draw nodes with clear styling
    for entity in G.nodes():
        info = entity_info.get(entity, {})
        entity_type = info.get('type', 'Unknown')
        country = info.get('country', 'Unknown')

        # Determine node color and style
        if entity_type == 'INDIVIDUAL':
            color = colors[0]  # Blue
            size = 3000
        elif entity == 'account_9459':  # Cash account
            color = colors[3]  # Reddish-orange
            size = 2000
        elif entity_type == 'ACCOUNT':
            if entity in overseas_accounts:
                color = colors[2]  # Green
            else:
                color = colors[4]  # Yellow-orange
            size = 2000
        else:
            color = 'gray'
            size = 1500

        # Draw node
        ax.scatter(pos[entity][0], pos[entity][1], c=color, s=size,
                   alpha=0.8, edgecolors='black', linewidth=2, zorder=3)

        # Create clear label
        if entity == 'account_9459':
            label = 'CASH'
        else:
            # Extract meaningful part of entity ID
            if entity.startswith('individual_'):
                label = f"Individual_{entity.split('_')[1]}"
            elif entity.startswith('account_'):
                label = f"Account_{entity.split('_')[1]}"
            else:
                label = entity

            if country != 'Unknown':
                label += f"\n[{country}]"

        # Position label
        ax.text(pos[entity][0], pos[entity][1], label,
                ha='center', va='center', fontsize=12, fontweight='bold', zorder=4)

    # Draw edges with detailed labels
    for (src, dest), edge_data in edges_info.items():
        src_pos = pos[src]
        dest_pos = pos[dest]

        # Determine line style and color
        tx_type = edge_data['type']
        if 'TRANSFER' in tx_type:
            color = colors[0]  # Blue
            linestyle = '-'
            alpha = 0.8
        else:  # WITHDRAWAL
            color = colors[3]  # Reddish-orange
            linestyle = '--'
            alpha = 0.8

        # Draw arrow
        ax.annotate('', xy=dest_pos, xytext=src_pos,
                    arrowprops=dict(arrowstyle='->', color=color, lw=3,
                                    linestyle=linestyle, alpha=alpha),
                    zorder=1)

        # Position label along the edge
        mid_x = (src_pos[0] + dest_pos[0]) / 2
        mid_y = (src_pos[1] + dest_pos[1]) / 2

        # Offset label slightly to avoid overlapping with arrow
        offset_y = 0.3 if src_pos[1] == dest_pos[1] else 0

        # Add transaction type label
        type_label = tx_type.replace('TransactionType.', '').title()
        ax.text(mid_x, mid_y + offset_y, type_label,
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.9),
                zorder=2)

    # Set axis properties
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0],
                   markersize=15, label='Individual'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2],
                   markersize=15, label='Overseas Account'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[4],
                   markersize=15, label='Individual Account'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[3],
                   markersize=15, label='Cash System'),
        plt.Line2D([0], [0], color=colors[0], linewidth=3, label='Transfers'),
        plt.Line2D([0], [0], color=colors[3], linewidth=3,
                   linestyle='--', label='Cash Withdrawals')
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=14)

    # Add summary statistics
    total_inflow = sum(tx['amount'] for tx in pattern_transactions
                       if tx['transaction_type'] == 'TransactionType.TRANSFER')
    total_outflow = sum(tx['amount'] for tx in pattern_transactions
                        if tx['transaction_type'] == 'TransactionType.WITHDRAWAL')
    ratio = total_outflow / total_inflow if total_inflow > 0 else 0

    stats_text = (f"Pattern Summary:\n"
                  f"Total Inflow: €{total_inflow:,.0f}\n"
                  f"Total Outflow: €{total_outflow:,.0f}\n"
                  f"Outflow Ratio: {ratio:.1%}\n"
                  f"Entities: {len(pattern_entities)}\n"
                  f"Transactions: {len(pattern_transactions)}")

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
                                               facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(pad=0)
    plt.savefig('rapid_fund_movement_network.png')
    print("✓ Network topology visualization saved as 'rapid_fund_movement_network.png'")
    plt.show()

    return pattern


def create_timeline_visualization(pattern_data, pattern, colors):
    """Create timeline visualization of transactions"""
    transactions = pattern['transactions']

    # Convert timestamps
    for tx in transactions:
        tx['datetime'] = datetime.datetime.fromisoformat(
            tx['timestamp'].replace('Z', ''))

    # Sort by timestamp
    transactions.sort(key=lambda x: x['datetime'])

    # Separate inflows and withdrawals
    inflows = [tx for tx in transactions if tx['transaction_type']
               == 'TransactionType.TRANSFER']
    withdrawals = [tx for tx in transactions if tx['transaction_type']
                   == 'TransactionType.WITHDRAWAL']

    # Create timeline plot
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
    fig.suptitle(f'RapidFundMovement Pattern Timeline: {pattern["pattern_id"]}',
                 fontsize=20, fontweight='bold')
    # Plot transaction amounts over time
    if inflows:
        inflow_times = [tx['datetime'] for tx in inflows]
        inflow_amounts = [tx['amount'] for tx in inflows]
        ax1.scatter(inflow_times, inflow_amounts,
                    color=colors[0], s=100, alpha=0.7, label=f'Inflows ({len(inflows)} txns)')
    if withdrawals:
        withdrawal_times = [tx['datetime'] for tx in withdrawals]
        withdrawal_amounts = [tx['amount'] for tx in withdrawals]
        ax1.scatter(withdrawal_times, withdrawal_amounts,
                    color=colors[3], s=100, alpha=0.7, label=f'Withdrawals ({len(withdrawals)} txns)')
    ax1.set_ylabel('Amount (€)', fontsize=14)
    ax1.set_title('Transaction Amounts Over Time',
                  fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(False)
    sns.despine(ax=ax1)
    # Force x-axis to show timestamps at the beginning and end of the timeline (horizontal)
    start_time = min(tx['datetime'] for tx in transactions)
    end_time = max(tx['datetime'] for tx in transactions)
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax1.set_xticks([start_time, end_time])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')
    # Re-add inflow and withdrawal phase annotations (if inflows and withdrawals exist)
    if inflows and withdrawals:
        inflow_end = max(tx['datetime'] for tx in inflows)
        withdrawal_start = min(tx['datetime'] for tx in withdrawals)
        ax1.axvline(x=inflow_end, color='gray', linestyle=':', alpha=0.7)
        ax1.axvline(x=withdrawal_start, color='gray', linestyle=':', alpha=0.7)
        inflow_mid = start_time + (inflow_end - start_time) / 2
        withdrawal_mid = withdrawal_start + (end_time - withdrawal_start) / 2
        ax1.text(inflow_mid, ax1.get_ylim()[
                 1] * 0.9, 'Inflow Phase', ha='center', va='center', fontsize=12, fontweight='bold', color=colors[0])
        ax1.text(withdrawal_mid, ax1.get_ylim()[
                 1] * 0.9, 'Withdrawal Phase', ha='center', va='center', fontsize=12, fontweight='bold', color=colors[3])
    plt.tight_layout(pad=1.5)
    plt.savefig('rapid_fund_movement_timeline.png')
    plt.show()


def main():
    """Main function to generate visualizations"""
    print("=== RapidFundMovement Pattern Visualization ===")
    colors = set_visualization_style()

    # Generate pattern data
    pattern_data, nodes_df, edges_df = generate_pattern_data()
    if pattern_data is None:
        return

    print(f"Generated {len(pattern_data['patterns'])} patterns")

    # Create visualizations
    print("\nCreating network topology visualization...")
    pattern = create_network_visualization(
        pattern_data, nodes_df, edges_df, colors)

    print("\nCreating timeline visualization...")
    create_timeline_visualization(pattern_data, pattern, colors)

    print("\nVisualizations saved:")
    print("- rapid_fund_movement_network.png")
    print("- rapid_fund_movement_timeline.png")


if __name__ == "__main__":
    main()
