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

# Import the H1 config


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


def create_network_visualization(pattern_data, nodes_df, edges_df):
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
                 fontsize=16, fontweight='bold')

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
            color = 'lightblue'
            size = 3000
        elif entity == 'account_9459':  # Cash account
            color = 'red'
            size = 2000
        elif entity_type == 'ACCOUNT':
            if entity in overseas_accounts:
                color = 'lightgreen'
            else:
                color = 'orange'
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
                ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

    # Draw edges with detailed labels
    for (src, dest), edge_data in edges_info.items():
        src_pos = pos[src]
        dest_pos = pos[dest]

        # Determine line style and color
        tx_type = edge_data['type']
        if 'TRANSFER' in tx_type:
            color = 'blue'
            linestyle = '-'
            alpha = 0.8
        else:  # WITHDRAWAL
            color = 'red'
            linestyle = '--'
            alpha = 0.8

        # Draw arrow
        ax.annotate('', xy=dest_pos, xytext=src_pos,
                    arrowprops=dict(arrowstyle='->', color=color, lw=3,
                                    linestyle=linestyle, alpha=alpha),
                    zorder=1)

        # Add transaction details as label
        total_amount = edge_data['total_amount']
        count = edge_data['count']

        # Create detailed label
        tx_details = []
        for tx in edge_data['transactions'][:3]:  # Show first 3 transactions
            amount = tx['amount']
            timestamp = tx['timestamp']
            # Parse timestamp for readable format
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', ''))
            time_str = dt.strftime('%H:%M, %d-%m')
            tx_details.append(f"€{amount:,.0f}\n{time_str}")

        if count > 3:
            tx_details.append(f"...+{count-3} more")

        label_text = '\n'.join(tx_details)

        # Position label along the edge
        mid_x = (src_pos[0] + dest_pos[0]) / 2
        mid_y = (src_pos[1] + dest_pos[1]) / 2

        # Offset label slightly to avoid overlapping with arrow
        offset_y = 0.3 if src_pos[1] == dest_pos[1] else 0

        ax.text(mid_x, mid_y + offset_y, label_text,
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.9),
                zorder=2)

        # Add transaction type label
        type_label = tx_type.replace('TransactionType.', '').title()
        ax.text(mid_x, mid_y - 0.2 + offset_y, type_label,
                ha='center', va='center', fontsize=8, fontweight='bold',
                color=color, zorder=2)

    # Set axis properties
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                   markersize=15, label='Individual'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                   markersize=15, label='Overseas Account'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=15, label='Individual Account'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=15, label='Cash System'),
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Transfers'),
        plt.Line2D([0], [0], color='red', linewidth=3,
                   linestyle='--', label='Cash Withdrawals')
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

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

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
                                               facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig('rapid_fund_movement_network.png',
                dpi=300, bbox_inches='tight')
    print("✓ Network topology visualization saved as 'rapid_fund_movement_network.png'")
    plt.show()

    return pattern


def create_timeline_visualization(pattern_data, pattern):
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
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'RapidFundMovement Pattern Timeline: {pattern["pattern_id"]}',
                 fontsize=16, fontweight='bold')

    # Plot 1: Transaction amounts over time
    if inflows:
        inflow_times = [tx['datetime'] for tx in inflows]
        inflow_amounts = [tx['amount'] for tx in inflows]
        ax1.scatter(inflow_times, inflow_amounts, color='blue', s=100, alpha=0.7,
                    label=f'Inflows ({len(inflows)} txns)')

    if withdrawals:
        withdrawal_times = [tx['datetime'] for tx in withdrawals]
        withdrawal_amounts = [tx['amount'] for tx in withdrawals]
        ax1.scatter(withdrawal_times, withdrawal_amounts, color='red', s=100, alpha=0.7,
                    label=f'Withdrawals ({len(withdrawals)} txns)')

    ax1.set_ylabel('Amount (€)', fontsize=12)
    ax1.set_title('Transaction Amounts Over Time',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Cumulative amounts
    all_times = [tx['datetime'] for tx in transactions]
    cumulative_inflow = []
    cumulative_outflow = []
    current_inflow = 0
    current_outflow = 0

    for tx in transactions:
        if tx['transaction_type'] == 'TransactionType.TRANSFER':
            current_inflow += tx['amount']
        else:
            current_outflow += tx['amount']
        cumulative_inflow.append(current_inflow)
        cumulative_outflow.append(current_outflow)

    ax2.plot(all_times, cumulative_inflow, color='blue', linewidth=2,
             label=f'Cumulative Inflow (€{current_inflow:,.0f})')
    ax2.plot(all_times, cumulative_outflow, color='red', linewidth=2,
             label=f'Cumulative Outflow (€{current_outflow:,.0f})')

    # Add ratio annotation
    ratio = current_outflow / current_inflow if current_inflow > 0 else 0
    ax2.axhline(y=current_inflow * 0.85, color='green', linestyle='--', alpha=0.5,
                label=f'85% threshold (€{current_inflow * 0.85:,.0f})')
    ax2.axhline(y=current_inflow * 0.95, color='green', linestyle='--', alpha=0.5,
                label=f'95% threshold (€{current_inflow * 0.95:,.0f})')

    ax2.set_ylabel('Cumulative Amount (€)', fontsize=12)
    ax2.set_title(
        f'Cumulative Flow (Outflow Ratio: {ratio:.1%})', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Plot 3: Transaction frequency over time (hourly bins)
    start_time = min(all_times)
    end_time = max(all_times)
    duration = end_time - start_time

    # Create hourly bins
    hours = int(duration.total_seconds() / 3600) + 1
    time_bins = [start_time +
                 datetime.timedelta(hours=i) for i in range(hours + 1)]

    inflow_counts = [0] * hours
    withdrawal_counts = [0] * hours

    for tx in transactions:
        hour_index = int((tx['datetime'] - start_time).total_seconds() / 3600)
        if hour_index < hours:
            if tx['transaction_type'] == 'TransactionType.TRANSFER':
                inflow_counts[hour_index] += 1
            else:
                withdrawal_counts[hour_index] += 1

    # Create bar chart
    bar_times = time_bins[:-1]
    width = datetime.timedelta(hours=0.4)

    ax3.bar([t - width/2 for t in bar_times], inflow_counts, width=width,
            color='blue', alpha=0.7, label='Inflows/hour')
    ax3.bar([t + width/2 for t in bar_times], withdrawal_counts, width=width,
            color='red', alpha=0.7, label='Withdrawals/hour')

    ax3.set_ylabel('Transactions/Hour', fontsize=12)
    ax3.set_xlabel('Time', fontsize=12)
    ax3.set_title('Transaction Frequency Pattern',
                  fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # Add phase annotations
    if inflows and withdrawals:
        inflow_end = max(tx['datetime'] for tx in inflows)
        withdrawal_start = min(tx['datetime'] for tx in withdrawals)

        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=inflow_end, color='gray', linestyle=':', alpha=0.7)
            ax.axvline(x=withdrawal_start, color='gray',
                       linestyle=':', alpha=0.7)

        # Add phase labels
        inflow_mid = start_time + (inflow_end - start_time) / 2
        withdrawal_mid = withdrawal_start + (end_time - withdrawal_start) / 2

        ax1.text(inflow_mid, ax1.get_ylim()[1] * 0.9, 'Inflow Phase',
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        ax1.text(withdrawal_mid, ax1.get_ylim()[1] * 0.9, 'Withdrawal Phase',
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

    plt.tight_layout()
    plt.savefig('rapid_fund_movement_timeline.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to generate visualizations"""
    print("=== RapidFundMovement Pattern Visualization ===")

    # Generate pattern data
    pattern_data, nodes_df, edges_df = generate_pattern_data()
    if pattern_data is None:
        return

    print(f"Generated {len(pattern_data['patterns'])} patterns")

    # Create visualizations
    print("\nCreating network topology visualization...")
    pattern = create_network_visualization(pattern_data, nodes_df, edges_df)

    print("\nCreating timeline visualization...")
    create_timeline_visualization(pattern_data, pattern)

    print("\nVisualizations saved:")
    print("- rapid_fund_movement_network.png")
    print("- rapid_fund_movement_timeline.png")


if __name__ == "__main__":
    main()
