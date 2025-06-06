#!/usr/bin/env python3
"""
RepeatedOverseasTransfers Pattern Visualization

Creates two visualizations for the RepeatedOverseasTransfers pattern:
1. Network topology showing the central entity, their domestic account, and the destination overseas accounts.
2. Timeline showing the temporal pattern of initial deposits followed by periodic or burst transfers.
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


def generate_pattern_data():
    """Generate pattern data for visualization"""
    print("Generating RepeatedOverseasTransfers pattern data...")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "viz_config.yaml")
        patterns_file = os.path.join(temp_dir, "generated_patterns.json")
        nodes_file = os.path.join(temp_dir, "generated_nodes.csv")
        edges_file = os.path.join(temp_dir, "generated_edges.csv")

        # Create config file
        config = create_h1_config()
        # Ensure the pattern is active
        if "pattern_configs" in config and "repeatedOverseas" in config["pattern_configs"]:
            config["pattern_configs"]["repeatedOverseas"]["enabled"] = True
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Run generation using subprocess
        project_root = Path(__file__).parent
        main_script = str(project_root / "main.py")
        cmd = [
            sys.executable, main_script,
            '--config', config_file,
            '--output-dir', temp_dir,
            '--patterns-file', 'generated_patterns.json'
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=project_root)
        print(result.stdout)
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
    # Focus on RepeatedOverseasTransfers patterns
    rot_patterns = [p for p in pattern_data['patterns']
                    if p['pattern_type'] == 'RepeatedOverseasTransfers']

    if not rot_patterns:
        print("No RepeatedOverseasTransfers patterns found!")
        return None

    # Use the first pattern for visualization
    pattern = rot_patterns[0]
    print(f"Visualizing pattern: {pattern['pattern_id']}")

    # Create a single, clear network diagram
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.suptitle(f'Repeated Overseas Transfers Pattern: {pattern["pattern_id"]}',
                 fontsize=16, fontweight='bold')

    # Extract pattern entities and transactions
    pattern_entities = set(pattern['entities'])
    pattern_transactions = pattern['transactions']

    # Get entity information
    entity_info = {node['node_id']: {
        'type': node['node_type'].replace('NodeType.', ''),
        'country': node.get('country_code', 'Unknown')
    } for _, node in nodes_df.iterrows() if node['node_id'] in pattern_entities}

    # Identify roles from transactions
    deposit_txs = [
        tx for tx in pattern_transactions if tx['transaction_type'] == 'TransactionType.DEPOSIT']
    transfer_txs = [
        tx for tx in pattern_transactions if tx['transaction_type'] == 'TransactionType.TRANSFER']

    if not deposit_txs or not transfer_txs:
        print("Pattern is missing deposit or transfer transactions.")
        return None

    central_entity = deposit_txs[0]['src']
    domestic_account = deposit_txs[0]['dest']
    overseas_accounts = list(set(tx['dest'] for tx in transfer_txs))

    # Find owners of overseas accounts
    overseas_account_owners = {}
    ownership_edges = edges_df[edges_df['edge_type'] == 'EdgeType.OWNS']
    for acc_id in overseas_accounts:
        owner_edge = ownership_edges[ownership_edges['dest'] == acc_id]
        if not owner_edge.empty:
            owner_id = owner_edge['src'].iloc[0]
            owner_node = nodes_df[nodes_df['node_id'] == owner_id]
            if not owner_node.empty:
                owner_type = owner_node['node_type'].iloc[0].replace(
                    'NodeType.', '')
                overseas_account_owners[acc_id] = {
                    'id': owner_id, 'type': owner_type}

    # Add a synthetic "Cash System" node for clarity
    cash_system_node = 'CASH_SYSTEM'
    entity_info[cash_system_node] = {'type': 'SYSTEM', 'country': 'N/A'}

    # Create structured horizontal layout for clear flow visualization
    pos = {}
    pos[cash_system_node] = (-6, 0)
    pos[central_entity] = (-3, 0)
    pos[domestic_account] = (0, 0)

    # Position overseas accounts on the right side (vertically distributed)
    num_overseas = len(overseas_accounts)
    if num_overseas > 0:
        y_spacing = 3.0 if num_overseas > 1 else 0
        y_start = (num_overseas - 1) * y_spacing / 2
        for i, account in enumerate(overseas_accounts):
            pos[account] = (4.5, y_start - i * y_spacing)

    # Create the graph
    G = nx.DiGraph()
    all_graph_nodes = [central_entity, domestic_account,
                       cash_system_node] + overseas_accounts
    G.add_nodes_from(all_graph_nodes)

    # Draw nodes with clear styling inspired by the example image
    node_colors = {'INDIVIDUAL': 'yellow', 'BUSINESS': 'yellow',
                   'ACCOUNT': 'orange', 'SYSTEM': 'red'}
    for entity in G.nodes():
        info = entity_info.get(entity, {})
        entity_type = info.get('type', 'Unknown')
        country = info.get('country', 'Unknown')

        color = node_colors.get(entity_type, 'gray')
        size = 3000

        # Override for specific roles
        if entity == domestic_account:
            color = 'orange'
        elif entity in overseas_accounts:
            color = 'deepskyblue'

        ax.scatter(pos[entity][0], pos[entity][1], c=color, s=size,
                   alpha=0.9, edgecolors='black', linewidth=1.5, zorder=3)

        # Create clear label
        if entity == cash_system_node:
            label = "Cash Deposits"
        elif entity == central_entity:
            id_part = entity.split('_')[-1]
            label = f"{entity_type.title()}_{id_part}\n[{country}]"
        elif entity == domestic_account:
            # Domestic account is always first
            label = f"Account_1\n[{country}]"
        elif entity in overseas_accounts:
            # Number overseas accounts sequentially
            acc_num = overseas_accounts.index(entity) + 1
            label = f"Business_{acc_num}_account\n[{country}]"
        else:
            id_part = entity.split('_')[-1]
            label = f"{entity_type.title()}_{id_part}\n[{country}]"

        ax.text(pos[entity][0], pos[entity][1], label,
                ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

    # --- Draw Edges ---
    # 1. Edge from Cash System to Domestic Account (Deposits)
    total_deposit_amount = sum(tx['amount'] for tx in deposit_txs)
    ax.annotate('', xy=pos[domestic_account], xytext=pos[cash_system_node],
                arrowprops=dict(arrowstyle='->', color='purple', lw=3,
                                linestyle='--', alpha=0.8), zorder=1)
    mid_x = (pos[cash_system_node][0] + pos[domestic_account][0]) / 2
    mid_y = (pos[cash_system_node][1] + pos[domestic_account][1]) / 2
    label_text = f"Deposits\n€{total_deposit_amount:,.0f}\n({len(deposit_txs)} txs)"
    ax.text(mid_x, mid_y + 0.4, label_text, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='purple', alpha=0.9), zorder=2)

    # 2. Edges from Domestic Account to Overseas Accounts
    for account in overseas_accounts:
        txs_to_account = [
            tx for tx in transfer_txs if tx['dest'] == account]
        total_amount = sum(tx['amount'] for tx in txs_to_account)
        count = len(txs_to_account)

        ax.annotate('', xy=pos[account], xytext=pos[domestic_account],
                    arrowprops=dict(arrowstyle='->', color='darkblue', lw=3,
                                    linestyle='-', alpha=0.8), zorder=1)
        mid_x = (pos[domestic_account][0] + pos[account][0]) / 2
        mid_y = (pos[domestic_account][1] + pos[account][1]) / 2
        label_text = f"Transfers\n€{total_amount:,.0f}\n({count} txs)"
        ax.text(mid_x, mid_y + 0.4, label_text, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='darkblue', alpha=0.9), zorder=2)

    # Set axis properties
    ax.set_xlim(-8, 8)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                   markersize=15, label='Central Entity (Owner)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=15, label='Domestic Source Account'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='deepskyblue',
                   markersize=15, label='Overseas Destination Accounts'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=15, label='Source of Funds (Cash)'),
        plt.Line2D([0], [0], color='purple', linewidth=3,
                   linestyle='--', label='Cash Deposits'),
        plt.Line2D([0], [0], color='darkblue', linewidth=3,
                   label='Overseas Transfers')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    # Add summary statistics
    total_transfers = sum(tx['amount'] for tx in transfer_txs)
    ratio = total_transfers / \
        total_deposit_amount if total_deposit_amount > 0 else 0
    stats_text = (f"Pattern Summary:\n"
                  f"Total Deposits: €{total_deposit_amount:,.0f}\n"
                  f"Total Transfers: €{total_transfers:,.0f}\n"
                  f"Transfer Ratio: {ratio:.1%}\n"
                  f"Dest. Countries: {len(set(entity_info[acc]['country'] for acc in overseas_accounts))}\n"
                  f"Transactions: {len(pattern_transactions)}")
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10, ha='right',
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig('repeated_overseas_transfers_network.png',
                dpi=300, bbox_inches='tight')
    print("✓ Network topology visualization saved as 'repeated_overseas_transfers_network.png'")
    plt.show()

    return pattern


def create_timeline_visualization(pattern, nodes_df, edges_df):
    """Create timeline visualization of transactions"""
    transactions = pattern['transactions']

    # Convert timestamps
    for tx in transactions:
        tx['datetime'] = datetime.datetime.fromisoformat(
            tx['timestamp'].replace('Z', ''))

    # Sort by timestamp
    transactions.sort(key=lambda x: x['datetime'])

    # Separate deposits and transfers
    deposits = [tx for tx in transactions if tx['transaction_type']
                == 'TransactionType.DEPOSIT']
    transfers = [tx for tx in transactions if tx['transaction_type']
                 == 'TransactionType.TRANSFER']

    # Create timeline plot - now only one subplot
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 6))
    fig.suptitle(f'Repeated Overseas Transfers Timeline: {pattern["pattern_id"]}',
                 fontsize=16, fontweight='bold')

    # Plot 1: Transaction amounts over time
    if deposits:
        deposit_times = [tx['datetime'] for tx in deposits]
        deposit_amounts = [tx['amount'] for tx in deposits]
        ax1.plot(deposit_times, deposit_amounts, 'o', color='purple',
                 markersize=8, alpha=0.7, label=f'Cash Deposits ({len(deposits)} txs)')

    if transfers:
        transfer_times = [tx['datetime'] for tx in transfers]
        transfer_amounts = [tx['amount'] for tx in transfers]
        ax1.plot(transfer_times, transfer_amounts, 'o', color='darkblue',
                 markersize=8, alpha=0.7, label=f'Overseas Transfers ({len(transfers)} txs)')

    ax1.set_ylabel('Amount (€)', fontsize=12)
    ax1.set_title('Transaction Amounts Over Time',
                  fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Formatting X-axis
    # Calculate appropriate interval based on time span
    all_times = [tx['datetime'] for tx in transactions]
    time_span = max(all_times) - min(all_times)
    if time_span.days <= 7:
        # For spans up to a week, show daily ticks
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        date_format = '%Y-%m-%d\n%H:%M'
    elif time_span.days <= 31:
        # For spans up to a month, show weekly ticks
        ax1.xaxis.set_major_locator(
            mdates.WeekdayLocator(byweekday=0))  # Monday
        date_format = '%Y-%m-%d'
    else:
        # For longer spans, show monthly ticks
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        date_format = '%Y-%m'

    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Add phase annotations
    if deposits and transfers:
        deposit_end = max(tx['datetime'] for tx in deposits)
        transfer_start = min(tx['datetime'] for tx in transfers)

        ax1.axvspan(min(all_times), deposit_end, color='plum',
                    alpha=0.2, label='Deposit Phase')
        ax1.axvspan(transfer_start, max(all_times), color='lightblue',
                    alpha=0.2, label='Transfer Phase')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('repeated_overseas_transfers_timeline.png',
                dpi=300, bbox_inches='tight')
    print("✓ Timeline visualization saved as 'repeated_overseas_transfers_timeline.png'")
    plt.show()


def main():
    """Main function to generate visualizations"""
    print("=== RepeatedOverseasTransfers Pattern Visualization ===")

    # Generate pattern data
    pattern_data, nodes_df, edges_df = generate_pattern_data()
    if pattern_data is None:
        print("Failed to generate data. Exiting.")
        return

    # Filter for the correct pattern type
    rot_patterns = [p for p in pattern_data['patterns']
                    if p['pattern_type'] == 'RepeatedOverseasTransfers']
    print(
        f"Generated {len(pattern_data['patterns'])} total patterns, found {len(rot_patterns)} RepeatedOverseasTransfers patterns.")

    if not rot_patterns:
        print("No RepeatedOverseasTransfers patterns were generated. Check your config.")
        return

    # Create visualizations
    print("\nCreating network topology visualization...")
    pattern = create_network_visualization(pattern_data, nodes_df, edges_df)

    if pattern:
        print("\nCreating timeline visualization...")
        create_timeline_visualization(pattern, nodes_df, edges_df)

        print("\nVisualizations saved:")
        print("- repeated_overseas_transfers_network.png")
        print("- repeated_overseas_transfers_timeline.png")


if __name__ == "__main__":
    main()
