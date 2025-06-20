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
import seaborn as sns

# Add the parent directory to path to import main
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))


def set_visualization_style():
    """Applies Tufte's principles and accessibility guidelines for visualizations."""
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("ticks")
    # Use a specific, colorblind-friendly palette for consistency
    colors = ['#1E88E5', '#D81B60', '#004D40', '#FFC107', '#FFC107']
    sns.set_palette(sns.color_palette(colors))

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "Bitstream Vera Sans", "sans-serif"],
    })
    return sns.color_palette(colors)


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
        config['random_seed'] = 69
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


def create_network_visualization(pattern_data, nodes_df, edges_df, colors):
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
    overseas_accounts.sort()

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

    # Create structured horizontal layout for clear flow visualization
    pos = {}
    pos[central_entity] = (-4, 0)
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
    all_graph_nodes = [central_entity, domestic_account] + overseas_accounts
    G.add_nodes_from(all_graph_nodes)

    # Draw nodes with clear styling inspired by the example image
    for entity in G.nodes():
        info = entity_info.get(entity, {})
        entity_type = info.get('type', 'Unknown')
        country = info.get('country', 'Unknown')

        size = 3000

        # Determine node color and style based on role
        if entity == central_entity:
            color = colors[0]  # Central Entity
        elif entity == domestic_account:
            color = colors[3]  # Domestic Account
        elif entity in overseas_accounts:
            color = colors[2]  # Overseas Account
        else:
            color = 'gray'

        ax.scatter(pos[entity][0], pos[entity][1], c=color, s=size,
                   alpha=0.9, edgecolors='black', linewidth=1.5, zorder=3)

        # Create clear label
        if entity == central_entity:
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

        # Position label above the node
        label_y_offset = 0.6
        ax.text(pos[entity][0], pos[entity][1] + label_y_offset, label,
                ha='center', va='bottom', fontsize=18, fontweight='bold', zorder=4)

    # --- Draw Edges ---
    # 1. Edge from Cash System to Domestic Account (Deposits)
    total_deposit_amount = sum(tx['amount'] for tx in deposit_txs)
    ax.annotate('', xy=pos[domestic_account], xytext=pos[central_entity],
                arrowprops=dict(arrowstyle='->', color=colors[1], lw=3,
                                linestyle='--', alpha=0.8, shrinkA=35, shrinkB=35), zorder=1)
    mid_x = (pos[central_entity][0] + pos[domestic_account][0]) / 2
    mid_y = (pos[central_entity][1] + pos[domestic_account][1]) / 2
    label_text = "Deposits"
    ax.text(mid_x, mid_y + 0.4, label_text, ha='center', va='center', fontsize=16,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=colors[1], alpha=0.9), zorder=2)

    # 2. Edges from Domestic Account to Overseas Accounts
    for account in overseas_accounts:
        ax.annotate('', xy=pos[account], xytext=pos[domestic_account],
                    arrowprops=dict(arrowstyle='->', color=colors[0], lw=3,
                                    linestyle='-', alpha=0.8, shrinkA=35, shrinkB=35), zorder=1)
        mid_x = (pos[domestic_account][0] + pos[account][0]) / 2
        mid_y = (pos[domestic_account][1] + pos[account][1]) / 2
        label_text = "Transfers"
        ax.text(mid_x, mid_y + 0.4, label_text, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=colors[0], alpha=0.9), zorder=2)

    # Set axis properties
    ax.set_xlim(-5, 6)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0],
                   markersize=15, label='Central Entity (Owner)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[3],
                   markersize=15, label='Domestic Source Account'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2],
                   markersize=15, label='Overseas Destination Accounts'),
        plt.Line2D([0], [0], color=colors[1], linewidth=3,
                   linestyle='--', label='Cash Deposits'),
        plt.Line2D([0], [0], color=colors[0], linewidth=3,
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


def create_timeline_visualization(pattern, nodes_df, edges_df, colors):
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
        ax1.scatter(deposit_times, deposit_amounts, color=colors[1],
                    s=100, alpha=0.7, label=f'Cash Deposits ({len(deposits)} txs)')

    if transfers:
        transfer_times = [tx['datetime'] for tx in transfers]
        transfer_amounts = [tx['amount'] for tx in transfers]
        ax1.scatter(transfer_times, transfer_amounts, color=colors[0],
                    s=100, alpha=0.7, label=f'Overseas Transfers ({len(transfers)} txs)')

    ax1.set_ylabel('Amount (€)', fontsize=12)
    ax1.set_title('Transaction Amounts Over Time',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(False)
    sns.despine(ax=ax1)

    # Formatting X-axis
    all_times = [tx['datetime'] for tx in transactions]
    start_time = min(all_times)
    end_time = max(all_times)
    ax1.set_xticks([start_time, end_time])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center')

    # Add phase annotations
    if deposits and transfers:
        deposit_end = max(tx['datetime'] for tx in deposits)
        transfer_start = min(tx['datetime'] for tx in transfers)

        ax1.axvline(x=deposit_end, color='gray', linestyle=':', alpha=0.7)
        ax1.axvline(x=transfer_start, color='gray', linestyle=':', alpha=0.7)
        deposit_mid = start_time + (deposit_end - start_time) / 2
        transfer_mid = transfer_start + (end_time - transfer_start) / 2
        ax1.text(deposit_mid, ax1.get_ylim()[
                 1] * 0.9, 'Deposit Phase', ha='center', va='center', fontsize=12, fontweight='bold', color=colors[1])
        ax1.text(transfer_mid, ax1.get_ylim()[
                 1] * 0.9, 'Transfer Phase', ha='center', va='center', fontsize=12, fontweight='bold', color=colors[0])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('repeated_overseas_transfers_timeline.png',
                dpi=300, bbox_inches='tight')
    print("✓ Timeline visualization saved as 'repeated_overseas_transfers_timeline.png'")
    plt.show()


def main():
    """Main function to generate visualizations"""
    print("=== RepeatedOverseasTransfers Pattern Visualization ===")
    colors = set_visualization_style()

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
    pattern = create_network_visualization(
        pattern_data, nodes_df, edges_df, colors)

    if pattern:
        print("\nCreating timeline visualization...")
        create_timeline_visualization(pattern, nodes_df, edges_df, colors)

        print("\nVisualizations saved:")
        print("- repeated_overseas_transfers_network.png")
        print("- repeated_overseas_transfers_timeline.png")


if __name__ == "__main__":
    main()
