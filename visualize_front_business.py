#!/usr/bin/env python3
"""
FrontBusinessActivity Pattern Visualization

Creates two visualizations for the FrontBusinessActivity pattern:
1. Network topology showing front business, its accounts, and overseas transfers
2. Timeline showing temporal pattern of deposits followed by transfers
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


def generate_pattern_data():
    """Generate pattern data for visualization"""
    print("Generating FrontBusinessActivity pattern data...")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "viz_config.yaml")
        patterns_file = os.path.join(temp_dir, "generated_patterns.json")
        nodes_file = os.path.join(temp_dir, "generated_nodes.csv")
        edges_file = os.path.join(temp_dir, "generated_edges.csv")

        # Create config file
        config = create_h1_config()
        config['random_seed'] = 69
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
    # Tufte-inspired styling and colorblind-friendly palette
    sns.set_style("white")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    color_palette = {
        'front_business': '#FFC107',    # Yellow
        'cash_system': '#D81B60',       # Pink/Red
        'business_accounts': '#1E88E5',  # Blue
        'overseas_accounts': '#004D40',  # Dark Green
        'deposits': '#D81B60',
        'transfers': '#1E88E5'
    }

    # Focus on FrontBusinessActivity patterns
    fb_patterns = [p for p in pattern_data['patterns']
                   if p['pattern_type'] == 'FrontBusinessActivity']

    if not fb_patterns:
        print("No FrontBusinessActivity patterns found!")
        return

    # Use the first pattern for visualization
    pattern = fb_patterns[0]
    print(f"Visualizing pattern: {pattern['pattern_id']}")

    # Create a single, clear network diagram
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_title(f'FrontBusinessActivity Pattern: {pattern["pattern_id"]}',
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

    # Find the front business and organize entities
    front_business = None
    front_business_accounts = []
    overseas_business_accounts = []

    # Dynamically find the cash source from deposit transactions
    cash_sources = set()

    # Identify front business (should be in pattern entities and type BUSINESS)
    for entity_id in pattern_entities:
        info = entity_info.get(entity_id, {})
        if info.get('type') == 'BUSINESS':
            front_business = entity_id
            break

    # If no business found directly in pattern entities, find it by account ownership
    if not front_business:
        for _, node in nodes_df.iterrows():
            if node['node_type'] == 'NodeType.BUSINESS':
                business_id = node['node_id']
                # Check if this business owns any accounts that receive deposits in this pattern
                for entity_id in pattern_entities:
                    if entity_info.get(entity_id, {}).get('type') == 'ACCOUNT':
                        for tx in pattern_transactions:
                            if tx['dest'] == entity_id and tx['transaction_type'] == 'TransactionType.DEPOSIT':
                                # Found business that owns accounts receiving deposits
                                entity_info[business_id] = {
                                    'type': 'BUSINESS',
                                    'country': node.get('country_code', 'Unknown'),
                                    'currency': node.get('currency', 'Unknown')
                                }
                                front_business = business_id
                                break
                if front_business:
                    break

    if not front_business:
        print("No front business found in pattern!")
        return

    front_business_country = entity_info[front_business]['country']

    # Categorize accounts
    for entity_id in pattern_entities:
        if entity_id == front_business:
            continue

        info = entity_info.get(entity_id, {})
        if info.get('type') == 'ACCOUNT':
            # Check if account receives deposits (making it a front business account)
            is_front_account = any(tx['dest'] == entity_id and tx['transaction_type'] == 'TransactionType.DEPOSIT'
                                   for tx in pattern_transactions)

            if is_front_account:
                front_business_accounts.append(entity_id)
                # The source of these deposits is the cash system
                for tx in pattern_transactions:
                    if tx['dest'] == entity_id and tx['transaction_type'] == 'TransactionType.DEPOSIT':
                        cash_sources.add(tx['src'])
            else:
                # Must be overseas destination account
                overseas_business_accounts.append(entity_id)

    # Sort accounts for deterministic layout
    front_business_accounts.sort()
    overseas_business_accounts.sort()

    # Assuming a single cash source for visualization clarity
    cash_account = sorted(list(cash_sources))[
        0] if cash_sources else 'cash_system_fallback'
    if not cash_sources:
        entity_info[cash_account] = {'type': 'CASH_SYSTEM', 'country': ''}

    # Create structured horizontal layout for clear flow visualization
    pos = {}

    # Position front business and cash account on the left
    pos[front_business] = (-6, 0.5)
    pos[cash_account] = (-6, 0.5)

    # Position front business accounts in the middle (vertically distributed)
    num_front_accounts = len(front_business_accounts)
    if num_front_accounts > 0:
        y_spacing = 3.0 if num_front_accounts > 1 else 0
        y_start = (num_front_accounts - 1) * y_spacing / 2
        for i, account in enumerate(front_business_accounts):
            pos[account] = (0, y_start - i * y_spacing)

    # Position overseas accounts on the right side (vertically distributed)
    num_overseas = len(overseas_business_accounts)
    print('num_overseas', num_overseas)
    if num_overseas > 0:
        y_spacing = 2.5 if num_overseas > 1 else 0
        y_start = (num_overseas - 1) * y_spacing / 2
        for i, account in enumerate(overseas_business_accounts):
            if account != cash_account:
                pos[account] = (4.5, y_start - i * y_spacing)

    # Create the graph
    G = nx.DiGraph()

    # Add all entities as nodes
    all_entities = [front_business] + front_business_accounts + \
        overseas_business_accounts + [cash_account]
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

    # Draw nodes with clear styling, ensuring cash_account is drawn on top of front_business
    node_list = list(G.nodes())
    print("front business id :", front_business)
    print("cash account id   :", cash_account)
    print("pos[fb]           :", pos[front_business])
    print("pos[cash]         :", pos[cash_account])
    if front_business in node_list:
        node_list.remove(front_business)
        node_list.insert(0, front_business)  # Draw first
    if cash_account in node_list:
        node_list.remove(cash_account)
        # Insert after front_business to be drawn on top
        node_list.insert(node_list.index(front_business) + 1, cash_account)

    for entity in node_list:
        info = entity_info.get(entity, {})
        entity_type = info.get('type', 'Unknown')
        country = info.get('country', 'Unknown')

        # Determine node color and style
        if entity == front_business:
            color = color_palette['front_business']
            size = 4000
        elif entity == cash_account:
            color = color_palette['cash_system']
            size = 2000
        elif entity in front_business_accounts:
            color = color_palette['business_accounts']
            size = 2500
        elif entity in overseas_business_accounts:
            color = color_palette['overseas_accounts']
            size = 2500
        else:
            color = 'gray'
            size = 1500

        # Draw node
        ax.scatter(pos[entity][0], pos[entity][1], c=color, s=size,
                   alpha=0.9, edgecolors='black', linewidth=1.5)

        # Create clear label matching the example format
        if entity == cash_account:
            label = ''
        elif entity == front_business:
            bus_id = entity.split('_')[1] if '_' in entity else '1'
            label = f"Business_{bus_id}\n[{country}]"
        else:
            # For accounts, show simplified labels
            if entity.startswith('account_'):
                if entity in front_business_accounts:
                    # Number front business accounts sequentially
                    acc_num = front_business_accounts.index(entity) + 1
                    label = f"Account_{acc_num}\n[{country}]"
                else:
                    # Number overseas accounts sequentially
                    acc_num = overseas_business_accounts.index(entity) + 1
                    label = f"Business_{acc_num}_account\n[{country}]"
            else:
                label = f"{entity}\n[{country}]"

        # Position label above the node for readability
        label_y_offset = 0.5
        ax.text(pos[entity][0], pos[entity][1] + label_y_offset, label,
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Draw edges and prepare for consolidated labels
    deposit_midpoints = []
    transfer_midpoints = []
    deposit_color = color_palette['deposits']
    transfer_color = color_palette['transfers']

    for (src, dest), edge_data in edges_info.items():
        src_pos = pos[src]
        dest_pos = pos[dest]

        # Determine line style and color
        tx_type = edge_data['type']
        if 'DEPOSIT' in tx_type:
            color = deposit_color
            linestyle = '--'  # Dashed for deposits
            alpha = 0.8
            deposit_midpoints.append(
                ((src_pos[0] + dest_pos[0]) / 2, (src_pos[1] + dest_pos[1]) / 2))
        elif 'TRANSFER' in tx_type:
            color = transfer_color
            linestyle = '-'   # Solid for transfers
            alpha = 0.8
            transfer_midpoints.append(
                ((src_pos[0] + dest_pos[0]) / 2, (src_pos[1] + dest_pos[1]) / 2))
        else:
            color = 'gray'
            linestyle = '-'
            alpha = 0.6

        # Draw arrow with adjusted width for visibility
        ax.annotate('', xy=dest_pos, xytext=src_pos,
                    arrowprops=dict(arrowstyle='->', color=color, lw=3,
                                    linestyle=linestyle, alpha=alpha,
                                    shrinkA=35, shrinkB=35))

    # Draw a single, consolidated label for all deposits
    if deposit_midpoints:
        avg_x = sum(p[0] for p in deposit_midpoints) / len(deposit_midpoints)
        avg_y = sum(p[1] for p in deposit_midpoints) / len(deposit_midpoints)
        offset_y = 0.3
        ax.text(avg_x, avg_y + offset_y, "Deposits",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFFCC", edgecolor=deposit_color, alpha=0.9))

    # Draw a single, consolidated label for all transfers
    if transfer_midpoints:
        avg_x = sum(p[0] for p in transfer_midpoints) / len(transfer_midpoints)
        avg_y = sum(p[1] for p in transfer_midpoints) / len(transfer_midpoints)
        offset_y = 0.3
        ax.text(avg_x, avg_y + offset_y, "Transfers",
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFFCC", edgecolor=transfer_color, alpha=0.9))

    # Set axis properties dynamically based on node positions so nothing is cut off
    all_x = [coord[0] for coord in pos.values()]
    all_y = [coord[1] for coord in pos.values()]
    x_margin, y_margin = 2, 2
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add summary statistics
    total_deposits = sum(tx['amount'] for tx in pattern_transactions
                         if tx['transaction_type'] == 'TransactionType.DEPOSIT')
    total_transfers = sum(tx['amount'] for tx in pattern_transactions
                          if tx['transaction_type'] == 'TransactionType.TRANSFER')
    ratio = total_transfers / total_deposits if total_deposits > 0 else 0

    plt.tight_layout(pad=0)
    plt.savefig('front_business_network.png', dpi=300, bbox_inches='tight')
    print("✓ Network topology visualization saved as 'front_business_network.png'")
    plt.show()

    return pattern


def create_timeline_visualization(pattern_data, pattern):
    """Create timeline visualization of transactions"""
    # Tufte-inspired styling and colorblind-friendly palette
    sns.set_style("ticks")
    color_palette = {
        'deposits': '#D81B60',
        'transfers': '#1E88E5'
    }

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

    # Create timeline plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    ax.set_title(f'FrontBusinessActivity Pattern Timeline: {pattern["pattern_id"]}',
                 fontsize=18, fontweight='bold')

    # Plot 1: Transaction amounts over time
    if deposits:
        deposit_times = [tx['datetime'] for tx in deposits]
        deposit_amounts = [tx['amount'] for tx in deposits]
        ax.scatter(deposit_times, deposit_amounts, color=color_palette['deposits'], s=100, alpha=0.7,
                   marker='s', label=f'Deposits ({len(deposits)} txns)')

    if transfers:
        transfer_times = [tx['datetime'] for tx in transfers]
        transfer_amounts = [tx['amount'] for tx in transfers]
        ax.scatter(transfer_times, transfer_amounts, color=color_palette['transfers'], s=100, alpha=0.7,
                   marker='o', label=f'Transfers ({len(transfers)} txns)')

    ax.set_ylabel('Amount (€)', fontsize=14)
    ax.set_title('Transaction Amounts Over Time',
                 fontsize=16, fontweight='bold')
    ax.grid(False)
    ax.legend(frameon=False, fontsize=12)
    # Show only first and last (and optionally midpoint) timestamps to avoid clutter
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    all_times = [tx['datetime'] for tx in transactions]
    if all_times:
        start_time = min(all_times)
        end_time = max(all_times)

        # Set axis limits to start at 0
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=start_time)

        # Always include start and end
        xticks = [start_time, end_time]
        # Optionally include midpoint if the range is wide enough (> 1 day)
        if (end_time - start_time).total_seconds() > 24 * 3600:
            mid_time = start_time + (end_time - start_time) / 2
            xticks.insert(1, mid_time)
        ax.set_xticks(xticks)
        plt.setp(ax.xaxis.get_majorticklabels(),
                 rotation=0, ha='center', fontsize=12)
    else:
        ax.set_xticks([])

    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig('front_business_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ Timeline visualization saved as 'front_business_timeline.png'")
    plt.show()


def main():
    """Main function to generate visualizations"""
    print("=== FrontBusinessActivity Pattern Visualization ===")

    # Generate pattern data
    pattern_data, nodes_df, edges_df = generate_pattern_data()
    if pattern_data is None:
        return

    print(f"Generated {len(pattern_data['patterns'])} patterns")

    # Create visualizations
    print("\nCreating network topology visualization...")
    pattern = create_network_visualization(pattern_data, nodes_df, edges_df)

    if pattern:
        print("\nCreating timeline visualization...")
        create_timeline_visualization(pattern_data, pattern)

        print("\nVisualizations saved:")
        print("- front_business_network.png")
        print("- front_business_timeline.png")


if __name__ == "__main__":
    main()
