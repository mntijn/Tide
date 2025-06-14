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
    fig.suptitle(f'FrontBusinessActivity Pattern Structure: {pattern["pattern_id"]}',
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

    # Find the front business and organize entities
    front_business = None
    front_business_accounts = []
    overseas_business_accounts = []
    cash_account = 'account_9459'

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
            # Check if account is owned by front business (receives deposits)
            is_front_account = any(tx['dest'] == entity_id and tx['transaction_type'] == 'TransactionType.DEPOSIT'
                                   for tx in pattern_transactions)

            if is_front_account:
                front_business_accounts.append(entity_id)
            else:
                # Must be overseas destination account
                overseas_business_accounts.append(entity_id)

    print(f"Front business: {front_business}")
    print(f"Front business accounts: {front_business_accounts}")
    print(f"Overseas accounts: {overseas_business_accounts}")

    # Create structured horizontal layout for clear flow visualization
    pos = {}

    # Position front business on the left
    pos[front_business] = (-3, 0)

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
            pos[account] = (4.5, y_start - i * y_spacing)

    # Position cash account on the far left (when it exists)
    if cash_account in entity_info:
        pos[cash_account] = (-6, 0)

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

    # Draw nodes with clear styling
    for entity in G.nodes():
        info = entity_info.get(entity, {})
        entity_type = info.get('type', 'Unknown')
        country = info.get('country', 'Unknown')

        # Determine node color and style
        if entity == front_business:
            color = 'yellow'
            size = 4000
        elif entity == cash_account:
            color = 'red'
            size = 2000
        elif entity in front_business_accounts:
            color = 'orange'
            size = 2500
        elif entity in overseas_business_accounts:
            color = 'lightgreen'
            size = 2500
        else:
            color = 'gray'
            size = 1500

        # Draw node
        ax.scatter(pos[entity][0], pos[entity][1], c=color, s=size,
                   alpha=0.8, edgecolors='black', linewidth=2, zorder=3)

        # Create clear label matching the example format
        if entity == cash_account:
            label = 'Cash System'
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

        # Position label
        ax.text(pos[entity][0], pos[entity][1], label,
                ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

    # Draw edges with detailed labels
    for (src, dest), edge_data in edges_info.items():
        src_pos = pos[src]
        dest_pos = pos[dest]

        # Determine line style and color
        tx_type = edge_data['type']
        if 'DEPOSIT' in tx_type:
            color = 'purple'
            linestyle = '--'  # Dashed for deposits
            alpha = 0.8
        elif 'TRANSFER' in tx_type:
            color = 'blue'
            linestyle = '-'   # Solid for transfers
            alpha = 0.8
        else:
            color = 'gray'
            linestyle = '-'
            alpha = 0.6

        # Draw arrow
        ax.annotate('', xy=dest_pos, xytext=src_pos,
                    arrowprops=dict(arrowstyle='->', color=color, lw=3,
                                    linestyle=linestyle, alpha=alpha),
                    zorder=1)

        # Add transaction details as label (simplified like example)
        total_amount = edge_data['total_amount']
        count = edge_data['count']

        # Create simplified label showing transaction type and main details
        if count == 1:
            tx = edge_data['transactions'][0]
            amount = tx['amount']
            timestamp = tx['timestamp']
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', ''))
            time_str = dt.strftime('%H:%M, %d-%m')

            if 'DEPOSIT' in tx_type:
                label_text = f"Cash deposit\n€{amount:,.0f}\n{time_str}"
            else:
                label_text = f"Transfer\n€{amount:,.0f}\n{time_str}"
        else:
            # Multiple transactions - show summary
            avg_amount = total_amount / count
            if 'DEPOSIT' in tx_type:
                label_text = f"Cash deposits\n€{avg_amount:,.0f} avg\n({count} txs)"
            else:
                label_text = f"Transfers\n€{avg_amount:,.0f} avg\n({count} txs)"

        # Position label along the edge
        mid_x = (src_pos[0] + dest_pos[0]) / 2
        mid_y = (src_pos[1] + dest_pos[1]) / 2

        # Offset label slightly to avoid overlapping with arrow
        offset_x = 0.2 if abs(
            src_pos[0] - dest_pos[0]) > abs(src_pos[1] - dest_pos[1]) else 0
        offset_y = 0.2 if abs(
            src_pos[1] - dest_pos[1]) > abs(src_pos[0] - dest_pos[0]) else 0

        ax.text(mid_x + offset_x, mid_y + offset_y, label_text,
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=color, alpha=0.9),
                zorder=2)

        # Add transaction type label
        type_label = tx_type.replace('TransactionType.', '').title()
        ax.text(mid_x - offset_x, mid_y - offset_y, type_label,
                ha='center', va='center', fontsize=8, fontweight='bold',
                color=color, zorder=2)

    # Set axis properties
    ax.set_xlim(-6, 8)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                   markersize=15, label='Front Business'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=15, label='Business Accounts'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                   markersize=15, label='Overseas Accounts'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=15, label='Cash System'),
        plt.Line2D([0], [0], color='purple', linewidth=3,
                   linestyle='--', label='Cash Deposits'),
        plt.Line2D([0], [0], color='blue', linewidth=3,
                   label='Overseas Transfers')
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    # Add summary statistics
    total_deposits = sum(tx['amount'] for tx in pattern_transactions
                         if tx['transaction_type'] == 'TransactionType.DEPOSIT')
    total_transfers = sum(tx['amount'] for tx in pattern_transactions
                          if tx['transaction_type'] == 'TransactionType.TRANSFER')
    ratio = total_transfers / total_deposits if total_deposits > 0 else 0

    stats_text = (f"Pattern Summary:\n"
                  f"Total Deposits: €{total_deposits:,.0f}\n"
                  f"Total Transfers: €{total_transfers:,.0f}\n"
                  f"Transfer Ratio: {ratio:.1%}\n"
                  f"Front Business: {front_business_country}\n"
                  f"Entities: {len(pattern_entities)}\n"
                  f"Transactions: {len(pattern_transactions)}")

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
                                               facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig('front_business_network.png', dpi=300, bbox_inches='tight')
    print("✓ Network topology visualization saved as 'front_business_network.png'")
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

    # Separate deposits and transfers
    deposits = [tx for tx in transactions if tx['transaction_type']
                == 'TransactionType.DEPOSIT']
    transfers = [tx for tx in transactions if tx['transaction_type']
                 == 'TransactionType.TRANSFER']

    # Create timeline plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'FrontBusinessActivity Pattern Timeline: {pattern["pattern_id"]}',
                 fontsize=16, fontweight='bold')

    # Plot 1: Transaction amounts over time
    if deposits:
        deposit_times = [tx['datetime'] for tx in deposits]
        deposit_amounts = [tx['amount'] for tx in deposits]
        ax1.scatter(deposit_times, deposit_amounts, color='purple', s=100, alpha=0.7,
                    marker='s', label=f'Deposits ({len(deposits)} txns)')

    if transfers:
        transfer_times = [tx['datetime'] for tx in transfers]
        transfer_amounts = [tx['amount'] for tx in transfers]
        ax1.scatter(transfer_times, transfer_amounts, color='blue', s=100, alpha=0.7,
                    marker='o', label=f'Transfers ({len(transfers)} txns)')

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
    cumulative_deposits = []
    cumulative_transfers = []
    current_deposits = 0
    current_transfers = 0

    for tx in transactions:
        if tx['transaction_type'] == 'TransactionType.DEPOSIT':
            current_deposits += tx['amount']
        else:
            current_transfers += tx['amount']
        cumulative_deposits.append(current_deposits)
        cumulative_transfers.append(current_transfers)

    ax2.plot(all_times, cumulative_deposits, color='purple', linewidth=2,
             label=f'Cumulative Deposits (€{current_deposits:,.0f})')
    ax2.plot(all_times, cumulative_transfers, color='blue', linewidth=2,
             label=f'Cumulative Transfers (€{current_transfers:,.0f})')

    # Add ratio thresholds
    ratio = current_transfers / current_deposits if current_deposits > 0 else 0
    ax2.axhline(y=current_deposits * 0.80, color='green', linestyle='--', alpha=0.5,
                label=f'80% threshold (€{current_deposits * 0.80:,.0f})')
    ax2.axhline(y=current_deposits * 1.00, color='green', linestyle='--', alpha=0.5,
                label=f'100% threshold (€{current_deposits * 1.00:,.0f})')

    ax2.set_ylabel('Cumulative Amount (€)', fontsize=12)
    ax2.set_title(
        f'Cumulative Flow (Transfer Ratio: {ratio:.1%})', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # Plot 3: Transaction frequency over time (hourly bins)
    if transactions:
        start_time = min(all_times)
        end_time = max(all_times)
        duration = end_time - start_time

        # Create hourly bins
        hours = max(1, int(duration.total_seconds() / 3600) + 1)
        time_bins = [start_time +
                     datetime.timedelta(hours=i) for i in range(hours + 1)]

        deposit_counts = [0] * hours
        transfer_counts = [0] * hours

        for tx in transactions:
            hour_index = int(
                (tx['datetime'] - start_time).total_seconds() / 3600)
            if hour_index < hours:
                if tx['transaction_type'] == 'TransactionType.DEPOSIT':
                    deposit_counts[hour_index] += 1
                else:
                    transfer_counts[hour_index] += 1

        # Create bar chart
        bar_times = time_bins[:-1]
        width = datetime.timedelta(hours=0.4)

        ax3.bar([t - width/2 for t in bar_times], deposit_counts, width=width,
                color='purple', alpha=0.7, label='Deposits/hour')
        ax3.bar([t + width/2 for t in bar_times], transfer_counts, width=width,
                color='blue', alpha=0.7, label='Transfers/hour')

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
    if deposits and transfers:
        deposit_end = max(tx['datetime'] for tx in deposits)
        transfer_start = min(tx['datetime'] for tx in transfers)

        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=deposit_end, color='gray', linestyle=':', alpha=0.7)
            ax.axvline(x=transfer_start, color='gray',
                       linestyle=':', alpha=0.7)

        # Add phase labels
        start_time = min(all_times)
        end_time = max(all_times)

        deposit_mid = start_time + (deposit_end - start_time) / 2
        transfer_mid = transfer_start + (end_time - transfer_start) / 2

        ax1.text(deposit_mid, ax1.get_ylim()[1] * 0.9, 'Deposit Phase',
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.8))
        ax1.text(transfer_mid, ax1.get_ylim()[1] * 0.9, 'Transfer Phase',
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

        # Calculate and display delay
        delay_hours = (transfer_start - deposit_end).total_seconds() / 3600
        ax1.text(deposit_end + (transfer_start - deposit_end) / 2, ax1.get_ylim()[1] * 0.8,
                 f'Delay: {delay_hours:.1f}h',
                 ha='center', va='center', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))

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
