#!/usr/bin/env python3
"""
UTurnTransactions Pattern Visualization

Creates two visualizations for the UTurnTransactions pattern:
1. Network topology showing the U-turn transaction path through intermediary accounts.
2. Timeline showing temporal pattern of transactions and delays.
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
import math

# Add the parent directory to path to import main
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))


def create_viz_config():
    """Create configuration for UTurnTransactions visualization"""
    config = create_h1_config()  # Start with base config
    config['pattern_frequency'] = {
        'random': False,
        'RapidFundMovement': 0,
        'FrontBusinessActivity': 0,
        'RepeatedOverseasTransfers': 0,
        'UTurnTransactions': 1  # Generate one U-Turn pattern
    }
    # Ensure there are enough intermediaries
    config['pattern_config']['uTurnTransactions']['min_intermediaries'] = 3
    config['pattern_config']['uTurnTransactions']['max_intermediaries'] = 5
    return config


def generate_pattern_data():
    """Generate pattern data for visualization"""
    print("Generating UTurnTransactions pattern data...")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        config_file = os.path.join(temp_dir, "viz_config.yaml")
        patterns_file = os.path.join(temp_dir, "generated_patterns.json")
        nodes_file = os.path.join(temp_dir, "generated_nodes.csv")
        edges_file = os.path.join(temp_dir, "generated_edges.csv")

        # Create config file
        config = create_viz_config()
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
            return None, None, None, None

        # Load pattern data
        with open(patterns_file, 'r') as f:
            pattern_data = json.load(f)

        # Load node and edge data
        nodes_df = pd.read_csv(nodes_file)
        edges_df = pd.read_csv(edges_file)

        return pattern_data, nodes_df, edges_df, config


def find_originator(account_id, edges_df, nodes_df):
    """Find the owner of an account."""
    print("\n--- Debugging find_originator ---")
    print(f"Searching for owner of account: {account_id}")

    # Check available edge types
    available_edge_types = edges_df['edge_type'].unique()
    print(f"Available edge types in data: {available_edge_types}")
    if 'EdgeType.OWNERSHIP' not in available_edge_types:
        print("ERROR: 'EdgeType.OWNS_ACCOUNT' not found in edge types.")

    # Filter for ownership edges for the specific account
    owner_edge = edges_df[(edges_df['dest'] == account_id) & (
        edges_df['edge_type'] == 'EdgeType.OWNERSHIP')]

    if owner_edge.empty:
        print(
            f"RESULT: Could not find an 'EdgeType.OWNS_ACCOUNT' edge where dest is '{account_id}'.")
        print("This means the originator entity cannot be identified from the data.")
        print("--- End Debugging ---\n")
        return None, None

    print("SUCCESS: Found ownership edge.")
    owner_id = owner_edge.iloc[0]['src']
    print(f"Owner entity ID from edge: {owner_id}")

    owner_info_series = nodes_df[nodes_df['node_id'] == owner_id]
    if owner_info_series.empty:
        print(
            f"ERROR: Found owner ID '{owner_id}' in edges, but this ID is not in the nodes file.")
        print("--- End Debugging ---\n")
        return None, None

    owner_info = owner_info_series.iloc[0]
    owner_type = owner_info['node_type']
    print(
        f"SUCCESS: Found owner '{owner_id}' in nodes file. Type: {owner_type}")
    print("--- End Debugging ---\n")

    return owner_id, {
        'type': owner_type,
        'country': owner_info.get('country_code', 'Unknown')
    }


def create_network_visualization(pattern_data, nodes_df, edges_df, config):
    """Create network topology visualization"""
    utt_patterns = [p for p in pattern_data['patterns']
                    if p['pattern_type'] == 'UTurnTransactions']

    if not utt_patterns:
        print("No UTurnTransactions patterns found!")
        return None

    pattern = utt_patterns[0]
    print(f"Visualizing pattern: {pattern['pattern_id']}")

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.suptitle(
        f'UTurnTransactions Pattern Structure: {pattern["pattern_id"]}', fontsize=16, fontweight='bold')

    transactions = sorted(pattern['transactions'],
                          key=lambda tx: tx['timestamp'])

    if not transactions:
        print("Pattern has no transactions.")
        return None

    # Reconstruct the path
    path_with_duplicates = []
    if transactions:
        path_with_duplicates.append(transactions[0]['src'])
        for tx in transactions:
            path_with_duplicates.append(tx['dest'])

    path = list(dict.fromkeys(path_with_duplicates))

    source_account = path[0]
    return_account = path[-1]
    intermediary_accounts = path[1:-1]

    # Get entity information
    entity_info = {}
    for _, node in nodes_df.iterrows():
        if node['node_id'] in path:
            entity_info[node['node_id']] = {
                'type': node['node_type'],
                'country': node.get('country_code', 'Unknown'),
                'currency': node.get('currency', 'Unknown')
            }

    # Find originator
    originator_id, originator_info = find_originator(
        source_account, edges_df, nodes_df)
    if originator_id:
        entity_info[originator_id] = originator_info

    # Create layout
    pos = {}
    if originator_id:
        pos[originator_id] = (0, 2.5)

    # Position accounts in a U-shape
    num_accounts = len(path)
    radius = 4
    for i, acc_id in enumerate(path):
        angle = math.pi + i * (math.pi / (num_accounts - 1))
        pos[acc_id] = (radius * math.cos(angle),
                       radius * math.sin(angle) * 0.7)

    G = nx.DiGraph()

    # Add nodes
    all_nodes = path
    if originator_id:
        all_nodes.append(originator_id)

    for node_id in all_nodes:
        G.add_node(node_id)

    if originator_id:
        G.add_edge(originator_id, source_account, type='ownership')
        G.add_edge(originator_id, return_account, type='ownership')

    for tx in transactions:
        G.add_edge(tx['src'], tx['dest'], type='transaction', **tx)

    # Draw nodes
    for node_id in G.nodes():
        node_info = entity_info.get(node_id, {})
        country = node_info.get('country', 'Unknown')
        node_type = node_info.get('type', 'Unknown')

        if node_id == originator_id:
            # Different colors for business vs individual
            if 'BUSINESS' in node_type:
                color = 'lightgreen'  # Green for businesses
                entity_type = "Business"
            elif 'INDIVIDUAL' in node_type:
                color = 'yellow'  # Yellow for individuals
                entity_type = "Individual"
            else:
                color = 'gray'  # Fallback color
                entity_type = "Unknown"
            size = 4000
            label = f"{entity_type}\n{node_id.split('_')[1]}\n[{country}]"
        elif node_id in [source_account, return_account]:
            color = 'orange'
            size = 2500
            label_prefix = "Source Acct" if node_id == source_account else "Return Acct"
            label = f"{label_prefix}\n{node_id.split('_')[1]}\n[{country}]"
        else:  # Intermediary
            color = 'skyblue'
            size = 2000
            label = f"Intermediary\n{node_id.split('_')[1]}\n[{country}]"

        ax.scatter(pos[node_id][0], pos[node_id][1], c=color,
                   s=size, alpha=0.9, edgecolors='black', zorder=3)
        ax.text(pos[node_id][0], pos[node_id][1], label, ha='center',
                va='center', fontsize=9, fontweight='bold', zorder=4)

    # Draw edges
    for src, dest, data in G.edges(data=True):
        src_pos = pos[src]
        dest_pos = pos[dest]

        if data['type'] == 'ownership':
            ax.annotate('', xy=dest_pos, xytext=src_pos,
                        arrowprops=dict(arrowstyle='-', color='gray',
                                        lw=2, linestyle=':', alpha=0.7),
                        zorder=1)
        elif data['type'] == 'transaction':
            rad_val = 0.1
            if src_pos[1] < dest_pos[1]:
                rad_val = 0.1
            elif src_pos[1] > dest_pos[1]:
                rad_val = -0.1
            else:  # handle straight line case
                if src_pos[0] > dest_pos[0]:
                    rad_val = 0.2
                else:
                    rad_val = -0.2

            ax.annotate('', xy=dest_pos, xytext=src_pos,
                        arrowprops=dict(arrowstyle='->', color='black', lw=2,
                                        connectionstyle=f"arc3,rad={rad_val}"),
                        zorder=2)

            amount = data['amount']
            dt = datetime.datetime.fromisoformat(
                data['timestamp'].replace('Z', ''))
            time_str = dt.strftime('%d-%m %H:%M')
            label = f"€{amount:,.0f}\n{time_str}"

            mid_x = (src_pos[0] + dest_pos[0]) / 2
            mid_y = (src_pos[1] + dest_pos[1]) / 2

            ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8), zorder=3)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 4)
    ax.set_aspect('equal')
    ax.axis('off')

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Originator',
                   markerfacecolor='yellow', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Originator Accounts',
                   markerfacecolor='orange', markersize=15),
        plt.Line2D([0], [0], marker='o', color='w', label='Intermediary Accounts',
                   markerfacecolor='skyblue', markersize=15),
        plt.Line2D([0], [0], color='black', lw=2, label='Transaction'),
        plt.Line2D([0], [0], color='gray', lw=2,
                   linestyle=':', label='Ownership')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    initial_amount = transactions[0]['amount']
    final_amount = transactions[-1]['amount']
    returned_ratio = final_amount / initial_amount if initial_amount > 0 else 0
    total_hops = len(transactions)

    stats_text = (f"Pattern Summary:\n"
                  f"Initial Amount: €{initial_amount:,.2f}\n"
                  f"Final Amount: €{final_amount:,.2f}\n"
                  f"Return Ratio: {returned_ratio:.1%}\n"
                  f"Hops: {total_hops}")

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig('u_turn_transactions_network.png',
                dpi=300, bbox_inches='tight')
    print("✓ Network topology visualization saved as 'u_turn_transactions_network.png'")
    plt.show()

    return pattern


def create_timeline_visualization(pattern_data, pattern):
    """Create timeline visualization of transactions"""
    transactions = sorted(pattern['transactions'],
                          key=lambda tx: tx['timestamp'])

    if not transactions:
        return

    for tx in transactions:
        tx['datetime'] = datetime.datetime.fromisoformat(
            tx['timestamp'].replace('Z', ''))

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle(
        f'UTurnTransactions Pattern Timeline: {pattern["pattern_id"]}', fontsize=16, fontweight='bold')

    times = [tx['datetime'] for tx in transactions]
    amounts = [tx['amount'] for tx in transactions]
    labels = [f"Hop {i+1}" for i in range(len(transactions))]

    ax.plot(times, amounts, '-o', color='navy',
            markersize=10, markerfacecolor='skyblue')

    for i, txt in enumerate(labels):
        ax.annotate(txt, (times[i], amounts[i]), textcoords="offset points", xytext=(
            0, 10), ha='center')

    ax.set_xlabel("Timestamp", fontsize=12)
    ax.set_ylabel("Amount (€)", fontsize=12)
    ax.set_title("Transaction Flow Over Time", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # Show delays
    for i in range(len(times) - 1):
        delay = times[i+1] - times[i]
        mid_time = times[i] + delay / 2
        mid_amount = (amounts[i] + amounts[i+1]) / 2
        if delay.total_seconds() > 0:
            ax.text(mid_time, mid_amount, f"{delay.days}d {delay.seconds//3600}h",
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7))

    plt.tight_layout()
    plt.savefig('u_turn_transactions_timeline.png',
                dpi=300, bbox_inches='tight')
    print("✓ Timeline visualization saved as 'u_turn_transactions_timeline.png'")
    plt.show()


def main():
    """Main function to generate visualizations"""
    print("=== UTurnTransactions Pattern Visualization ===")

    # Generate pattern data
    data = generate_pattern_data()
    if data is None:
        return
    pattern_data, nodes_df, edges_df, config = data

    print(f"Generated {len(pattern_data.get('patterns', []))} patterns")

    # Create visualizations
    print("\nCreating network topology visualization...")
    pattern = create_network_visualization(
        pattern_data, nodes_df, edges_df, config)

    if pattern:
        print("\nCreating timeline visualization...")
        create_timeline_visualization(pattern_data, pattern)

        print("\nVisualizations saved:")
        print("- u_turn_transactions_network.png")
        print("- u_turn_transactions_timeline.png")


if __name__ == "__main__":
    main()
