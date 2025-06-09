#!/usr/bin/env python3
"""
Simple AML Pattern Visualizer - Ground Truth Approach

This script visualizes the patterns that were actually injected during graph generation
by reading the tracked pattern data, rather than trying to detect them afterwards.

This approach is:
- More accurate (shows exactly what was generated)
- Simpler (no complex detection algorithms)
- Faster (no need to analyze the entire graph)
- Ground truth validation (perfect verification)
"""

import argparse
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from typing import Dict, List, Any
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


class SimplePatternVisualizer:
    """Visualizes tracked patterns directly from injection data"""

    def __init__(self, patterns_file: str, nodes_file: str, edges_file: str):
        """Initialize with pattern tracking data and graph data"""
        # Load tracked patterns
        with open(patterns_file, 'r') as f:
            self.patterns_data = json.load(f)

        self.patterns = self.patterns_data.get('patterns', [])
        self.metadata = self.patterns_data.get('metadata', {})

        # Load graph data for visualization
        self.nodes_df = pd.read_csv(nodes_file)
        self.edges_df = pd.read_csv(edges_file)
        self.graph = self._build_graph()

        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        sns.set_palette("husl")

    def _build_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from CSV data"""
        G = nx.DiGraph()

        # Add nodes
        for _, row in self.nodes_df.iterrows():
            node_attrs = row.to_dict()
            G.add_node(row['node_id'], **node_attrs)

        # Add edges
        for _, row in self.edges_df.iterrows():
            edge_attrs = row.to_dict()
            G.add_edge(row['src'], row['dest'], **edge_attrs)

        return G

    def create_summary_dashboard(self, save_path: str = "ground_truth_patterns.png"):
        """Create dashboard showing exactly what patterns were injected"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("AML Patterns - Ground Truth (What Was Actually Generated)",
                     fontsize=16, fontweight='bold')

        # 1. Pattern counts by type
        pattern_counts = Counter(p['pattern_type'] for p in self.patterns)

        if pattern_counts:
            pattern_names = list(pattern_counts.keys())
            counts = list(pattern_counts.values())

            bars = axes[0, 0].bar(pattern_names, counts, color=[
                                  '#FF6B6B', '#4ECDC4', '#45B7D1'])
            axes[0, 0].set_title("Injected Pattern Counts")
            axes[0, 0].set_ylabel("Number of Patterns")

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                str(count), ha='center', va='bottom', fontweight='bold')

            plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha='right')

        # 2. Transaction types breakdown
        transaction_types = []
        for pattern in self.patterns:
            for tx in pattern.get('transactions', []):
                tx_type = tx.get('transaction_type', '').replace(
                    'TransactionType.', '').lower()
                transaction_types.append(tx_type)

        if transaction_types:
            tx_type_counts = Counter(transaction_types)

            # Define colors for transaction types
            tx_colors = {
                'deposit': '#FF6B6B',
                'transfer': '#4ECDC4',
                'withdrawal': '#45B7D1',
                'payment': '#96CEB4',
                'salary': '#FFEAA7'
            }

            types = list(tx_type_counts.keys())
            counts = list(tx_type_counts.values())
            colors = [tx_colors.get(t, '#DDA0DD') for t in types]

            bars = axes[0, 1].bar(types, counts, color=colors)
            axes[0, 1].set_title("Transaction Types in Patterns")
            axes[0, 1].set_ylabel("Number of Transactions")

            # Add count labels on bars
            for bar, count in zip(bars, counts):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                                str(count), ha='center', va='bottom', fontweight='bold')

            plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right')

        # 3. Geographic distribution
        all_countries = []
        for pattern in self.patterns:
            countries = pattern.get('countries', [])
            all_countries.extend(countries)

        if all_countries:
            country_counts = Counter(all_countries)
            top_countries = dict(country_counts.most_common(8))

            axes[0, 2].bar(top_countries.keys(),
                           top_countries.values(), color='#96CEB4')
            axes[0, 2].set_title("Countries Involved in Patterns")
            axes[0, 2].set_ylabel("Pattern Involvement")
            plt.setp(axes[0, 2].get_xticklabels(), rotation=45, ha='right')

        # 4. Temporal analysis
        timestamps = []
        for pattern in self.patterns:
            if pattern.get('start_time'):
                try:
                    if isinstance(pattern['start_time'], str):
                        timestamp = datetime.fromisoformat(
                            pattern['start_time'].replace('Z', '+00:00'))
                    else:
                        timestamp = pattern['start_time']
                    timestamps.append(timestamp)
                except:
                    continue

        if timestamps:
            dates = [ts.date() for ts in timestamps]
            date_counts = Counter(dates)
            sorted_dates = sorted(date_counts.keys())
            counts = [date_counts[d] for d in sorted_dates]

            axes[1, 0].plot(sorted_dates, counts, marker='o',
                            linewidth=2, markersize=6, color='#45B7D1')
            axes[1, 0].set_title("Pattern Generation Timeline")
            axes[1, 0].set_xlabel("Date")
            axes[1, 0].set_ylabel("Patterns Created")
            plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')

        # 5. Transaction flow analysis by pattern type
        flow_analysis = {}
        for pattern in self.patterns:
            pattern_type = pattern['pattern_type']
            if pattern_type not in flow_analysis:
                flow_analysis[pattern_type] = Counter()

            for tx in pattern.get('transactions', []):
                tx_type = tx.get('transaction_type', '').replace(
                    'TransactionType.', '').lower()
                flow_analysis[pattern_type][tx_type] += 1

        if flow_analysis:
            # Create a stacked bar chart showing transaction types per pattern
            pattern_types = list(flow_analysis.keys())
            tx_types = set()
            for counts in flow_analysis.values():
                tx_types.update(counts.keys())
            tx_types = sorted(list(tx_types))

            bottom = np.zeros(len(pattern_types))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

            for i, tx_type in enumerate(tx_types):
                values = [flow_analysis[pt].get(
                    tx_type, 0) for pt in pattern_types]
                axes[1, 1].bar(pattern_types, values, bottom=bottom,
                               label=tx_type.title(), color=colors[i % len(colors)])
                bottom += values

            axes[1, 1].set_title("Transaction Types by Pattern")
            axes[1, 1].set_ylabel("Number of Transactions")
            axes[1, 1].legend(loc='upper right', fontsize=8)
            plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

        # 6. Summary statistics
        total_patterns = len(self.patterns)
        total_transactions = sum(p.get('num_transactions', 0)
                                 for p in self.patterns)
        total_amount = sum(p.get('total_amount', 0) for p in self.patterns)
        avg_amount = total_amount / max(total_transactions, 1)

        # Add transaction type summary
        tx_type_summary = ""
        if transaction_types:
            tx_type_counts = Counter(transaction_types)
            for tx_type, count in tx_type_counts.most_common():
                percentage = (count / len(transaction_types)) * 100
                tx_type_summary += f"• {tx_type.title()}: {count} ({percentage:.1f}%)\n        "

        stats_text = f"""
        🎯 GROUND TRUTH SUMMARY

        Total Patterns Generated: {total_patterns}
        Total Fraudulent Transactions: {total_transactions}
        Total Amount: ${total_amount:,.2f}
        Average Transaction: ${avg_amount:,.2f}

        Transaction Types:
        {tx_type_summary}

        Graph Statistics:
        • Nodes: {self.metadata.get('graph_nodes', 'N/A'):,}
        • Edges: {self.metadata.get('graph_edges', 'N/A'):,}

        Pattern Breakdown:
        """

        for pattern_type, count in pattern_counts.items():
            stats_text += f"• {pattern_type}: {count}\n        "

        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title("Generation Summary")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Ground truth dashboard saved to: {save_path}")
        plt.close()
        return save_path

    def visualize_pattern_network(self, pattern_data: Dict[str, Any], save_path: str = None):
        """Visualize a specific pattern as a network subgraph"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))

        # Get entities involved in this pattern
        entities = set(pattern_data.get('entities', []))
        transactions = pattern_data.get('transactions', [])

        if not entities:
            print(
                f"No entities found for pattern {pattern_data.get('pattern_id', 'unknown')}")
            plt.close()
            return None

        # Build subgraph
        subgraph = self.graph.subgraph(entities)

        # Set up layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50)

        # Color nodes by type
        node_colors = []
        node_sizes = []
        for node in subgraph.nodes():
            node_type = str(self.graph.nodes[node].get(
                'node_type', 'unknown')).lower()
            if 'account' in node_type:
                node_colors.append('#4ECDC4')
                node_sizes.append(400)
            elif 'individual' in node_type:
                node_colors.append('#FF6B6B')
                node_sizes.append(500)
            elif 'business' in node_type:
                node_colors.append('#45B7D1')
                node_sizes.append(600)
            else:
                node_colors.append('#96CEB4')
                node_sizes.append(300)

        # Draw network
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8, ax=ax)

        # Group transactions by type for different visualization
        tx_by_type = {}
        for tx in transactions:
            tx_type = tx.get('transaction_type', '').replace(
                'TransactionType.', '').lower()
            if tx_type not in tx_by_type:
                tx_by_type[tx_type] = []
            tx_by_type[tx_type].append((tx['src'], tx['dest']))

        # Define colors and styles for different transaction types
        tx_styles = {
            'deposit': {'color': '#FF6B6B', 'width': 3, 'style': 'solid', 'alpha': 0.9},
            'transfer': {'color': '#4ECDC4', 'width': 2.5, 'style': 'solid', 'alpha': 0.8},
            'withdrawal': {'color': '#45B7D1', 'width': 2, 'style': 'dashed', 'alpha': 0.8},
            'payment': {'color': '#96CEB4', 'width': 2, 'style': 'dotted', 'alpha': 0.7},
            'salary': {'color': '#FFEAA7', 'width': 2, 'style': 'dashdot', 'alpha': 0.7}
        }

        # Draw other edges (non-pattern transactions) in gray
        pattern_edges = set((tx['src'], tx['dest']) for tx in transactions)
        normal_edges = []
        for src, dest in subgraph.edges():
            if (src, dest) not in pattern_edges and (dest, src) not in pattern_edges:
                normal_edges.append((src, dest))

        if normal_edges:
            nx.draw_networkx_edges(subgraph, pos, edgelist=normal_edges,
                                   edge_color='lightgray', width=1, alpha=0.3,
                                   style='solid', ax=ax)

        # Draw pattern transactions by type
        legend_elements = []
        for tx_type, edges in tx_by_type.items():
            if edges:
                style = tx_styles.get(
                    tx_type, {'color': '#DDA0DD', 'width': 2, 'style': 'solid', 'alpha': 0.8})

                # Convert matplotlib linestyle names
                matplotlib_style = {
                    'solid': '-',
                    'dashed': '--',
                    'dotted': ':',
                    'dashdot': '-.'
                }.get(style['style'], '-')

                nx.draw_networkx_edges(subgraph, pos, edgelist=edges,
                                       edge_color=style['color'],
                                       width=style['width'],
                                       alpha=style['alpha'],
                                       style=matplotlib_style,
                                       ax=ax)

                # Add to legend
                legend_elements.append(
                    plt.Line2D([0], [0], color=style['color'], linewidth=style['width'],
                               linestyle=matplotlib_style, alpha=style['alpha'],
                               label=f'{tx_type.title()} ({len(edges)})')
                )

        # Add labels with enhanced information
        labels = {}
        for node in subgraph.nodes():
            node_type = str(self.graph.nodes[node].get(
                'node_type', 'unknown')).lower()
            country = self.graph.nodes[node].get('country_code', '')

            # Check if this node is involved in transactions
            tx_count = sum(
                1 for tx in transactions if tx['src'] == node or tx['dest'] == node)

            labels[node] = f"{node[:8]}\n({node_type[:3]})\n{country}\n[{tx_count}tx]"

        nx.draw_networkx_labels(subgraph, pos, labels, font_size=7, ax=ax)

        # Title with pattern details and transaction breakdown
        pattern_type = pattern_data.get('pattern_type', 'Unknown')
        pattern_id = pattern_data.get('pattern_id', 'Unknown')
        num_transactions = pattern_data.get('num_transactions', 0)
        total_amount = pattern_data.get('total_amount', 0)

        # Create transaction type summary for title
        tx_type_summary = []
        for tx_type, edges in tx_by_type.items():
            if edges:
                tx_type_summary.append(f"{tx_type.title()}: {len(edges)}")

        title = f"GROUND TRUTH: {pattern_type}\n"
        title += f"ID: {pattern_id} | Total: {num_transactions} tx | Amount: ${total_amount:,.2f}\n"
        title += f"Transaction Types: {' | '.join(tx_type_summary)}"

        ax.set_title(title, fontsize=11, fontweight='bold', pad=20)

        # Create comprehensive legend
        node_legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4ECDC4',
                       markersize=12, label='Account'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
                       markersize=12, label='Individual'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#45B7D1',
                       markersize=12, label='Business'),
            plt.Line2D([0], [0], color='lightgray', linewidth=1,
                       label='Other Connection')
        ]

        # Combine node and transaction type legends
        all_legend_elements = node_legend_elements + legend_elements

        ax.legend(handles=all_legend_elements, loc='upper left', bbox_to_anchor=(1, 1),
                  fontsize=9, title="Legend", title_fontsize=10)

        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🔗 Pattern network saved to: {save_path}")

        plt.close()
        return save_path

    def create_detailed_report(self, output_file: str = "ground_truth_report.html"):
        """Generate detailed HTML report of all tracked patterns"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AML Patterns - Ground Truth Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
                .pattern-section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .pattern-overseas {{ border-left: 5px solid #FF6B6B; background: #fdf8f8; }}
                .pattern-rapid {{ border-left: 5px solid #4ECDC4; background: #f8fdfd; }}
                .pattern-business {{ border-left: 5px solid #45B7D1; background: #f8fbfd; }}
                .stats-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .stats-table th {{ background: #f8f9fa; }}
                .ground-truth {{ color: #27ae60; font-weight: bold; }}
            </style>
        </head>
        <body>
        """

        # Header
        total_patterns = len(self.patterns)
        total_transactions = sum(p.get('num_transactions', 0)
                                 for p in self.patterns)

        html_content += f"""
        <div class="header">
            <h1>🎯 AML Patterns - Ground Truth Report</h1>
            <p class="ground-truth">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p class="ground-truth">Total Patterns Generated: {total_patterns}</p>
            <p class="ground-truth">Total Fraudulent Transactions: {total_transactions}</p>
            <p><em>This report shows exactly what patterns were injected during graph generation - no detection algorithms involved!</em></p>
        </div>
        """

        # Group patterns by type
        patterns_by_type = {}
        for pattern in self.patterns:
            pattern_type = pattern['pattern_type']
            if pattern_type not in patterns_by_type:
                patterns_by_type[pattern_type] = []
            patterns_by_type[pattern_type].append(pattern)

        # Pattern sections
        for pattern_type, type_patterns in patterns_by_type.items():
            css_class = f"pattern-{pattern_type.lower().replace('overseas', 'overseas').replace('fund', 'rapid').replace('business', 'business')}"

            html_content += f"""
            <div class="pattern-section {css_class}">
                <h2>📊 {pattern_type} ({len(type_patterns)} patterns)</h2>
            """

            for i, pattern in enumerate(type_patterns, 1):
                html_content += f"""
                <div class="pattern-section">
                    <h3>Pattern {i}: {pattern.get('pattern_id', 'Unknown')}</h3>
                    <table class="stats-table">
                        <tr><th>Number of Transactions</th><td>{pattern.get('num_transactions', 0)}</td></tr>
                        <tr><th>Total Amount</th><td>${pattern.get('total_amount', 0):,.2f}</td></tr>
                        <tr><th>Average Amount</th><td>${pattern.get('avg_amount', 0):,.2f}</td></tr>
                        <tr><th>Countries Involved</th><td>{', '.join(pattern.get('countries', []))}</td></tr>
                        <tr><th>Entities Involved</th><td>{len(pattern.get('entities', []))}</td></tr>
                """

                # Add transaction type breakdown
                tx_breakdown = {}
                for tx in pattern.get('transactions', []):
                    tx_type = tx.get('transaction_type', '').replace(
                        'TransactionType.', '').lower()
                    tx_breakdown[tx_type] = tx_breakdown.get(tx_type, 0) + 1

                if tx_breakdown:
                    tx_breakdown_str = ', '.join(
                        [f"{tx_type.title()}: {count}" for tx_type, count in tx_breakdown.items()])
                    html_content += f"""
                        <tr><th>Transaction Types</th><td>{tx_breakdown_str}</td></tr>
                    """

                # Add pattern-specific details
                if pattern_type == 'RepeatedOverseasTransfers':
                    html_content += f"""
                        <tr><th>Source Account</th><td>{pattern.get('source_account', 'N/A')}</td></tr>
                        <tr><th>Destination Countries</th><td>{', '.join(pattern.get('destination_countries', []))}</td></tr>
                        <tr><th>Overseas Destinations</th><td>{pattern.get('num_overseas_destinations', 0)}</td></tr>
                    """
                elif pattern_type == 'RapidFundMovement':
                    html_content += f"""
                        <tr><th>Central Account</th><td>{pattern.get('central_account', 'N/A')}</td></tr>
                        <tr><th>Movement Speed</th><td>{pattern.get('movement_speed_hours', 0):.1f} hours</td></tr>
                    """
                elif pattern_type == 'FrontBusinessActivity':
                    html_content += f"""
                        <tr><th>Business Entity</th><td>{pattern.get('business_entity', 'N/A')}</td></tr>
                        <tr><th>Deposits</th><td>{pattern.get('num_deposits', 0)} (${pattern.get('deposit_amount', 0):,.2f})</td></tr>
                        <tr><th>Transfers</th><td>{pattern.get('num_transfers', 0)} (${pattern.get('transfer_amount', 0):,.2f})</td></tr>
                    """

                html_content += """
                    </table>

                    <h4>Transaction Details:</h4>
                    <table class="stats-table">
                        <tr><th>Source</th><th>Destination</th><th>Type</th><th>Amount</th><th>Timestamp</th></tr>
                """

                # Add individual transaction details
                # Limit to first 10 transactions
                for tx in pattern.get('transactions', [])[:10]:
                    tx_type = tx.get('transaction_type', '').replace(
                        'TransactionType.', '')
                    html_content += f"""
                        <tr>
                            <td>{tx.get('src', 'N/A')[:12]}...</td>
                            <td>{tx.get('dest', 'N/A')[:12]}...</td>
                            <td>{tx_type}</td>
                            <td>${tx.get('amount', 0):,.2f}</td>
                            <td>{tx.get('timestamp', 'N/A')[:16]}</td>
                        </tr>
                    """

                if len(pattern.get('transactions', [])) > 10:
                    html_content += f"""
                        <tr><td colspan="5"><em>... and {len(pattern.get('transactions', [])) - 10} more transactions</em></td></tr>
                    """

                html_content += """
                    </table>
                </div>
                """

            html_content += "</div>"

        html_content += """
        </body>
        </html>
        """

        with open(output_file, 'w') as f:
            f.write(html_content)

        print(f"📄 Ground truth report saved to: {output_file}")

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary statistics of tracked patterns"""
        pattern_counts = Counter(p['pattern_type'] for p in self.patterns)
        total_transactions = sum(p.get('num_transactions', 0)
                                 for p in self.patterns)
        total_amount = sum(p.get('total_amount', 0) for p in self.patterns)

        return {
            'total_patterns': len(self.patterns),
            'pattern_counts': dict(pattern_counts),
            'total_transactions': total_transactions,
            'total_amount': total_amount,
            'avg_transaction_amount': total_amount / max(total_transactions, 1),
            'metadata': self.metadata
        }


def main():
    parser = argparse.ArgumentParser(
        description="Simple AML pattern visualizer using ground truth data")
    parser.add_argument("--patterns", required=True,
                        help="Path to generated patterns JSON file")
    parser.add_argument("--nodes", required=True,
                        help="Path to nodes CSV file")
    parser.add_argument("--edges", required=True,
                        help="Path to edges CSV file")
    parser.add_argument("--output-dir", default="./ground_truth_results",
                        help="Directory to save output files")

    args = parser.parse_args()

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    print("🎯 Starting Ground Truth AML Pattern Visualization")
    print(f"📂 Output directory: {args.output_dir}")

    # Initialize visualizer
    visualizer = SimplePatternVisualizer(args.patterns, args.nodes, args.edges)

    # Get summary
    summary = visualizer.get_pattern_summary()
    print(f"📊 Found {summary['total_patterns']} tracked patterns")
    for pattern_type, count in summary['pattern_counts'].items():
        print(f"   • {pattern_type}: {count}")

    # Create summary dashboard
    summary_path = os.path.join(args.output_dir, "ground_truth_patterns.png")
    visualizer.create_summary_dashboard(summary_path)

    # Create detailed report
    report_path = os.path.join(args.output_dir, "ground_truth_report.html")
    visualizer.create_detailed_report(report_path)

    # Visualize individual patterns
    for i, pattern in enumerate(visualizer.patterns):
        pattern_type = pattern['pattern_type']
        pattern_id = pattern.get('pattern_id', f'pattern_{i+1}')
        network_path = os.path.join(
            args.output_dir, f"ground_truth_{pattern_type}_{i+1}.png")
        visualizer.visualize_pattern_network(pattern, network_path)

    print(f"\n✅ Ground truth visualization completed!")
    print(f"📁 All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
