"""
Tide: A Customisable Dataset Generator for Benchmarking Money Laundering Detection.

Usage:
    python main.py --config configs/graph.yaml
    python main.py --config configs/graph.yaml --output-dir output/

To convert generated CSVs to PyTorch Geometric format:
    python tools/csv_to_pytorch.py --nodes generated_nodes.csv --edges generated_transactions.csv --output graph.pt
"""

import datetime
import yaml
import networkx as nx
from typing import Dict, Any
from enum import Enum
import json
import argparse
import os
import pickle

from tide.graph_generator import GraphGenerator
from tide.outputs import export_to_csv
from tide.datastructures.enums import EdgeType


def compute_homophily(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Compute fraud-class homophily metrics over transaction edges.

    Returns dict with:
      - edge_homophily: fraction of transaction edges connecting same-class nodes
      - fraud_homophily: among edges with at least one fraud endpoint,
                         fraction where both endpoints are fraud
      - fraud_node_ratio: fraction of nodes labeled fraudulent
      - fraud_edge_ratio: fraction of transaction edges labeled fraudulent
    """
    fraud_set = set()
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get('is_fraudulent', False):
            fraud_set.add(node_id)

    same_class = 0
    total_tx = 0
    fraud_incident = 0
    both_fraud = 0
    fraud_edges = 0

    for u, v, attrs in graph.edges(data=True):
        et = attrs.get('edge_type')
        if not (et == EdgeType.TRANSACTION or str(et) == 'transaction'):
            continue
        total_tx += 1
        if attrs.get('is_fraudulent', False):
            fraud_edges += 1

        u_fraud = u in fraud_set
        v_fraud = v in fraud_set
        if u_fraud == v_fraud:
            same_class += 1
        if u_fraud or v_fraud:
            fraud_incident += 1
            if u_fraud and v_fraud:
                both_fraud += 1

    total_nodes = graph.number_of_nodes()
    return {
        'edge_homophily': same_class / total_tx if total_tx else 0.0,
        'fraud_homophily': both_fraud / fraud_incident if fraud_incident else 0.0,
        'fraud_node_ratio': len(fraud_set) / total_nodes if total_nodes else 0.0,
        'fraud_edge_ratio': fraud_edges / total_tx if total_tx else 0.0,
    }


def compute_edge_label_homophily(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Compute edge-label homophily in the line graph.

    For each edge, measures what fraction of its neighboring edges (edges sharing
    a node) have the same label.

    Returns dict with:
      - homophily_overall: weighted average across all classes
      - homophily_class_0: for legitimate edges, fraction of neighbors also legitimate
      - homophily_class_1: for fraud edges, fraction of neighbors also fraud
      - fraud_rate: fraction of transaction edges that are fraudulent
    """
    from collections import defaultdict

    tx_edges = []
    for u, v, attrs in graph.edges(data=True):
        et = attrs.get('edge_type')
        if not (et == EdgeType.TRANSACTION or str(et) == 'transaction'):
            continue
        label = 1 if attrs.get('is_fraudulent', False) else 0
        tx_edges.append((u, v, label))

    if not tx_edges:
        return {'homophily_overall': 0.0, 'fraud_rate': 0.0}

    node_class_counts = defaultdict(lambda: defaultdict(int))
    node_degree = defaultdict(int)

    for u, v, label in tx_edges:
        node_class_counts[u][label] += 1
        node_class_counts[v][label] += 1
        node_degree[u] += 1
        node_degree[v] += 1

    class_same = defaultdict(float)
    class_total = defaultdict(float)

    for u, v, label in tx_edges:
        same = node_class_counts[u][label] + node_class_counts[v][label] - 2
        total = node_degree[u] + node_degree[v] - 2
        class_same[label] += same
        class_total[label] += total

    result = {}
    total_same_all = 0.0
    total_all = 0.0

    for c in sorted(class_same.keys()):
        if class_total[c] > 0:
            result[f'homophily_class_{c}'] = class_same[c] / class_total[c]
        else:
            result[f'homophily_class_{c}'] = 0.0
        total_same_all += class_same[c]
        total_all += class_total[c]

    result['homophily_overall'] = total_same_all / total_all if total_all > 0 else 0.0

    num_fraud = sum(1 for _, _, l in tx_edges if l == 1)
    result['fraud_rate'] = num_fraud / len(tx_edges)

    return result


def load_configurations(config_file: str = "configs/graph.yaml") -> Dict[str, Any]:
    """Load and merge main config and patterns config."""
    with open(config_file, 'r') as f:
        main_config = yaml.safe_load(f)

    patterns_config_file = "configs/patterns.yaml"
    if os.path.exists(patterns_config_file):
        with open(patterns_config_file, 'r') as f:
            patterns_config = yaml.safe_load(f)

        if "pattern_config" not in main_config:
            main_config["pattern_config"] = {}

        for pattern_name, pattern_config in patterns_config.items():
            if pattern_name not in main_config["pattern_config"]:
                main_config["pattern_config"][pattern_name] = pattern_config

    return main_config


def convert_enums_to_strings(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Convert all non-serializable values in graph attributes to strings.

    Required for GraphML/Gpickle export, which only supports basic data types.
    """
    def convert_value(value):
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return str(value.value)
        if isinstance(value, datetime.datetime):
            return value.isoformat()
        elif isinstance(value, datetime.date):
            return value.isoformat()
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, dict):
            try:
                return str({k: convert_value(v) for k, v in value.items()})
            except Exception:
                return str(value)
        if isinstance(value, (list, tuple)):
            try:
                return str([convert_value(v) for v in value])
            except Exception:
                return str(value)
        try:
            if hasattr(value, '__name__'):
                return value.__name__
            elif hasattr(value, 'name'):
                return str(value.name)
            elif hasattr(value, 'value'):
                return str(value.value)
            else:
                return str(value)
        except Exception:
            return str(type(value).__name__)

    converted_graph = graph.copy()

    for node_id, attrs in converted_graph.nodes(data=True):
        for key, value in list(attrs.items()):
            try:
                attrs[key] = convert_value(value)
            except Exception as e:
                print(f"Warning: Could not convert node attribute {key}={value}: {e}")
                attrs[key] = str(type(value).__name__)

    for src, dest, attrs in converted_graph.edges(data=True):
        for key, value in list(attrs.items()):
            try:
                attrs[key] = convert_value(value)
            except Exception as e:
                print(f"Warning: Could not convert edge attribute {key}={value}: {e}")
                attrs[key] = str(type(value).__name__)

    return converted_graph


def filter_transactions_only(graph: nx.DiGraph) -> nx.DiGraph:
    """Return a copy of the graph containing only TRANSACTION edges."""
    filtered = graph.copy()
    edges_to_remove = []
    for u, v, attrs in filtered.edges(data=True):
        et = attrs.get('edge_type')
        if not (et == EdgeType.TRANSACTION or str(et) == 'transaction'):
            edges_to_remove.append((u, v))
    filtered.remove_edges_from(edges_to_remove)
    return filtered


def filter_by_institution_country(graph: nx.DiGraph, filter_country: str) -> nx.DiGraph:
    """
    Filter graph to only include edges where at least one endpoint account
    belongs to an institution in the specified country.
    """
    from tide.datastructures.enums import NodeType

    target_institutions = set()
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get('node_type')
        if node_type == NodeType.INSTITUTION or str(node_type) == 'institution':
            if attrs.get('country_code') == filter_country:
                target_institutions.add(node_id)

    filtered_accounts = set()
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get('node_type')
        if node_type == NodeType.ACCOUNT or str(node_type) == 'account':
            if attrs.get('institution_id') in target_institutions:
                filtered_accounts.add(node_id)

    print(f"Institution filter: country={filter_country}, "
          f"found {len(target_institutions)} institutions, {len(filtered_accounts)} accounts")

    filtered = graph.copy()
    edges_to_remove = []
    for u, v, attrs in filtered.edges(data=True):
        if u not in filtered_accounts and v not in filtered_accounts:
            edges_to_remove.append((u, v))

    filtered.remove_edges_from(edges_to_remove)

    total_edges = graph.number_of_edges()
    remaining_edges = filtered.number_of_edges()

    isolated_nodes = list(nx.isolates(filtered))
    total_nodes = filtered.number_of_nodes()
    filtered.remove_nodes_from(isolated_nodes)
    remaining_nodes = filtered.number_of_nodes()

    print(f"Institution filter applied: {remaining_edges}/{total_edges} edges retained, "
          f"{remaining_nodes}/{total_nodes} nodes retained ({len(isolated_nodes)} isolated nodes removed)")

    return filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic AML dataset with Tide",
        epilog="To convert CSVs to PyTorch format: python tools/csv_to_pytorch.py --nodes <nodes.csv> --edges <transactions.csv> --output <graph.pt>",
    )
    parser.add_argument("--config", default="configs/graph.yaml",
                        help="Path to graph configuration file (default: configs/graph.yaml)")
    parser.add_argument("--output-config", default="configs/output.yaml",
                        help="Path to output configuration file (default: configs/output.yaml)")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for generated files (default: current directory)")
    parser.add_argument("--nodes-file", default="generated_nodes.csv",
                        help="Filename for nodes CSV")
    parser.add_argument("--edges-file", default="generated_edges.csv",
                        help="Filename for edges CSV")
    parser.add_argument("--transactions-file", default="generated_transactions.csv",
                        help="Filename for transactions-only CSV")
    parser.add_argument("--patterns-file", default="generated_patterns.json",
                        help="Filename for patterns JSON")
    parser.add_argument("--graphml-file", default="generated_graph.graphml",
                        help="Filename for GraphML export")
    parser.add_argument("--gpickle-file", default="generated_graph.gpickle",
                        help="Filename for Gpickle export")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct full file paths
    nodes_filepath = os.path.join(args.output_dir, args.nodes_file)
    edges_filepath = os.path.join(args.output_dir, args.edges_file)
    transactions_filepath = os.path.join(args.output_dir, args.transactions_file)
    patterns_filepath = os.path.join(args.output_dir, args.patterns_file)
    graphml_filepath = os.path.join(args.output_dir, args.graphml_file)
    gpickle_filepath = os.path.join(args.output_dir, args.gpickle_file)

    # Load and merge configurations
    generator_parameters = load_configurations(args.config)
    print(f'Loading configuration from: {args.config}')

    # Load output configurations
    output_config_path = args.output_config
    if os.path.exists(output_config_path):
        with open(output_config_path, 'r') as f:
            output_config = yaml.safe_load(f)
            print(f'Loading output configuration from: {output_config_path}')
    else:
        print(f"Output configuration file not found at {output_config_path}, using defaults.")
        output_config = {
            'CSVfiles': True,
            'GraphML': False,
            'Gpickle': False,
        }

    # Convert date strings to datetime objects
    generator_parameters["time_span"]["start_date"] = datetime.datetime.fromisoformat(
        generator_parameters["time_span"]["start_date"])
    generator_parameters["time_span"]["end_date"] = datetime.datetime.fromisoformat(
        generator_parameters["time_span"]["end_date"])

    aml_graph_gen = GraphGenerator(params=generator_parameters)
    graph = aml_graph_gen.generate_graph()

    print(f"\n--- Graph Summary ---")
    print(f"Number of nodes: {aml_graph_gen.num_of_nodes()}")
    print(f"Number of edges: {aml_graph_gen.num_of_edges()}")

    homophily = compute_homophily(graph)
    edge_label_h = compute_edge_label_homophily(graph)

    print(f"\n--- Node-label homophily ---")
    print(f"  Edge homophily (all classes):  {homophily['edge_homophily']:.6f}")
    print(f"  Fraud-class homophily:         {homophily['fraud_homophily']:.6f}")
    print(f"  Fraud node ratio:              {homophily['fraud_node_ratio']:.6f}")
    print(f"  Fraud edge ratio:              {homophily['fraud_edge_ratio']:.6f}")

    print(f"\n--- Edge-label homophily ---")
    print(f"  Homophily (overall):           {edge_label_h['homophily_overall']:.6f}")
    if 'homophily_class_0' in edge_label_h:
        print(f"  Homophily (class 0 - legit):   {edge_label_h['homophily_class_0']:.6f}")
    if 'homophily_class_1' in edge_label_h:
        print(f"  Homophily (class 1 - fraud):   {edge_label_h['homophily_class_1']:.6f}")
    print(f"  Fraud rate:                    {edge_label_h['fraud_rate']:.6f}")

    # Get filter settings
    institution_filter_country = generator_parameters.get('institution_filter_country')
    if institution_filter_country:
        print(f"\nInstitution filter enabled: filtering to country '{institution_filter_country}'")

    remove_isolated_nodes = output_config.get('RemoveIsolatedNodes', True)

    # Export to CSV
    if output_config.get('CSVfiles', True):
        print("\nExporting to CSV...")
        export_to_csv(
            graph=graph,
            nodes_filepath=nodes_filepath,
            edges_filepath=edges_filepath,
            transactions_filepath=transactions_filepath,
            institution_filter_country=institution_filter_country,
            remove_isolated_nodes=remove_isolated_nodes
        )

    # Export tracked patterns as JSON
    patterns_data = {
        'metadata': {
            'generation_timestamp': datetime.datetime.now().isoformat(),
            'total_patterns': len(aml_graph_gen.injected_patterns),
            'graph_nodes': aml_graph_gen.num_of_nodes(),
            'graph_edges': aml_graph_gen.num_of_edges(),
            'config_file': args.config,
            'output_directory': args.output_dir
        },
        'patterns': aml_graph_gen.injected_patterns
    }

    with open(patterns_filepath, 'w') as f:
        json.dump(patterns_data, f, indent=2, default=str)

    print(f"Exported {len(aml_graph_gen.injected_patterns)} tracked patterns to: {patterns_filepath}")
    print(f"Generated files saved to: {args.output_dir}")

    # Prepare graph for additional exports (GraphML, Gpickle)
    transactions_only = output_config.get('TransactionsOnly', False)
    graph_for_exports = filter_transactions_only(graph) if transactions_only else graph

    if institution_filter_country:
        graph_for_exports = filter_by_institution_country(
            graph_for_exports, institution_filter_country)
    elif remove_isolated_nodes:
        isolated = list(nx.isolates(graph_for_exports))
        if isolated:
            total_before = graph_for_exports.number_of_nodes()
            graph_for_exports = graph_for_exports.copy()
            graph_for_exports.remove_nodes_from(isolated)
            print(f"Removed {len(isolated)} isolated nodes "
                  f"({graph_for_exports.number_of_nodes()}/{total_before} nodes retained)")

    # Export to GraphML
    if output_config.get('GraphML', False):
        print("\nSaving graph in GraphML format...")
        converted_graph = convert_enums_to_strings(graph_for_exports)
        nx.write_graphml(converted_graph, graphml_filepath)
        print(f"Graph saved as: {graphml_filepath}")

    # Export to Gpickle
    if output_config.get('Gpickle', False):
        print("\nSaving graph in Gpickle format...")
        converted_graph = convert_enums_to_strings(graph_for_exports)
        with open(gpickle_filepath, 'wb') as f:
            pickle.dump(converted_graph, f, pickle.HIGHEST_PROTOCOL)
        print(f"Graph saved as: {gpickle_filepath}")

    print("\nTo convert to PyTorch Geometric format, run:")
    print(f"  python tools/csv_to_pytorch.py --nodes {nodes_filepath} --edges {transactions_filepath} --output graph.pt")
