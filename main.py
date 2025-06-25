import random
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


def load_configurations(config_file: str = "configs/graph.yaml") -> Dict[str, Any]:
    """Load and merge main config and patterns config"""
    with open(config_file, 'r') as f:
        main_config = yaml.safe_load(f)

    # Try to load patterns config if it exists
    patterns_config_file = "configs/patterns.yaml"
    if os.path.exists(patterns_config_file):
        with open(patterns_config_file, 'r') as f:
            patterns_config = yaml.safe_load(f)

        if "pattern_config" not in main_config:
            main_config["pattern_config"] = {}

        # Only add pattern configs that don't already exist in main config
        # This preserves any pattern configs that are explicitly set in the main config
        for pattern_name, pattern_config in patterns_config.items():
            if pattern_name not in main_config["pattern_config"]:
                main_config["pattern_config"][pattern_name] = pattern_config

    return main_config


def convert_enums_to_strings(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Convert all non-serializable values in graph attributes to strings for GraphML compatibility.
    GraphML only supports basic data types: string, int, float, boolean.

    Args:
        graph: The original graph with complex data types

    Returns:
        A new graph with all values converted to GraphML-compatible types
    """
    def convert_value(value):
        """Convert a single value to GraphML-compatible format."""
        # Handle None
        if value is None:
            return ""

        # Handle basic types that GraphML supports
        if isinstance(value, (str, int, float, bool)):
            return value

        # Handle enums
        if isinstance(value, Enum):
            return str(value.value)

        # Handle datetime objects
        if isinstance(value, datetime.datetime):
            return value.isoformat()
        elif isinstance(value, datetime.date):
            return value.isoformat()

        # Handle class types
        if isinstance(value, type):
            return value.__name__

        # Handle dictionaries (convert to JSON-like string)
        if isinstance(value, dict):
            try:
                # Convert dict values recursively
                converted_dict = {k: convert_value(
                    v) for k, v in value.items()}
                return str(converted_dict)
            except:
                return str(value)

        # Handle lists/tuples
        if isinstance(value, (list, tuple)):
            try:
                converted_list = [convert_value(v) for v in value]
                return str(converted_list)
            except:
                return str(value)

        # Handle any other object - convert to string
        try:
            # Try to get a meaningful string representation
            if hasattr(value, '__name__'):
                return value.__name__
            elif hasattr(value, 'name'):
                return str(value.name)
            elif hasattr(value, 'value'):
                return str(value.value)
            else:
                return str(value)
        except:
            # Fallback to basic string conversion
            return str(type(value).__name__)

    # Create a copy of the graph
    converted_graph = graph.copy()

    # Convert node attributes
    for node_id, attrs in converted_graph.nodes(data=True):
        # Use list() to avoid modification during iteration
        for key, value in list(attrs.items()):
            try:
                converted_value = convert_value(value)
                attrs[key] = converted_value
            except Exception as e:
                # If conversion fails, use a safe fallback
                print(
                    f"Warning: Could not convert node attribute {key}={value}, using fallback. Error: {e}")
                attrs[key] = str(type(value).__name__)

    # Convert edge attributes
    for src, dest, attrs in converted_graph.edges(data=True):
        # Use list() to avoid modification during iteration
        for key, value in list(attrs.items()):
            try:
                converted_value = convert_value(value)
                attrs[key] = converted_value
            except Exception as e:
                # If conversion fails, use a safe fallback
                print(
                    f"Warning: Could not convert edge attribute {key}={value}, using fallback. Error: {e}")
                attrs[key] = str(type(value).__name__)

    return converted_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic AML dataset with Tide")
    parser.add_argument("--config", default="configs/graph.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output-config", default="configs/output.yaml",
                        help="Path to output configuration file")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory for generated files")
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
    transactions_filepath = os.path.join(
        args.output_dir, args.transactions_file)
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
        # Default values if file doesn't exist
        print(
            f"Output configuration file not found at {output_config_path}, using defaults.")
        output_config = {
            'CSVfiles': True,
            'GraphML': False,
            'Gpickle': False
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

    # Export to CSV
    if output_config.get('CSVfiles', True):
        print("\nExporting to CSV...")
        export_to_csv(
            graph=graph,
            nodes_filepath=nodes_filepath,
            edges_filepath=edges_filepath,
            transactions_filepath=transactions_filepath
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

    print(
        f"Exported {len(aml_graph_gen.injected_patterns)} tracked patterns to: {patterns_filepath}")
    print(f"Generated files saved to: {args.output_dir}")

    # Export to GraphML for visualization
    if output_config.get('GraphML', False):
        print("\nSaving graph in GraphML format for visualization...")
        # Convert enums to strings for GraphML compatibility
        converted_graph = convert_enums_to_strings(graph)
        nx.write_graphml(converted_graph, graphml_filepath)
        print(f"Graph saved as: {graphml_filepath}")

    # Export to Gpickle
    if output_config.get('Gpickle', False):
        print("\nSaving graph in Gpickle format...")
        with open(gpickle_filepath, 'wb') as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        print(f"Graph saved as: {gpickle_filepath}")

    print("\nTo visualize the patterns, run:")
    print("python visualize_patterns.py --graph_file generated_graph.graphml --config_file configs/graph.yaml")
