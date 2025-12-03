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
import torch
import numpy as np

from tide.graph_generator import GraphGenerator
from tide.outputs import export_to_csv
from tide.datastructures.enums import EdgeType, TransactionType


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


def filter_transactions_only(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Return a shallow copy of the graph containing only TRANSACTION edges.
    Nodes are preserved; non-transaction edges are removed.
    """
    filtered = graph.copy()
    edges_to_remove = []
    for u, v, attrs in filtered.edges(data=True):
        et = attrs.get('edge_type')
        if not (et == EdgeType.TRANSACTION or str(et) == 'transaction'):
            edges_to_remove.append((u, v))
    filtered.remove_edges_from(edges_to_remove)
    return filtered


def convert_graph_to_pytorch(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Convert NetworkX graph to PyTorch format suitable for Graph Neural Networks.

    This function properly handles heterogeneous graphs with different edge types
    (transactions and ownership) and converts all attributes to tensors.

    Args:
        graph: The NetworkX directed graph with node and edge attributes

    Returns:
        A dictionary containing:
        - node_features: Tensor of node features
        - node_ids: List mapping node indices to original node IDs
        - edge_index: Tensor of edge indices [2, num_edges]
        - edge_attr: Dict with separate tensors for each edge type
        - metadata: Information about feature names and types
    """
    # Create node ID mapping
    node_list = list(graph.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

    # ------------------------------
    # Build node feature matrix
    # Best-practice: include numeric/boolean + one-hot encoded categoricals
    # ------------------------------

    # Helper to normalize enum/string values
    def normalize_value(val):
        if val is None:
            return None
        try:
            # Handle enums
            if hasattr(val, 'value'):
                return str(val.value)
            return str(val)
        except Exception:
            return str(val)

    # Collect category frequencies
    categorical_fields = [
        'node_type', 'country_code', 'account_category', 'gender',
        'age_group', 'currency', 'business_category'
    ]
    category_counts: Dict[str, Dict[str, int]] = {
        f: {} for f in categorical_fields}

    # Also collect creation years
    creation_years: Dict[str, int] = {}

    for node_id in node_list:
        attrs = graph.nodes[node_id]
        # Count categories
        for field in categorical_fields:
            raw = attrs.get(field)
            norm = normalize_value(raw)
            if norm is not None and norm != "":
                category_counts[field][norm] = category_counts[field].get(
                    norm, 0) + 1

        # Parse creation year
        cdate = attrs.get('creation_date')
        year_val = None
        if isinstance(cdate, datetime.datetime) or isinstance(cdate, datetime.date):
            year_val = cdate.year
        elif isinstance(cdate, str):
            try:
                dt = datetime.datetime.fromisoformat(cdate)
                year_val = dt.year
            except Exception:
                year_val = None
        if year_val is not None:
            creation_years[node_id] = int(year_val)

    # Build encoding maps (with top-K caps for high-cardinality fields)
    topk_caps = {
        'country_code': 30,
        'currency': 5,
        'business_category': 20
    }

    encoding_maps: Dict[str, Dict[str, int]] = {}
    for field, counts in category_counts.items():
        if not counts:
            encoding_maps[field] = {}
            continue
        # Sort by frequency desc
        sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        if field in topk_caps:
            k = topk_caps[field]
            keep = [name for name, _ in sorted_items[:k]]
            # Map kept categories to indices, reserve last for 'other'
            mapping = {name: idx for idx, name in enumerate(keep)}
            mapping['__other__'] = len(keep)
            encoding_maps[field] = mapping
        else:
            # Small cardinality: use all categories, no 'other'
            mapping = {name: idx for idx, (name, _) in enumerate(sorted_items)}
            encoding_maps[field] = mapping

    # Numeric/boolean fields to include
    # NOTE: risk_score, is_high_risk_category, is_high_risk_country are EXCLUDED
    # because fraud patterns SELECT entities based on these features, creating
    # indirect data leakage (high risk → selected for fraud → is_fraudulent=True)
    numeric_fields = [
        'incorporation_year'
    ]

    # Build feature names list order
    node_feature_names: list = []
    # 1) numeric/boolean
    node_feature_names.extend(numeric_fields)
    # 2) derived numeric from date
    node_feature_names.append('creation_year')
    # 3) categoricals one-hot columns
    categorical_expanded_names: Dict[str, list] = {}
    for field in categorical_fields:
        mapping = encoding_maps[field]
        if not mapping:
            categorical_expanded_names[field] = []
            continue
        names = []
        # If '__other__' exists, ensure it is included last
        for name, idx in sorted(mapping.items(), key=lambda x: x[1]):
            if name == '__other__':
                continue
            names.append(f"{field}={name}")
        if '__other__' in mapping:
            names.append(f"{field}=__other__")
        categorical_expanded_names[field] = names
        node_feature_names.extend(names)

    # Assemble feature matrix
    node_features_list: list = []
    for node_id in node_list:
        attrs = graph.nodes[node_id]
        row: list = []
        # Numeric/boolean
        for nf in numeric_fields:
            val = attrs.get(nf)
            if val is None:
                row.append(0.0)
            elif isinstance(val, bool):
                row.append(float(val))
            else:
                try:
                    row.append(float(val))
                except Exception:
                    row.append(0.0)
        # creation_year
        cy = creation_years.get(node_id)
        row.append(float(cy) if cy is not None else 0.0)
        # Categorical one-hot
        for field in categorical_fields:
            mapping = encoding_maps[field]
            num_cols = len(categorical_expanded_names[field])
            if num_cols == 0:
                continue
            vec = [0.0] * num_cols
            raw = attrs.get(field)
            norm = normalize_value(raw)
            if mapping:
                if field in topk_caps:
                    # Use '__other__' bucket when not in mapping
                    key = norm if norm in mapping and norm != '__other__' else '__other__'
                    idx = mapping.get(key)
                else:
                    idx = mapping.get(norm)
                if idx is not None:
                    # Map idx to position in names list (sorted except '__other__' last)
                    # Build name for that idx
                    # Reverse lookup name for stable ordering
                    name_for_idx = None
                    for name, mapped_idx in mapping.items():
                        if mapped_idx == idx:
                            name_for_idx = name
                            break
                    if name_for_idx is not None:
                        try:
                            col_index = categorical_expanded_names[field].index(
                                f"{field}={name_for_idx}"
                            )
                        except ValueError:
                            col_index = None
                        if col_index is not None and 0 <= col_index < num_cols:
                            vec[col_index] = 1.0
            row.extend(vec)
        node_features_list.append(row)

    node_features_tensor = torch.tensor(
        node_features_list, dtype=torch.float32)

    # Separate edges by type
    transaction_edges = []
    ownership_edges = []
    transaction_attrs_list = []
    ownership_attrs_list = []
    transaction_labels = []  # Store edge labels separately

    # First pass: collect unique currencies for one-hot encoding
    unique_currencies = set()
    for src, dest, attrs in graph.edges(data=True):
        if attrs.get('edge_type') in [EdgeType.TRANSACTION, 'transaction']:
            currency = attrs.get('currency')
            if currency:
                unique_currencies.add(currency)
    # Sort for consistent ordering
    unique_currencies = sorted(unique_currencies)

    for src, dest, attrs in graph.edges(data=True):
        edge_type = attrs.get('edge_type')
        src_idx = node_to_idx[src]
        dest_idx = node_to_idx[dest]

        if edge_type == EdgeType.TRANSACTION or edge_type == 'transaction':
            transaction_edges.append([src_idx, dest_idx])

            # Extract transaction attributes (FEATURES only - no labels!)
            trans_attrs = {
                'amount': float(attrs.get('amount', 0.0)),
                # is_fraudulent removed - it's a LABEL, not a feature!
                'timestamp': attrs.get('timestamp').timestamp() if attrs.get('timestamp') else 0.0,
                'time_since_previous': float(attrs.get('time_since_previous_transaction').total_seconds())
                if attrs.get('time_since_previous_transaction') else 0.0
            }

            # Add transaction_type as one-hot encoded features
            transaction_type = attrs.get('transaction_type')
            if transaction_type:
                # Handle both enum and string representations
                if isinstance(transaction_type, str):
                    # String format like "TransactionType.PAYMENT"
                    type_str = transaction_type.split(
                        '.')[-1] if '.' in transaction_type else transaction_type
                else:
                    # Enum format
                    type_str = transaction_type.name

                # One-hot encode transaction type
                for tt in TransactionType:
                    trans_attrs[f'transaction_type={tt.name.lower()}'] = float(
                        type_str.upper() == tt.name)
            else:
                # If no transaction type, set all to 0
                for tt in TransactionType:
                    trans_attrs[f'transaction_type={tt.name.lower()}'] = 0.0

            # Add currency as one-hot encoded features
            currency = attrs.get('currency')
            for curr in unique_currencies:
                trans_attrs[f'currency={curr}'] = float(
                    currency == curr if currency else False)

            transaction_attrs_list.append(trans_attrs)

            # Store label separately
            transaction_labels.append(
                1 if attrs.get('is_fraudulent', False) else 0)

        elif edge_type == EdgeType.OWNERSHIP or edge_type == 'ownership':
            ownership_edges.append([src_idx, dest_idx])

            # Extract ownership attributes
            own_attrs = {
                'ownership_percentage': float(attrs.get('ownership_percentage', 0.0)),
                'ownership_start_date': attrs.get('ownership_start_date').toordinal()
                if attrs.get('ownership_start_date') else 0.0
            }
            ownership_attrs_list.append(own_attrs)

    # Convert edge lists to tensors
    result = {
        'num_nodes': len(node_list),
        'node_ids': node_list,
        'node_features': node_features_tensor,
        'node_feature_names': node_feature_names,
        'metadata': {
            'num_transaction_edges': len(transaction_edges),
            'num_ownership_edges': len(ownership_edges),
            'total_edges': len(transaction_edges) + len(ownership_edges)
        }
    }

    # Add transaction edge data
    if transaction_edges:
        result['transaction_edge_index'] = torch.tensor(
            transaction_edges, dtype=torch.long).t().contiguous()

        # Convert transaction attributes to tensors
        trans_feat_dict = {}
        for key in transaction_attrs_list[0].keys():
            trans_feat_dict[key] = torch.tensor([attrs[key] for attrs in transaction_attrs_list],
                                                dtype=torch.float32)
        result['transaction_edge_attr'] = trans_feat_dict
        result['transaction_edge_attr_names'] = list(trans_feat_dict.keys())

        # Add edge labels separately
        result['transaction_edge_labels'] = torch.tensor(
            transaction_labels, dtype=torch.long)
    else:
        result['transaction_edge_index'] = torch.empty(
            (2, 0), dtype=torch.long)
        result['transaction_edge_attr'] = {}
        result['transaction_edge_attr_names'] = []
        result['transaction_edge_labels'] = torch.empty((0,), dtype=torch.long)

    # Add ownership edge data
    if ownership_edges:
        result['ownership_edge_index'] = torch.tensor(
            ownership_edges, dtype=torch.long).t().contiguous()

        # Convert ownership attributes to tensors
        own_feat_dict = {}
        for key in ownership_attrs_list[0].keys():
            own_feat_dict[key] = torch.tensor([attrs[key] for attrs in ownership_attrs_list],
                                              dtype=torch.float32)
        result['ownership_edge_attr'] = own_feat_dict
        result['ownership_edge_attr_names'] = list(own_feat_dict.keys())
    else:
        result['ownership_edge_index'] = torch.empty((2, 0), dtype=torch.long)
        result['ownership_edge_attr'] = {}
        result['ownership_edge_attr_names'] = []

    # Add combined edge index for models that don't distinguish edge types
    all_edges = transaction_edges + ownership_edges
    if all_edges:
        result['edge_index'] = torch.tensor(
            all_edges, dtype=torch.long).t().contiguous()
    else:
        result['edge_index'] = torch.empty((2, 0), dtype=torch.long)

    return result


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
    parser.add_argument("--pytorch-file", default="generated_graph.pt",
                        help="Filename for PyTorch export")

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
    pytorch_filepath = os.path.join(args.output_dir, args.pytorch_file)

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
            'Gpickle': False,
            'PyTorch': False
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

    # Respect export filter setting
    transactions_only = output_config.get('TransactionsOnly', False)

    # Optionally create a transactions-only view for exports that use the graph structure
    graph_for_exports = filter_transactions_only(
        graph) if transactions_only else graph

    # Export to GraphML for visualization
    if output_config.get('GraphML', False):
        print("\nSaving graph in GraphML format for visualization...")
        # Convert enums to strings for GraphML compatibility
        converted_graph = convert_enums_to_strings(graph_for_exports)
        nx.write_graphml(converted_graph, graphml_filepath)
        print(f"Graph saved as: {graphml_filepath}")

    # Export to Gpickle
    if output_config.get('Gpickle', False):
        print("\nSaving graph in Gpickle format...")
        # Convert enums to strings to remove tide module dependencies
        converted_graph = convert_enums_to_strings(graph_for_exports)
        with open(gpickle_filepath, 'wb') as f:
            pickle.dump(converted_graph, f, pickle.HIGHEST_PROTOCOL)
        print(f"Graph saved as: {gpickle_filepath}")

    # Export to PyTorch
    if output_config.get('PyTorch', False):
        print("\nSaving graph in PyTorch format...")
        # Convert graph to proper PyTorch format with all edge attributes
        pytorch_data = convert_graph_to_pytorch(graph_for_exports)

        # Try to create PyTorch Geometric Data object if library is available
        try:
            from torch_geometric.data import Data

            # Create PyG Data object with transaction edges (most relevant for AML)
            if pytorch_data['metadata']['num_transaction_edges'] > 0:
                # Combine transaction edge features into a single tensor
                edge_attr_list = []
                for attr_name in pytorch_data['transaction_edge_attr_names']:
                    edge_attr_list.append(
                        pytorch_data['transaction_edge_attr'][attr_name].unsqueeze(
                            1)
                    )
                transaction_edge_features = torch.cat(edge_attr_list, dim=1)

                # Extract node labels (is_fraudulent) as y
                node_labels = []
                for node_id in pytorch_data['node_ids']:
                    is_fraudulent = graph.nodes[node_id].get(
                        'is_fraudulent', False)
                    node_labels.append(1 if is_fraudulent else 0)
                node_labels_tensor = torch.tensor(
                    node_labels, dtype=torch.long)

                pyg_data = Data(
                    x=pytorch_data['node_features'],
                    edge_index=pytorch_data['transaction_edge_index'],
                    edge_attr=transaction_edge_features,  # Now WITHOUT is_fraudulent
                    # Edge labels stored separately
                    edge_y=pytorch_data['transaction_edge_labels'],
                    y=node_labels_tensor,  # Node labels
                    num_nodes=pytorch_data['num_nodes']
                )

                # Add metadata as attributes
                pyg_data.node_feature_names = pytorch_data['node_feature_names']
                pyg_data.edge_attr_names = pytorch_data['transaction_edge_attr_names']
                pyg_data.node_ids = pytorch_data['node_ids']

                # Save the PyG Data object
                torch.save(pyg_data, pytorch_filepath)

                print(
                    f"Graph saved as PyTorch Geometric Data: {pytorch_filepath}")
                print(f"  - Nodes: {pyg_data.num_nodes}")
                print(f"  - Node features: {pyg_data.x.shape}")
                print(f"  - Node feature names: {pyg_data.node_feature_names}")
                print(
                    f"  - Node labels (y): {pyg_data.y.shape}, classes: {pyg_data.y.unique().tolist()}")
                print(
                    f"  - Fraudulent nodes: {pyg_data.y.sum().item()} ({100*pyg_data.y.sum().item()/pyg_data.num_nodes:.2f}%)")
                print(f"  - Transaction edges: {pyg_data.edge_index.shape[1]}")
                print(
                    f"  - Edge features (edge_attr): {pyg_data.edge_attr.shape}")
                print(f"  - Edge feature names: {pyg_data.edge_attr_names}")
                print(
                    f"  - Edge labels (edge_y): {pyg_data.edge_y.shape}, classes: {pyg_data.edge_y.unique().tolist()}")
                print(
                    f"  - Fraudulent edges: {pyg_data.edge_y.sum().item()} ({100*pyg_data.edge_y.sum().item()/pyg_data.edge_index.shape[1]:.2f}%)")
            else:
                # Fallback to dict if no transaction edges
                torch.save(pytorch_data, pytorch_filepath)
                print(
                    f"Graph saved as dictionary (no transaction edges): {pytorch_filepath}")

        except ImportError:
            # If PyTorch Geometric not available, save as dictionary
            torch.save(pytorch_data, pytorch_filepath)
            print(
                f"PyTorch Geometric not installed, saving as dictionary: {pytorch_filepath}")
            print(f"Install with: pip install torch-geometric")

        # Print summary
        print(f"\nSummary:")
        print(f"  - Total nodes: {pytorch_data['num_nodes']}")
        print(
            f"  - Transaction edges: {pytorch_data['metadata']['num_transaction_edges']}")
        if pytorch_data['metadata']['num_transaction_edges'] > 0:
            print(
                f"    - Transaction edge attributes: {pytorch_data['transaction_edge_attr_names']}")
        print(
            f"  - Ownership edges: {pytorch_data['metadata']['num_ownership_edges']}")
        if pytorch_data['metadata']['num_ownership_edges'] > 0:
            print(
                f"    - Ownership edge attributes: {pytorch_data['ownership_edge_attr_names']}")

    print("\nTo visualize the patterns, run:")
    print("python visualize_patterns.py --graph_file generated_graph.graphml --config_file configs/graph.yaml")
