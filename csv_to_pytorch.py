#!/usr/bin/env python3
"""
Build a PyTorch Geometric (.pt) graph file from CSV node and edge files.

This script converts generated CSV files (nodes and edges/transactions) into
a PyTorch Geometric Data object for use with GNN models (GIN, PNA, GBT, etc.)
for fraud edge detection.

Usage:
    python csv_to_pytorch.py --nodes generated_nodes.csv --edges generated_transactions.csv --output graph.pt
    python csv_to_pytorch.py --nodes nodes.csv --edges edges.csv --output graph.pt --include-ownership

The output .pt file contains:
    - x: Node feature tensor
    - edge_index: Edge connectivity (COO format)
    - edge_attr: Edge features
    - edge_y: Edge labels (is_fraudulent)
    - y: Node labels (is_fraudulent)
    - Metadata attributes for feature names
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def normalize_enum_column(series: pd.Series) -> pd.Series:
    """Vectorized normalization of enum-like values to plain strings."""
    # Convert to string, handle NaN
    str_series = series.fillna("").astype(str)
    # Handle "EnumClass.VALUE" format - extract after last dot
    return str_series.str.split(".").str[-1].str.lower().replace("", pd.NA)


def build_node_features(
    nodes_df: pd.DataFrame,
    node_id_col: str = "node_id",
) -> tuple[torch.Tensor, list[str], dict[str, int], list[str], list[int]]:
    """
    Build node feature matrix from nodes DataFrame using vectorized operations.

    Returns:
        - node_features: Tensor of shape (num_nodes, num_features)
        - feature_names: List of feature names
        - node_to_idx: Mapping from node_id to index
        - node_ids: Ordered list of node IDs
        - node_labels: List of fraud labels (0/1)
    """
    logger.info("Building node ID mapping...")
    node_ids = nodes_df[node_id_col].tolist()
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    logger.info(f"  Mapped {len(node_ids)} nodes to indices")

    # Categorical fields and their top-k caps
    categorical_fields = [
        "node_type",
        "country_code",
        "account_category",
        "gender",
        "age_group",
        "currency",
        "business_category",
    ]
    topk_caps = {
        "country_code": 30,
        "currency": 5,
        "business_category": 20,
    }

    # Normalize all categorical columns at once (vectorized)
    logger.info("Normalizing categorical columns...")
    normalized_cats = {}
    for field in categorical_fields:
        if field in nodes_df.columns:
            normalized_cats[field] = normalize_enum_column(nodes_df[field])
        else:
            normalized_cats[field] = pd.Series([pd.NA] * len(nodes_df))

    # Build encoding maps using value_counts (much faster than iterrows)
    logger.info("Building categorical encoding maps...")
    encoding_maps: dict[str, dict[str, int]] = {}

    for field in categorical_fields:
        counts = normalized_cats[field].value_counts()
        if len(counts) == 0:
            encoding_maps[field] = {}
            continue

        if field in topk_caps:
            k = topk_caps[field]
            top_values = counts.head(k).index.tolist()
            mapping = {name: idx for idx, name in enumerate(top_values)}
            mapping["__other__"] = len(top_values)
        else:
            mapping = {name: idx for idx, name in enumerate(counts.index)}
        encoding_maps[field] = mapping
        logger.debug(f"  {field}: {len(mapping)} categories")

    # Build feature names
    feature_names: list[str] = ["incorporation_year", "creation_year"]
    categorical_columns: dict[str, list[str]] = {}

    for field in categorical_fields:
        mapping = encoding_maps[field]
        if not mapping:
            categorical_columns[field] = []
            continue
        names = []
        for name, _ in sorted(mapping.items(), key=lambda x: x[1]):
            if name == "__other__":
                continue
            names.append(f"{field}={name}")
        if "__other__" in mapping:
            names.append(f"{field}=__other__")
        categorical_columns[field] = names
        feature_names.extend(names)

    logger.info(f"Encoding {len(nodes_df)} nodes into feature vectors...")

    # Pre-allocate feature matrix
    num_features = len(feature_names)
    num_nodes = len(nodes_df)
    features = np.zeros((num_nodes, num_features), dtype=np.float32)

    # Feature index tracker
    feat_idx = 0

    # Numeric: incorporation_year (vectorized)
    if "incorporation_year" in nodes_df.columns:
        features[:, feat_idx] = pd.to_numeric(
            nodes_df["incorporation_year"], errors="coerce").fillna(0).values
    feat_idx += 1

    # Derived: creation_year (vectorized datetime parsing)
    if "creation_date" in nodes_df.columns:
        creation_dates = pd.to_datetime(
            nodes_df["creation_date"], errors="coerce")
        features[:, feat_idx] = creation_dates.dt.year.fillna(0).values
    feat_idx += 1

    # Categorical one-hot encoding (vectorized)
    for field in categorical_fields:
        mapping = encoding_maps[field]
        col_names = categorical_columns[field]
        num_cols = len(col_names)
        if num_cols == 0:
            continue

        series = normalized_cats[field]

        if field in topk_caps:
            # Map non-top-k values to __other__
            top_values = set(mapping.keys()) - {"__other__"}
            mapped = series.where(series.isin(top_values), "__other__")
        else:
            mapped = series

        # Use pandas get_dummies for efficient one-hot encoding
        for col_name in col_names:
            # Extract value from "field=value"
            value = col_name.split("=", 1)[1]
            features[:, feat_idx] = (mapped == value).astype(np.float32).values
            feat_idx += 1

    logger.info(f"  Completed encoding {num_nodes} nodes")
    node_features = torch.from_numpy(features)
    logger.info(f"  Node feature tensor shape: {node_features.shape}")

    # Node labels (vectorized)
    if "is_fraudulent" in nodes_df.columns:
        fraud_col = nodes_df["is_fraudulent"]
        if fraud_col.dtype == object:
            node_labels = (fraud_col.str.lower() ==
                           "true").astype(int).tolist()
        else:
            node_labels = fraud_col.fillna(False).astype(int).tolist()
    else:
        node_labels = [0] * num_nodes

    return node_features, feature_names, node_to_idx, node_ids, node_labels


def build_edge_features(
    edges_df: pd.DataFrame,
    node_to_idx: dict[str, int],
    include_ownership: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[str],
    torch.Tensor,
    torch.Tensor | None,
    dict[str, torch.Tensor] | None,
    list[str] | None,
]:
    """
    Build edge index and features from edges DataFrame using vectorized operations.
    """
    logger.info(f"Processing {len(edges_df)} edges...")

    # Normalize edge_type column (vectorized)
    edges_df = edges_df.copy()
    edges_df["edge_type_norm"] = normalize_enum_column(edges_df["edge_type"])

    # Map node IDs to indices (vectorized)
    edges_df["src_idx"] = edges_df["src"].map(node_to_idx)
    edges_df["dest_idx"] = edges_df["dest"].map(node_to_idx)

    # Filter out edges with unknown nodes
    valid_mask = edges_df["src_idx"].notna() & edges_df["dest_idx"].notna()
    skipped_nodes = (~valid_mask).sum()
    if skipped_nodes > 0:
        logger.warning(
            f"  Skipping {skipped_nodes} edges with unknown node IDs")

    edges_df = edges_df[valid_mask].copy()
    edges_df["src_idx"] = edges_df["src_idx"].astype(int)
    edges_df["dest_idx"] = edges_df["dest_idx"].astype(int)

    # Split into transaction and ownership edges
    transaction_mask = edges_df["edge_type_norm"] == "transaction"
    trans_df = edges_df[transaction_mask]

    logger.info(f"  Transaction edges: {len(trans_df)}")

    # Build transaction edge features
    transaction_types = ["payment", "salary",
                         "transfer", "deposit", "withdrawal"]

    if len(trans_df) > 0:
        # Edge index
        transaction_edge_index = torch.tensor(
            np.column_stack([trans_df["src_idx"].values,
                            trans_df["dest_idx"].values]),
            dtype=torch.long
        ).t().contiguous()

        # Get unique currencies
        unique_currencies = sorted(
            trans_df["currency"].dropna().unique().tolist())

        # Build attribute names
        attr_names = ["amount", "timestamp", "time_since_previous"]
        attr_names += [f"transaction_type={tt}" for tt in transaction_types]
        attr_names += [f"currency={curr}" for curr in unique_currencies]

        # Pre-allocate feature matrix
        num_edges = len(trans_df)
        num_features = len(attr_names)
        edge_features = np.zeros((num_edges, num_features), dtype=np.float32)

        feat_idx = 0

        # Amount (vectorized)
        edge_features[:, feat_idx] = pd.to_numeric(
            trans_df["amount"], errors="coerce").fillna(0).values
        feat_idx += 1

        # Timestamp (vectorized)
        timestamps = pd.to_datetime(trans_df["timestamp"], errors="coerce")
        # Convert to Unix timestamp (seconds since epoch)
        edge_features[:, feat_idx] = timestamps.astype(
            np.int64).fillna(0).values / 1e9
        feat_idx += 1

        # Time since previous transaction (vectorized parsing of H:MM:SS)
        if "time_since_previous_transaction" in trans_df.columns:
            time_since = trans_df["time_since_previous_transaction"].fillna("")
            # Parse H:MM:SS format to seconds

            def parse_timedelta_seconds(s):
                if pd.isna(s) or s == "":
                    return 0.0
                try:
                    parts = str(s).split(":")
                    if len(parts) == 3:
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    elif len(parts) == 2:
                        return int(parts[0]) * 3600 + int(parts[1]) * 60
                except (ValueError, TypeError):
                    pass
                return 0.0

            edge_features[:, feat_idx] = time_since.apply(
                parse_timedelta_seconds).values
        feat_idx += 1

        # One-hot encode transaction type (vectorized)
        trans_type_norm = normalize_enum_column(trans_df["transaction_type"])
        for tt in transaction_types:
            edge_features[:, feat_idx] = (
                trans_type_norm == tt).astype(np.float32).values
            feat_idx += 1

        # One-hot encode currency (vectorized)
        for curr in unique_currencies:
            edge_features[:, feat_idx] = (
                trans_df["currency"] == curr).astype(np.float32).values
            feat_idx += 1

        transaction_edge_attr = torch.from_numpy(edge_features)

        # Labels (vectorized)
        if "is_fraudulent" in trans_df.columns:
            fraud_col = trans_df["is_fraudulent"]
            if fraud_col.dtype == object:
                labels = (fraud_col.str.lower() == "true").astype(int).values
            else:
                labels = fraud_col.fillna(False).astype(int).values
        else:
            labels = np.zeros(num_edges, dtype=int)

        transaction_edge_labels = torch.tensor(labels, dtype=torch.long)
    else:
        transaction_edge_index = torch.empty((2, 0), dtype=torch.long)
        transaction_edge_attr = torch.empty((0, 0), dtype=torch.float32)
        attr_names = []
        transaction_edge_labels = torch.empty((0,), dtype=torch.long)

    # Build ownership edges if requested
    ownership_edge_index = None
    ownership_attr_dict = None
    ownership_attr_names = None

    if include_ownership:
        ownership_mask = edges_df["edge_type_norm"] == "ownership"
        own_df = edges_df[ownership_mask]
        logger.info(f"  Ownership edges: {len(own_df)}")

        if len(own_df) > 0:
            ownership_edge_index = torch.tensor(
                np.column_stack([own_df["src_idx"].values,
                                own_df["dest_idx"].values]),
                dtype=torch.long
            ).t().contiguous()

            ownership_attr_names = [
                "ownership_percentage", "ownership_start_date"]

            # Ownership percentage (vectorized)
            pct = pd.to_numeric(own_df.get("ownership_percentage", pd.Series([0.0] * len(own_df))),
                                errors="coerce").fillna(0).values

            # Ownership start date (vectorized)
            start_dates = pd.to_datetime(own_df.get("ownership_start_date", pd.Series([pd.NaT] * len(own_df))),
                                         errors="coerce")
            # Convert to ordinal
            start_ordinals = start_dates.apply(
                lambda x: x.toordinal() if pd.notna(x) else 0.0).values

            ownership_attr_dict = {
                "ownership_percentage": torch.tensor(pct, dtype=torch.float32),
                "ownership_start_date": torch.tensor(start_ordinals, dtype=torch.float32),
            }

    logger.info("  Edge processing complete")

    return (
        transaction_edge_index,
        transaction_edge_attr,
        attr_names,
        transaction_edge_labels,
        ownership_edge_index,
        ownership_attr_dict,
        ownership_attr_names,
    )


def build_pytorch_graph(
    nodes_csv: str | Path,
    edges_csv: str | Path,
    output_path: str | Path,
    include_ownership: bool = False,
) -> None:
    """
    Build PyTorch Geometric graph from CSV files and save to .pt file.

    Args:
        nodes_csv: Path to nodes CSV file
        edges_csv: Path to edges/transactions CSV file
        output_path: Output .pt file path
        include_ownership: Whether to include ownership edges
    """
    logger.info(f"Loading nodes from: {nodes_csv}")
    nodes_df = pd.read_csv(nodes_csv, low_memory=False)
    logger.info(f"  Loaded {len(nodes_df)} nodes")

    logger.info(f"Loading edges from: {edges_csv}")
    edges_df = pd.read_csv(edges_csv, low_memory=False)
    logger.info(f"  Loaded {len(edges_df)} edges")

    logger.info("Starting node feature extraction...")
    (
        node_features,
        node_feature_names,
        node_to_idx,
        node_ids,
        node_labels,
    ) = build_node_features(nodes_df)
    logger.info(f"Node features complete: shape={node_features.shape}")
    logger.info(
        f"  Feature names ({len(node_feature_names)}): {node_feature_names[:5]}...")

    logger.info("Starting edge feature extraction...")
    (
        transaction_edge_index,
        transaction_edge_attr,
        transaction_attr_names,
        transaction_edge_labels,
        ownership_edge_index,
        ownership_attr_dict,
        ownership_attr_names,
    ) = build_edge_features(edges_df, node_to_idx, include_ownership)

    num_trans = transaction_edge_index.shape[1]
    num_own = ownership_edge_index.shape[1] if ownership_edge_index is not None else 0
    logger.info(
        f"Edge features complete: {num_trans} transactions, {num_own} ownership")
    if transaction_attr_names:
        logger.info(
            f"  Edge feature names ({len(transaction_attr_names)}): {transaction_attr_names[:5]}...")

    logger.info("Creating PyTorch Geometric Data object...")
    try:
        from torch_geometric.data import Data

        node_labels_tensor = torch.tensor(node_labels, dtype=torch.long)

        data = Data(
            x=node_features,
            edge_index=transaction_edge_index,
            edge_attr=transaction_edge_attr,
            edge_y=transaction_edge_labels,
            y=node_labels_tensor,
            num_nodes=len(node_ids),
        )

        # Add metadata
        data.node_feature_names = node_feature_names
        data.edge_attr_names = transaction_attr_names
        data.node_ids = node_ids

        # Optionally add ownership edges
        if ownership_edge_index is not None:
            data.ownership_edge_index = ownership_edge_index
            data.ownership_edge_attr = ownership_attr_dict
            data.ownership_edge_attr_names = ownership_attr_names
            logger.info(
                f"  Added ownership edges: {ownership_edge_index.shape[1]}")

        logger.info(f"Saving PyTorch Geometric Data to: {output_path}")
        torch.save(data, output_path)
        logger.info("Save complete!")

        # Summary
        logger.info("=== Graph Summary ===")
        logger.info(f"  Nodes: {data.num_nodes}")
        logger.info(f"  Node features: {data.x.shape}")
        logger.info(
            f"  Fraudulent nodes: {data.y.sum().item()} ({100*data.y.sum().item()/data.num_nodes:.2f}%)")
        logger.info(f"  Transaction edges: {data.edge_index.shape[1]}")
        logger.info(f"  Edge features: {data.edge_attr.shape}")
        logger.info(
            f"  Fraudulent edges: {data.edge_y.sum().item()} ({100*data.edge_y.sum().item()/data.edge_index.shape[1]:.4f}%)")
        if ownership_edge_index is not None:
            logger.info(f"  Ownership edges: {ownership_edge_index.shape[1]}")

    except ImportError:
        # Fallback to dictionary format
        logger.warning(
            "PyTorch Geometric not installed, saving as dictionary format...")
        result = {
            "num_nodes": len(node_ids),
            "node_ids": node_ids,
            "node_features": node_features,
            "node_feature_names": node_feature_names,
            "node_labels": torch.tensor(node_labels, dtype=torch.long),
            "transaction_edge_index": transaction_edge_index,
            "transaction_edge_attr": transaction_edge_attr,
            "transaction_edge_attr_names": transaction_attr_names,
            "transaction_edge_labels": transaction_edge_labels,
            "edge_index": transaction_edge_index,  # Alias for compatibility
            "metadata": {
                "num_transaction_edges": num_trans,
                "num_ownership_edges": num_own,
                "total_edges": num_trans + num_own,
            },
        }

        if ownership_edge_index is not None:
            result["ownership_edge_index"] = ownership_edge_index
            result["ownership_edge_attr"] = ownership_attr_dict
            result["ownership_edge_attr_names"] = ownership_attr_names

        torch.save(result, output_path)
        logger.info(f"Saved dictionary to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CSV node/edge files to PyTorch Geometric format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with transaction edges only
    python csv_to_pytorch.py --nodes nodes.csv --edges transactions.csv --output graph.pt

    # Include ownership edges
    python csv_to_pytorch.py --nodes nodes.csv --edges edges.csv --output graph.pt --include-ownership

Output format:
    The .pt file contains a PyTorch Geometric Data object with:
    - x: Node features (including one-hot encoded categoricals)
    - edge_index: Transaction edge connectivity
    - edge_attr: Transaction edge features
    - edge_y: Fraud labels for edges (target for fraud detection)
    - y: Fraud labels for nodes
    - node_feature_names, edge_attr_names: Feature name lists
""",
    )
    parser.add_argument(
        "--nodes",
        required=True,
        help="Path to nodes CSV file (generated_nodes.csv)",
    )
    parser.add_argument(
        "--edges",
        required=True,
        help="Path to edges CSV file (generated_edges.csv or generated_transactions.csv)",
    )
    parser.add_argument(
        "--output",
        default="graph.pt",
        help="Output .pt file path (default: graph.pt)",
    )
    parser.add_argument(
        "--include-ownership",
        action="store_true",
        help="Include ownership edges in addition to transaction edges",
    )

    args = parser.parse_args()

    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    output_path = Path(args.output)

    if not nodes_path.exists():
        print(f"Error: Nodes file not found: {nodes_path}", file=sys.stderr)
        return 1
    if not edges_path.exists():
        print(f"Error: Edges file not found: {edges_path}", file=sys.stderr)
        return 1

    build_pytorch_graph(
        nodes_csv=nodes_path,
        edges_csv=edges_path,
        output_path=output_path,
        include_ownership=args.include_ownership,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
