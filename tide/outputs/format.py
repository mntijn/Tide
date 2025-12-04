import csv
from typing import Dict, Any, Optional, Set
import networkx as nx
from ..datastructures.attributes import (
    NodeAttributes, AccountAttributes, EdgeAttributes,
    TransactionAttributes, OwnershipAttributes,
    IndividualAttributes, BusinessAttributes, InstitutionAttributes
)
from ..datastructures.enums import EdgeType, NodeType


def get_filtered_accounts(graph: nx.DiGraph, filter_country: str) -> Set[str]:
    """
    Get the set of account IDs that belong to institutions in the specified country.

    Args:
        graph: The NetworkX graph containing nodes
        filter_country: Country code to filter by (e.g., "NL" for Netherlands)

    Returns:
        Set of account node IDs belonging to institutions in the target country
    """
    # Find all institution IDs in the target country
    target_institutions = set()
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get('node_type')
        if node_type == NodeType.INSTITUTION or str(node_type) == 'institution':
            if attrs.get('country_code') == filter_country:
                target_institutions.add(node_id)

    # Find all accounts belonging to those institutions
    filtered_accounts = set()
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get('node_type')
        if node_type == NodeType.ACCOUNT or str(node_type) == 'account':
            if attrs.get('institution_id') in target_institutions:
                filtered_accounts.add(node_id)

    return filtered_accounts


def export_to_csv(
    graph: nx.DiGraph,
    nodes_filepath: str = "nodes.csv",
    edges_filepath: str = "edges.csv",
    transactions_filepath: str = "generated_transactions.csv",
    institution_filter_country: Optional[str] = None,
    remove_isolated_nodes: bool = False
):
    """Export the graph to CSV files for nodes and edges.

    Args:
        graph: The NetworkX graph to export
        nodes_filepath: Output path for nodes CSV
        edges_filepath: Output path for edges CSV
        transactions_filepath: Output path for transactions-only CSV
        institution_filter_country: If set, only export transactions where at least
            one account belongs to an institution in this country (e.g., "NL")
        remove_isolated_nodes: If True, only export nodes that have at least one edge
    """
    # Build filtered account set if filtering is enabled
    filtered_accounts: Optional[Set[str]] = None
    nodes_with_edges: Optional[Set[str]] = None

    if institution_filter_country:
        filtered_accounts = get_filtered_accounts(
            graph, institution_filter_country)
        print(f"Institution filter enabled: country={institution_filter_country}, "
              f"found {len(filtered_accounts)} accounts in target institutions")

    # Track which nodes have at least one edge (for node filtering)
    if institution_filter_country or remove_isolated_nodes:
        nodes_with_edges = set()

    print(
        f"Exporting graph to CSV: {nodes_filepath}, {edges_filepath}, {transactions_filepath}")

    # Edges (process first to identify nodes with edges)
    edge_fieldnames = ['src', 'dest'] + \
                      [f.name for f in EdgeAttributes.__dataclass_fields__.values()] + \
                      [f.name for f in TransactionAttributes.__dataclass_fields__.values()] + \
                      [f.name for f in OwnershipAttributes.__dataclass_fields__.values()]
    edge_fieldnames = sorted(
        list(set(edge_fieldnames) - {'edge_type'}))
    edge_fieldnames = ['src', 'dest', 'edge_type'] + \
        [fn for fn in edge_fieldnames if fn not in [
            'src', 'dest', 'edge_type']]

    total_edges = 0
    filtered_edges = 0
    with open(edges_filepath, 'w', newline='') as f_edges:
        writer = csv.DictWriter(
            f_edges, fieldnames=edge_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for src, dest, attrs in graph.edges(data=True):
            total_edges += 1
            # Apply institution filter: include if src OR dest is in filtered accounts
            if filtered_accounts is not None:
                if src not in filtered_accounts and dest not in filtered_accounts:
                    continue
            filtered_edges += 1

            # Track nodes that have edges
            if nodes_with_edges is not None:
                nodes_with_edges.add(src)
                nodes_with_edges.add(dest)

            row = {'src': src, 'dest': dest, **attrs}
            if isinstance(row.get('edge_type'), EdgeType):
                row['edge_type'] = row['edge_type'].value
            writer.writerow(row)

    # Transactions only
    total_transactions = 0
    filtered_transactions = 0
    with open(transactions_filepath, 'w', newline='') as f_transactions:
        writer = csv.DictWriter(
            f_transactions, fieldnames=edge_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for src, dest, attrs in graph.edges(data=True):
            if attrs.get('edge_type') == EdgeType.TRANSACTION:
                total_transactions += 1
                # Apply institution filter: include if src OR dest is in filtered accounts
                if filtered_accounts is not None:
                    if src not in filtered_accounts and dest not in filtered_accounts:
                        continue
                filtered_transactions += 1
                row = {'src': src, 'dest': dest, **attrs}
                if isinstance(row.get('edge_type'), EdgeType):
                    row['edge_type'] = row['edge_type'].value
                writer.writerow(row)

    # Nodes (write after edges so we know which nodes to keep)
    node_fieldnames = ['node_id'] + \
                      [f.name for f in NodeAttributes.__dataclass_fields__.values()] + \
                      [f.name for f in AccountAttributes.__dataclass_fields__.values()] + \
                      [f.name for f in IndividualAttributes.__dataclass_fields__.values()] + \
                      [f.name for f in BusinessAttributes.__dataclass_fields__.values()] + \
                      [f.name for f in InstitutionAttributes.__dataclass_fields__.values()]
    node_fieldnames = sorted(list(set(node_fieldnames)))

    total_nodes = 0
    filtered_nodes = 0
    with open(nodes_filepath, 'w', newline='') as f_nodes:
        writer = csv.DictWriter(
            f_nodes, fieldnames=node_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for node_id, attrs in graph.nodes(data=True):
            total_nodes += 1
            # If filtering is enabled, only export nodes that have edges
            if nodes_with_edges is not None:
                if node_id not in nodes_with_edges:
                    continue
            filtered_nodes += 1
            row = {'node_id': node_id, **attrs}
            writer.writerow(row)

    # Log filtering summary
    if nodes_with_edges is not None:
        if filtered_accounts is not None:
            print(f"Institution filter applied: {filtered_edges}/{total_edges} edges exported, "
                  f"{filtered_transactions}/{total_transactions} transactions exported, "
                  f"{filtered_nodes}/{total_nodes} nodes exported")
        else:
            print(
                f"Isolated nodes removed: {filtered_nodes}/{total_nodes} nodes exported")
