import csv
from typing import Dict, Any
import networkx as nx
from ..datastructures.attributes import (
    NodeAttributes, AccountAttributes, EdgeAttributes,
    TransactionAttributes, OwnershipAttributes,
    IndividualAttributes, BusinessAttributes, InstitutionAttributes
)
from ..datastructures.enums import EdgeType


def export_to_csv(graph: nx.DiGraph, nodes_filepath: str = "nodes.csv", edges_filepath: str = "edges.csv", transactions_filepath: str = "generated_transactions.csv"):
    """Export the graph to CSV files for nodes and edges."""
    print(
        f"Exporting graph to CSV: {nodes_filepath}, {edges_filepath}, {transactions_filepath}")

    # Nodes
    with open(nodes_filepath, 'w', newline='') as f_nodes:
        node_fieldnames = ['node_id'] + \
                          [f.name for f in NodeAttributes.__dataclass_fields__.values()] + \
                          [f.name for f in AccountAttributes.__dataclass_fields__.values()] + \
                          [f.name for f in IndividualAttributes.__dataclass_fields__.values()] + \
                          [f.name for f in BusinessAttributes.__dataclass_fields__.values()] + \
                          [f.name for f in InstitutionAttributes.__dataclass_fields__.values()]
        node_fieldnames = sorted(list(set(node_fieldnames)))

        writer = csv.DictWriter(
            f_nodes, fieldnames=node_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for node_id, attrs in graph.nodes(data=True):
            row = {'node_id': node_id, **attrs}
            writer.writerow(row)

    # Edges
    edge_fieldnames = ['src', 'dest'] + \
                      [f.name for f in EdgeAttributes.__dataclass_fields__.values()] + \
                      [f.name for f in TransactionAttributes.__dataclass_fields__.values()] + \
                      [f.name for f in OwnershipAttributes.__dataclass_fields__.values()]
    edge_fieldnames = sorted(
        list(set(edge_fieldnames) - {'edge_type'}))
    edge_fieldnames = ['src', 'dest', 'edge_type'] + \
        [fn for fn in edge_fieldnames if fn not in [
            'src', 'dest', 'edge_type']]

    with open(edges_filepath, 'w', newline='') as f_edges:
        writer = csv.DictWriter(
            f_edges, fieldnames=edge_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for src, dest, attrs in graph.edges(data=True):
            row = {'src': src, 'dest': dest, **attrs}
            if isinstance(row.get('edge_type'), EdgeType):
                row['edge_type'] = row['edge_type'].value
            writer.writerow(row)

    # Transactions only
    with open(transactions_filepath, 'w', newline='') as f_transactions:
        writer = csv.DictWriter(
            f_transactions, fieldnames=edge_fieldnames, extrasaction='ignore')
        writer.writeheader()
        for src, dest, attrs in graph.edges(data=True):
            if attrs.get('edge_type') == EdgeType.TRANSACTION:
                row = {'src': src, 'dest': dest, **attrs}
                if isinstance(row.get('edge_type'), EdgeType):
                    row['edge_type'] = row['edge_type'].value
                writer.writerow(row)
