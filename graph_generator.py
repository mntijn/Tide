import networkx as nx
import random
import datetime
import csv
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple

from data_structures import (
    NodeType, EdgeType, TransactionType,
    NodeAttributes, AccountAttributes, EdgeAttributes,
    TransactionAttributes, OwnershipAttributes
)
from utils import SAMPLE_GEO_LOCATIONS


class GraphGenerator:
    def __init__(self, params: Dict[str, Any]):

        self.params = params
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.all_nodes: Dict[NodeType, List[Any]] = {nt: [] for nt in NodeType}

        self.graph_scale = params.get("graph_scale", {
            "individuals": 50,
            "businesses": 10,
            "institutions": 10,
            "individual_accounts_per_institution_range": (1, 3),
            "business_accounts_per_institution_range": (1, 6)
        })
        self.time_span = params.get("time_span", {
            # Time span of the simulation, entities can be created before this date.
            # Format is (year, month, day, hour, minute, second)
            "start_date": datetime.datetime(2023, 1, 1, 0, 0, 0),
            "end_date": datetime.datetime(2023, 3, 15, 23, 59, 59)
        })

        self.background_tx_rate_per_account_per_day = params.get(
            "transaction_rates", {}).get("per_account_per_day", 0.2)
        self.num_aml_patterns = params.get(
            "pattern_frequency", {}).get("num_illicit_patterns", 5)

    def num_of_nodes(self):
        return self.node_counter

    def num_of_edges(self):
        return self.graph.number_of_edges()

    def _add_node(self, node_type: NodeType, creation_date: datetime.datetime, **kwargs) -> str:
        node_id = f"{node_type.value}_{self.node_counter}"
        self.node_counter += 1

        base_attrs = asdict(NodeAttributes(
            node_type=node_type,
            creation_date=creation_date,
            geo_location=random.choice(
                SAMPLE_GEO_LOCATIONS),
            is_fraudulent=False
        ))

        specific_attrs = {}
        if node_type == NodeType.ACCOUNT:
            start_balance = kwargs.pop(
                "start_balance", random.uniform(100, 100000))
            institution_id = kwargs.pop("institution_id", None)
            specific_attrs = asdict(AccountAttributes(
                start_balance=start_balance,
                current_balance=start_balance,
                institution_id=institution_id
            ))

        self.graph.add_node(node_id, **base_attrs, **specific_attrs, **kwargs)
        self.all_nodes[node_type].append(node_id)
        return node_id

    def _add_edge(self, src_id: str, dest_id: str, attributes: EdgeAttributes):
        self.graph.add_edge(src_id, dest_id, **asdict(attributes))

    def initialize_entities(self):
        print("Creating graph entities..")
        start_date = self.time_span["start_date"]

        for i in range(self.graph_scale.get("institutions")):
            inst_id = self._add_node(NodeType.INSTITUTION, creation_date=start_date -
                                     datetime.timedelta(days=random.randint(365, 3650)), name=f"Bank_{i+1}")

        for i in range(self.graph_scale.get("individuals")):
            ind_id = self._add_node(NodeType.INDIVIDUAL, creation_date=start_date -
                                    datetime.timedelta(days=random.randint(30, 1000)))
            self.create_accounts_for_entity(ind_id, start_date)

        for i in range(self.graph_scale.get("businesses")):
            bus_id = self._add_node(NodeType.BUSINESS, creation_date=start_date -
                                    datetime.timedelta(days=random.randint(90, 2000)))
            self.create_accounts_for_entity(bus_id, start_date)

        print("Created entities.")

    def create_accounts_for_entity(self, entity_id: str, sim_start_date: datetime.datetime):
        # Get entity attributes using its ID
        entity_attrs = self.graph.nodes[entity_id]
        entity_node_type = entity_attrs['node_type']
        entity_creation_date = entity_attrs['creation_date']

        if entity_node_type == NodeType.INDIVIDUAL:
            num_accounts = random.randint(
                *self.graph_scale.get("individual_accounts_per_institution_range"))
        elif entity_node_type == NodeType.BUSINESS:
            num_accounts = random.randint(
                *self.graph_scale.get("business_accounts_per_institution_range"))
        else:
            raise ValueError(
                f"Invalid entity type for account creation: {entity_node_type}")

        available_institutions = self.all_nodes.get(NodeType.INSTITUTION)
        if not available_institutions:
            raise ValueError(
                "No institutions available to create accounts for entity")

        for _ in range(num_accounts):
            # Account creation date should be between entity creation and simulation start
            min_date = entity_creation_date
            max_date = sim_start_date

            time_delta_days = (max_date - min_date).days

            acc_creation_offset_days = random.randint(
                0, time_delta_days) if time_delta_days >= 0 else 0
            acc_creation_date = entity_creation_date + \
                datetime.timedelta(days=acc_creation_offset_days)

            chosen_institution_id = random.choice(available_institutions)
            start_balance = random.uniform(100, 100000)

            acc_id = self._add_node(
                NodeType.ACCOUNT,
                creation_date=acc_creation_date,
                institution_id=chosen_institution_id,
                start_balance=start_balance
            )

            ownership_instance = OwnershipAttributes(
                ownership_start_date=acc_creation_date.date(),
            )
            self._add_edge(entity_id, acc_id, ownership_instance)

    def generate_graph(self):
        print("Starting graph generation...")
        self.initialize_entities()
        print(
            f"Graph generation complete. Total nodes: {self.num_of_nodes()}, Total edges: {self.num_of_edges()}")
        return self.graph

    def export_to_csv(self, nodes_filepath="nodes.csv", edges_filepath="edges.csv"):
        print(f"Exporting graph to CSV: {nodes_filepath}, {edges_filepath}")

        # Nodes
        with open(nodes_filepath, 'w', newline='') as f_nodes:
            node_fieldnames = ['node_id'] + [f.name for f in NodeAttributes.__dataclass_fields__.values()] + \
                              [f.name for f in AccountAttributes.__dataclass_fields__.values()]
            node_fieldnames = sorted(list(set(node_fieldnames)))

            writer = csv.DictWriter(
                f_nodes, fieldnames=node_fieldnames, extrasaction='ignore')
            writer.writeheader()
            for node_id, attrs in self.graph.nodes(data=True):
                row = {'node_id': node_id, **attrs}
                writer.writerow(row)

        # Edges
        with open(edges_filepath, 'w', newline='') as f_edges:
            edge_fieldnames = ['src', 'dest'] + \
                              [f.name for f in EdgeAttributes.__dataclass_fields__.values()] + \
                              [f.name for f in TransactionAttributes.__dataclass_fields__.values()] + \
                              [f.name for f in OwnershipAttributes.__dataclass_fields__.values()]
            edge_fieldnames = sorted(
                list(set(edge_fieldnames) - {'edge_type'}))
            edge_fieldnames = ['src', 'dest', 'edge_type'] + \
                [fn for fn in edge_fieldnames if fn not in [
                    'src', 'dest', 'edge_type']]

            writer = csv.DictWriter(
                f_edges, fieldnames=edge_fieldnames, extrasaction='ignore')
            writer.writeheader()
            for src, dest, attrs in self.graph.edges(data=True):
                row = {'src': src, 'dest': dest, **attrs}
                writer.writerow(row)
