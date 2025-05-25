import networkx as nx
import random
import datetime
import csv
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple
from faker import Faker

from .data_structures import (
    NodeType, EdgeType, TransactionType,
    NodeAttributes, AccountAttributes, EdgeAttributes,
    TransactionAttributes, OwnershipAttributes,
    IndividualAttributes, BusinessAttributes, InstitutionAttributes
)
from .utils import COUNTRY_CODES
from .entity_creators import (
    InstitutionCreator, IndividualCreator, BusinessCreator, AccountCreator,
    calculate_age_specific_business_rates
)


class GraphGenerator:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.all_nodes: Dict[NodeType, List[str]] = {nt: [] for nt in NodeType}
        self.fraudulent_entities_map: Dict[NodeType, List[str]] = {
            nt: [] for nt in NodeType}
        self.faker = Faker()

        self.graph_scale = params.get("graph_scale", {})
        self.time_span = params.get("time_span", {})
        self.fraud_selection_config = params.get("fraud_selection_config", {})

        if isinstance(self.time_span.get("start_date"), str):
            self.time_span["start_date"] = datetime.datetime.fromisoformat(
                self.time_span["start_date"])
        else:
            self.time_span["start_date"] = self.time_span.get(
                "start_date", datetime.datetime(2023, 1, 1, 0, 0, 0))

        if isinstance(self.time_span.get("end_date"), str):
            self.time_span["end_date"] = datetime.datetime.fromisoformat(
                self.time_span["end_date"])
        else:
            self.time_span["end_date"] = self.time_span.get(
                "end_date", datetime.datetime(2023, 3, 15, 23, 59, 59))

        self.params["time_span"] = self.time_span

        self.institution_creator = InstitutionCreator(self.params)
        self.individual_creator = IndividualCreator(self.params)
        self.business_creator = BusinessCreator(self.params)
        self.account_creator: Optional[AccountCreator] = None

        self.background_tx_rate_per_account_per_day = params.get(
            "transaction_rates", {}).get("per_account_per_day", 0.2)
        self.num_aml_patterns = params.get(
            "pattern_frequency", {}).get("num_illicit_patterns", 5)

        self.validate_configuration()

    def num_of_nodes(self):
        return self.node_counter

    def num_of_edges(self):
        return self.graph.number_of_edges()

    def validate_configuration(self):
        """Validate that the configuration parameters make sense."""
        total_entities = (self.graph_scale.get("individuals", 0) +
                          self.graph_scale.get("businesses", 0))

        if self.num_aml_patterns > total_entities:
            print(
                f"Warning: Number of AML patterns ({self.num_aml_patterns}) exceeds total entities ({total_entities})")
            print("Consider reducing num_illicit_patterns or increasing graph scale.")

        if self.graph_scale.get("institutions", 0) == 0:
            print("Warning: No institutions configured. Accounts cannot be created.")

        min_risk_threshold = self.fraud_selection_config.get(
            "min_risk_score_for_fraud_consideration", 0.20)
        if min_risk_threshold > 1.0 or min_risk_threshold < 0.0:
            print(
                f"Warning: min_risk_score_for_fraud_consideration ({min_risk_threshold}) should be between 0.0 and 1.0")

    def _add_node(self,
                  node_type: NodeType,
                  common_attrs: Dict[str, Any],
                  specific_attrs: Dict[str, Any],
                  creation_date: Optional[datetime.datetime] = None,
                  **kwargs) -> str:
        node_id = f"{node_type.value}_{self.node_counter}"
        self.node_counter += 1

        node_attributes_data = {
            "node_type": node_type,
            **common_attrs
        }
        if creation_date is not None:
            node_attributes_data["creation_date"] = creation_date

        final_node_attrs = asdict(NodeAttributes(**node_attributes_data))

        self.graph.add_node(node_id, **final_node_attrs,
                            **specific_attrs, **kwargs)
        self.all_nodes[node_type].append(node_id)
        return node_id

    def _add_edge(self, src_id: str, dest_id: str, attributes: EdgeAttributes):
        self.graph.add_edge(src_id, dest_id, **asdict(attributes))

    def initialize_entities(self):
        print("Creating graph entities..")
        sim_start_date = self.time_span["start_date"]

        # Create Institutions
        institutions_data = self.institution_creator.generate_institutions_data()
        institution_countries = {}
        for common_attrs, specific_attrs in institutions_data:
            institution_id = self._add_node(NodeType.INSTITUTION, common_attrs,
                                            specific_attrs, creation_date=None)
            # Store the institution's country for account creation
            institution_countries[institution_id] = common_attrs["address"]["country"]

        # Initialize AccountCreator now that institutions exist
        all_institution_ids = self.all_nodes.get(NodeType.INSTITUTION, [])
        self.account_creator = AccountCreator(
            self.params, all_institution_ids, institution_countries)

        # Create Individuals and their Accounts
        individuals_data = self.individual_creator.generate_individuals_data()
        individual_business_ownership_rate = self.params.get(
            "individual_business_ownership_rate")
        age_group_probabilities = self.params.get(
            "age_group_business_probabilities")

        # Calculate normalized age-specific probabilities
        age_specific_rates = calculate_age_specific_business_rates(
            individuals_data, individual_business_ownership_rate, age_group_probabilities)

        for ind_creation_date, common_attrs, specific_attrs in individuals_data:
            ind_id = self._add_node(
                NodeType.INDIVIDUAL, common_attrs, specific_attrs, creation_date=ind_creation_date)

            # Check if this individual should own a business using age-specific probability
            age_group_value = specific_attrs["age_group"].value
            age_specific_rate = age_specific_rates.get(age_group_value, 0.0)
            if random.random() < age_specific_rate:
                # Create an age-consistent business for this individual
                business_data = self.business_creator.generate_age_consistent_business_for_individual(
                    individual_age_group=specific_attrs["age_group"],
                    individual_creation_date=ind_creation_date,
                    sim_start_date=sim_start_date
                )
                bus_creation_date, bus_common_attrs, bus_specific_attrs = business_data
                bus_id = self._add_node(
                    NodeType.BUSINESS, bus_common_attrs, bus_specific_attrs, creation_date=bus_creation_date)

                # Create ownership relationship between individual and business
                ownership_attrs = OwnershipAttributes(
                    ownership_start_date=bus_creation_date.date(),
                    ownership_percentage=100.0
                )
                self._add_edge(ind_id, bus_id, ownership_attrs)

                # Create accounts for the business
                if self.account_creator and all_institution_ids:
                    bus_accounts_and_ownerships = self.account_creator.generate_accounts_and_ownership_data_for_entity(
                        entity_node_type=NodeType.BUSINESS,
                        entity_creation_date=bus_creation_date,
                        entity_country_code=bus_common_attrs["address"]["country"],
                        sim_start_date=sim_start_date
                    )
                    for acc_creation_date, acc_common, acc_specific, owner_specific in bus_accounts_and_ownerships:
                        acc_id = self._add_node(
                            NodeType.ACCOUNT, acc_common, acc_specific, creation_date=acc_creation_date)
                        ownership_instance = OwnershipAttributes(
                            **owner_specific)
                        self._add_edge(bus_id, acc_id, ownership_instance)

            # Now, generate and add accounts for this individual
            if self.account_creator and all_institution_ids:
                accounts_and_ownerships = self.account_creator.generate_accounts_and_ownership_data_for_entity(
                    entity_node_type=NodeType.INDIVIDUAL,
                    entity_creation_date=ind_creation_date,
                    entity_country_code=common_attrs["address"]["country"],
                    sim_start_date=sim_start_date
                )
                for acc_creation_date, acc_common, acc_specific, owner_specific in accounts_and_ownerships:
                    acc_id = self._add_node(
                        NodeType.ACCOUNT, acc_common, acc_specific, creation_date=acc_creation_date)
                    ownership_instance = OwnershipAttributes(**owner_specific)
                    self._add_edge(ind_id, acc_id, ownership_instance)

        # Create remaining standalone businesses and their accounts
        current_business_count = len(self.all_nodes.get(NodeType.BUSINESS, []))
        target_business_count = self.graph_scale.get("businesses", 0)
        remaining_businesses_to_create = max(
            0, target_business_count - current_business_count)

        if remaining_businesses_to_create > 0:
            original_business_count = self.graph_scale.get("businesses")
            self.graph_scale["businesses"] = remaining_businesses_to_create

            businesses_data = self.business_creator.generate_businesses_data()
            for bus_creation_date, common_attrs, specific_attrs in businesses_data:
                bus_id = self._add_node(
                    NodeType.BUSINESS, common_attrs, specific_attrs, creation_date=bus_creation_date)
                # Generate and add accounts for this business
                if self.account_creator and all_institution_ids:
                    accounts_and_ownerships = self.account_creator.generate_accounts_and_ownership_data_for_entity(
                        entity_node_type=NodeType.BUSINESS,
                        entity_creation_date=bus_creation_date,
                        entity_country_code=common_attrs["address"]["country"],
                        sim_start_date=sim_start_date
                    )
                    for acc_creation_date, acc_common, acc_specific, owner_specific in accounts_and_ownerships:
                        acc_id = self._add_node(
                            NodeType.ACCOUNT, acc_common, acc_specific, creation_date=acc_creation_date)
                        ownership_instance = OwnershipAttributes(
                            **owner_specific)
                        self._add_edge(bus_id, acc_id, ownership_instance)

            self.graph_scale["businesses"] = original_business_count

        print("Created entities.")

    def select_fraudulent_entities(self):
        print("Selecting fraudulent entities based on risk scores...")
        min_risk_score = self.fraud_selection_config.get(
            "min_risk_score_for_fraud_consideration", 0.20)
        base_prob = self.fraud_selection_config.get(
            "base_fraud_probability_if_considered", 0.10)

        fraud_candidates = 0
        selected_fraudulent = 0

        # Consider Individuals and Businesses for being marked as fraudulent
        for node_type in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
            for node_id in self.all_nodes.get(node_type, []):
                node_data = self.graph.nodes[node_id]
                risk_score = node_data.get("risk_score", 0.0)

                if risk_score >= min_risk_score:
                    fraud_candidates += 1
                    # Use the base probability for all entities that meet the minimum risk threshold
                    if random.random() < base_prob:
                        self.graph.nodes[node_id]["is_fraudulent"] = True
                        self.fraudulent_entities_map[node_type].append(node_id)
                        selected_fraudulent += 1

        print(
            f"Considered {fraud_candidates} entities for fraud based on risk_score >= {min_risk_score}.")
        print(f"Selected {selected_fraudulent} entities as fraudulent.")
        for nt, count in self.fraudulent_entities_map.items():
            if count:  # Only print if there are fraudulent entities of this type
                print(f"  - {len(count)} {nt.value}(s)")

    def inject_aml_patterns(self):
        """Inject combined temporal and structural AML patterns using PatternManager."""
        print(f"Injecting up to {self.num_aml_patterns} AML patterns...")
        return

    def simulate_background_activity(self):
        print("Simulating background financial activity...")
        pass

    def generate_graph(self):
        print("Starting graph generation...")
        self.initialize_entities()
        self.select_fraudulent_entities()
        self.inject_aml_patterns()
        self.simulate_background_activity()
        print(
            f"Graph generation complete. Total nodes: {self.num_of_nodes()}, Total edges: {self.num_of_edges()}")
        return self.graph

    def export_to_csv(self, nodes_filepath="nodes.csv", edges_filepath="edges.csv"):
        print(f"Exporting graph to CSV: {nodes_filepath}, {edges_filepath}")

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
