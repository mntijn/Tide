import networkx as nx
import random
import datetime
import csv
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple
from faker import Faker
import threading
from queue import Queue, Empty

from .data_structures import (
    NodeType, EdgeType, TransactionType, AccountCategory,
    NodeAttributes, AccountAttributes, EdgeAttributes,
    TransactionAttributes, OwnershipAttributes,
    IndividualAttributes, BusinessAttributes, InstitutionAttributes
)
from .utils import COUNTRY_CODES, COUNTRY_TO_CURRENCY, map_occupation_to_business_category, HIGH_RISK_BUSINESS_CATEGORIES
from .entity_creators import (
    InstitutionCreator, IndividualCreator, BusinessCreator, AccountCreator
)
from .patterns.manager import PatternManager


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
        self.results_queue = Queue()
        self.individual_cash_accounts: Dict[str, str] = {}
        self.cash_account_id: Optional[str] = None

        self.pattern_manager = PatternManager(self, self.params)
        self.pattern_manager.entity_selector = self

        self.background_tx_rate_per_account_per_day = params.get(
            "transaction_rates", {}).get("per_account_per_day", 0.2)
        self.num_aml_patterns = params.get(
            "pattern_frequency", {}).get("num_illicit_patterns", 5)

        # Probability (0-1) that an individual with no occupation-based match will still start a high-risk business
        self.high_risk_business_probability = params.get(
            "high_risk_business_probability", 0.05
        )

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

    def _task_generate_institutions(self):
        """Gen institution data in a thread."""
        data = self.institution_creator.generate_institutions_data()
        self.results_queue.put(('institutions', data))

    def _task_generate_individuals(self):
        """Gen individual data in a thread."""
        data = self.individual_creator.generate_individuals_data()
        self.results_queue.put(('individuals', data))

    def initialize_entities(self):
        print("Starting entity generation...")
        sim_start_date = self.time_span["start_date"]

        thread_institutions = threading.Thread(
            target=self._task_generate_institutions)
        thread_individuals = threading.Thread(
            target=self._task_generate_individuals)

        thread_institutions.start()
        thread_individuals.start()

        thread_institutions.join()
        thread_individuals.join()

        thread_results = {}
        while not self.results_queue.empty():
            try:
                key, value = self.results_queue.get_nowait()
                thread_results[key] = value
            except Empty:
                break

        institutions_data = thread_results.get('institutions', [])
        individuals_data = thread_results.get('individuals', [])

        print("Initial entity data fetched. Adding to graph and creating relationships...")

        # Create accounts for Institutions
        institution_countries = {}
        if institutions_data:
            for common_attrs, specific_attrs in institutions_data:
                institution_id = self._add_node(NodeType.INSTITUTION, common_attrs,
                                                specific_attrs, creation_date=None)
                institution_countries[institution_id] = common_attrs["address"]["country"]

        all_institution_ids = self.all_nodes.get(NodeType.INSTITUTION, [])
        self.account_creator = AccountCreator(
            self.params, all_institution_ids, institution_countries)

        # Create accounts for Individuals and create owned businesses
        num_owned_businesses_created = 0
        if individuals_data:
            for ind_creation_date_dt, ind_common_attrs, ind_specific_attrs in individuals_data:
                # Add the individual node
                ind_id = self._add_node(
                    NodeType.INDIVIDUAL,
                    ind_common_attrs,
                    ind_specific_attrs,
                    creation_date=ind_creation_date_dt,
                )

                # Determine if this individual's occupation suggests business ownership
                occupation_str: str = ind_specific_attrs.get("occupation", "")
                suggested_category = map_occupation_to_business_category(
                    occupation_str)

                # Fallback: assign a random HIGH_RISK_BUSINESS_CATEGORY even if occupation provides no clear mapping.
                if suggested_category is None and random.random() < self.high_risk_business_probability:
                    suggested_category = random.choice(
                        HIGH_RISK_BUSINESS_CATEGORIES)

                if suggested_category:
                    # Create a business that aligns with the occupation
                    business_data_tuple = self.business_creator.generate_age_consistent_business_for_individual(
                        individual_age_group=ind_specific_attrs["age_group"],
                        individual_creation_date=ind_creation_date_dt,
                        sim_start_date=sim_start_date,
                        business_category_override=suggested_category,
                    )

                    bus_creation_date, bus_common_attrs, bus_specific_attrs = business_data_tuple

                    bus_id = self._add_node(
                        NodeType.BUSINESS,
                        bus_common_attrs,
                        bus_specific_attrs,
                        creation_date=bus_creation_date,
                    )

                    ownership_attrs = OwnershipAttributes(
                        ownership_start_date=bus_creation_date.date(),
                        ownership_percentage=100.0,
                    )

                    self._add_edge(ind_id, bus_id, ownership_attrs)
                    num_owned_businesses_created += 1

        # If a target business count is set in config (graph_scale.businesses),
        # ensure we meet it by creating additional businesses owned by random
        # individuals. These extra businesses do not rely on occupation matching.
        target_business_count = self.graph_scale.get("businesses", 0)
        if target_business_count > num_owned_businesses_created:
            additional_needed = target_business_count - num_owned_businesses_created
            print(
                f"Config override: Creating {additional_needed} extra business(es) to reach target of {target_business_count}."
            )

            individual_ids = self.all_nodes.get(NodeType.INDIVIDUAL, [])
            if not individual_ids:
                print(
                    "Warning: No individuals available to own extra businesses. Skipping override.")
            else:
                for _ in range(additional_needed):
                    owner_id = random.choice(individual_ids)
                    owner_data = self.graph.nodes[owner_id]

                    owner_age_group = owner_data.get("age_group")
                    owner_creation_date = owner_data.get("creation_date")

                    # Generate a business with random (potentially high-risk) category
                    bus_tuple = self.business_creator.generate_age_consistent_business_for_individual(
                        individual_age_group=owner_age_group,
                        individual_creation_date=owner_creation_date,
                        sim_start_date=sim_start_date,
                    )

                    bus_creation_date, bus_common_attrs, bus_specific_attrs = bus_tuple

                    bus_id = self._add_node(
                        NodeType.BUSINESS,
                        bus_common_attrs,
                        bus_specific_attrs,
                        creation_date=bus_creation_date,
                    )

                    self._add_edge(
                        owner_id,
                        bus_id,
                        OwnershipAttributes(
                            ownership_start_date=bus_creation_date.date(),
                            ownership_percentage=100.0,
                        ),
                    )
                    num_owned_businesses_created += 1

        print("All entity nodes (Institutions, Individuals, Businesses) created.")
        print("Proceeding to create accounts for all Individuals and Businesses...")

        # Create accounts for all entities
        if self.account_creator and all_institution_ids:
            # Accounts for Individuals
            for ind_id in self.all_nodes.get(NodeType.INDIVIDUAL, []):
                node_data = self.graph.nodes[ind_id]
                ind_creation_date = node_data.get("creation_date")
                ind_country_code = node_data.get("address", {}).get("country")

                if ind_creation_date and ind_country_code:
                    accounts_and_ownerships = self.account_creator.generate_accounts_and_ownership_data_for_entity(
                        entity_node_type=NodeType.INDIVIDUAL,
                        entity_creation_date=ind_creation_date,
                        entity_country_code=ind_country_code,
                        sim_start_date=sim_start_date
                    )
                    for acc_creation_date, acc_common, acc_specific, owner_specific in accounts_and_ownerships:
                        acc_id = self._add_node(
                            NodeType.ACCOUNT, acc_common, acc_specific, creation_date=acc_creation_date)
                        ownership_instance = OwnershipAttributes(
                            **owner_specific)
                        self._add_edge(ind_id, acc_id, ownership_instance)

                # Create a dedicated "cash" account for the individual.
                # Used for deposits and withdrawals.

                if ind_id not in self.individual_cash_accounts:
                    cash_account_common = {
                        "address": node_data.get("address"),
                        "is_fraudulent": False,
                    }
                    cash_account_specific = {
                        "start_balance": 0.0,
                        "current_balance": 0.0,
                        "institution_id": None,
                        "account_category": AccountCategory.CASH,
                        "currency": COUNTRY_TO_CURRENCY[ind_country_code],
                    }
                    cash_acc_id = self._add_node(
                        NodeType.ACCOUNT,
                        cash_account_common,
                        cash_account_specific,
                        creation_date=ind_creation_date,
                    )
                    # Link ownership
                    self._add_edge(
                        ind_id,
                        cash_acc_id,
                        OwnershipAttributes(
                            ownership_start_date=ind_creation_date.date(),
                            ownership_percentage=100.0,
                        ),
                    )
                    self.individual_cash_accounts[ind_id] = cash_acc_id

            # Accounts for Businesses
            for bus_id in self.all_nodes.get(NodeType.BUSINESS, []):
                node_data = self.graph.nodes[bus_id]
                bus_creation_date = node_data.get("creation_date")
                bus_country_code = node_data.get("address", {}).get("country")

                if bus_creation_date and bus_country_code:
                    bus_accounts_and_ownerships = self.account_creator.generate_accounts_and_ownership_data_for_entity(
                        entity_node_type=NodeType.BUSINESS,
                        entity_creation_date=bus_creation_date,
                        entity_country_code=bus_country_code,
                        sim_start_date=sim_start_date
                    )
                    for acc_creation_date, acc_common, acc_specific, owner_specific in bus_accounts_and_ownerships:
                        acc_id = self._add_node(
                            NodeType.ACCOUNT, acc_common, acc_specific, creation_date=acc_creation_date)
                        ownership_instance = OwnershipAttributes(
                            **owner_specific)
                        self._add_edge(bus_id, acc_id, ownership_instance)

        print("Entity and account generation complete.")

        # Create a single dummy account that represents the physical cash
        # system (ATMs / cash in transit).
        if self.cash_account_id is None:
            self.cash_account_id = self._add_node(
                NodeType.ACCOUNT,
                {"address": {"country": "CASH"}},
                {"start_balance": 0.0, "current_balance": 0.0, "currency": "EUR"},
                creation_date=self.time_span["start_date"],
            )
            print(
                f"Created global cash system account: {self.cash_account_id}")

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
        for nt, items in self.fraudulent_entities_map.items():
            if items:
                print(f"  - {len(items)} {nt.value}(s)")

    def select_entities_for_pattern(self, pattern_name: str, num_entities_required: int) -> List[str]:
        """
        Selects entities for a given AML pattern.
        Prioritizes fraudulent entities (Individuals and Businesses).
        If not enough fraudulent entities are available, non-fraudulent ones are considered.
        The number of entities to select is determined by num_entities_required.
        """
        candidate_entities = []

        # Prioritize fraudulent entities
        for node_type in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
            candidate_entities.extend(
                self.fraudulent_entities_map.get(node_type, []))

        # If not enough fraudulent entities, add non-fraudulent ones
        if len(candidate_entities) < num_entities_required:
            needed_more = num_entities_required - len(candidate_entities)
            non_fraudulent_pool = []
            for node_type in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
                all_of_type = self.all_nodes.get(node_type, [])
                fraudulent_of_type = set(
                    self.fraudulent_entities_map.get(node_type, []))
                non_fraudulent_pool.extend(
                    [node_id for node_id in all_of_type if node_id not in fraudulent_of_type])

            if non_fraudulent_pool:
                candidate_entities.extend(random.sample(
                    non_fraudulent_pool, min(needed_more, len(non_fraudulent_pool))))

        if not candidate_entities:
            print(f"Warning: No entities available for pattern {pattern_name}")
            return []

        # Shuffle to ensure randomness if we have more than needed, then select
        random.shuffle(candidate_entities)
        return candidate_entities[:num_entities_required]

    def inject_aml_patterns(self):
        """Inject combined temporal and structural AML patterns using PatternManager."""
        print(f"Injecting up to {self.num_aml_patterns} AML patterns...")
        if not self.pattern_manager:
            print("Error: PatternManager not initialized.")
            return

        fraudulent_edges = self.pattern_manager.inject_patterns()
        if fraudulent_edges:
            print(
                f"PatternManager injected {len(fraudulent_edges)} new fraudulent edges.")
        else:
            print("PatternManager did not inject any new fraudulent edges.")

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
                if 'edge_type' not in attrs and isinstance(attrs.get('edge_type'), EdgeType):
                    row['edge_type'] = attrs['edge_type'].value
                writer.writerow(row)
