import networkx as nx
import random
import datetime
import csv
import logging
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple
from faker import Faker
import threading
from queue import Queue, Empty
import multiprocessing

from .datastructures.enums import (
    NodeType, EdgeType, TransactionType, AccountCategory,
    AgeGroup, Gender
)
from .datastructures.attributes import (
    NodeAttributes, AccountAttributes, EdgeAttributes,
    TransactionAttributes, OwnershipAttributes,
    IndividualAttributes, BusinessAttributes, InstitutionAttributes
)
from .utils.constants import (
    COUNTRY_CODES, COUNTRY_TO_CURRENCY, HIGH_RISK_BUSINESS_CATEGORIES
)
from .utils.business import map_occupation_to_business_category
from .entities import Individual, Business, Institution, Account
from .patterns.manager import PatternManager
from .utils.accounts import batchify, process_individual_batch, process_business_batch
from .utils.threading import run_in_threads

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

        # Initialize entity instances
        self.institution = Institution(self.params)
        self.individual = Individual(self.params)
        self.business = Business(self.params)
        self.account: Optional[Account] = None
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
            logger.warning(
                f"Number of AML patterns ({self.num_aml_patterns}) exceeds total entities ({total_entities})")

        if self.graph_scale.get("institutions", 0) == 0:
            logger.warning(
                "No institutions configured. Accounts cannot be created.")

        min_risk_threshold = self.fraud_selection_config.get(
            "min_risk_score_for_fraud_consideration", 0.20)
        if min_risk_threshold > 1.0 or min_risk_threshold < 0.0:
            logger.warning(
                f"min_risk_score_for_fraud_consideration ({min_risk_threshold}) should be between 0.0 and 1.0")

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
        data = self.institution.generate_data()
        self.results_queue.put(('institutions', data))

    def _task_generate_individuals(self):
        """Gen individual data in a thread."""
        data = self.individual.generate_data()
        self.results_queue.put(('individuals', data))

    def initialize_entities(self):
        logger.info("Starting entity generation...")
        sim_start_date = self.time_span["start_date"]

        # Use run_in_threads for entity data generation in parallel
        tasks = [
            (self._task_generate_institutions, ()),
            (self._task_generate_individuals, ())
        ]
        run_in_threads(tasks)

        thread_results = {}
        while not self.results_queue.empty():
            try:
                key, value = self.results_queue.get_nowait()
                thread_results[key] = value
            except Empty:
                break

        institutions_data = thread_results.get('institutions', [])
        individuals_data = thread_results.get('individuals', [])

        # Create accounts for Institutions
        institution_countries = {}
        if institutions_data:
            for common_attrs, specific_attrs in institutions_data:
                try:
                    institution_id = self._add_node(NodeType.INSTITUTION, common_attrs,
                                                    specific_attrs, creation_date=None)
                    institution_countries[institution_id] = common_attrs["address"]["country"]
                except Exception as e:
                    logger.error(f"Error creating institution node: {str(e)}")

        all_institution_ids = self.all_nodes.get(NodeType.INSTITUTION, [])
        logger.info(f"Created {len(all_institution_ids)} institutions")

        self.account = Account(
            self.params, all_institution_ids, institution_countries)

        # Create accounts for Individuals and create owned businesses
        num_owned_businesses_created = 0
        if individuals_data:
            for ind_creation_date_dt, ind_common_attrs, ind_specific_attrs in individuals_data:
                try:
                    # Add the individual node
                    ind_id = self._add_node(
                        NodeType.INDIVIDUAL,
                        ind_common_attrs,
                        ind_specific_attrs,
                        creation_date=ind_creation_date_dt,
                    )

                    # Determine if this individual's occupation suggests business ownership
                    occupation_str: str = ind_specific_attrs.get(
                        "occupation", "")
                    suggested_category = map_occupation_to_business_category(
                        occupation_str)

                    if suggested_category:
                        # Create a business that aligns with the occupation
                        business_data_tuple = self.business.generate_age_consistent_business_for_individual(
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
                except Exception as e:
                    logger.error(f"Error processing individual data: {str(e)}")

        # Create additional businesses if needed
        target_business_count = self.graph_scale.get("businesses", 0)
        if target_business_count > num_owned_businesses_created:
            additional_needed = target_business_count - num_owned_businesses_created
            logger.info(
                f"Creating {additional_needed} additional businesses to reach target of {target_business_count}")

            individual_ids = self.all_nodes.get(NodeType.INDIVIDUAL, [])
            if not individual_ids:
                logger.warning(
                    "No individuals available to own extra businesses. Skipping override.")
            else:
                for i in range(additional_needed):
                    try:
                        owner_id = random.choice(individual_ids)
                        owner_data = self.graph.nodes[owner_id]

                        owner_age_group = owner_data.get("age_group")
                        owner_creation_date = owner_data.get("creation_date")

                        bus_tuple = self.business.generate_age_consistent_business_for_individual(
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
                    except Exception as e:
                        logger.error(
                            f"Error creating additional business: {str(e)}")

        logger.info(f"Created {num_owned_businesses_created} total businesses")

        # Generate accounts in parallel
        num_threads = multiprocessing.cpu_count()
        logger.info(
            f"Generating accounts in parallel using {num_threads} threads...")

        lock = threading.Lock()
        individual_ids = self.all_nodes.get(NodeType.INDIVIDUAL, [])
        business_ids = self.all_nodes.get(NodeType.BUSINESS, [])

        individual_batches = batchify(individual_ids, num_threads)
        business_batches = batchify(business_ids, num_threads)

        account_tasks = []
        for batch in individual_batches:
            account_tasks.append(
                (process_individual_batch, (self, batch, lock, sim_start_date)))
        for batch in business_batches:
            account_tasks.append(
                (process_business_batch, (self, batch, lock, sim_start_date)))

        run_in_threads(account_tasks)
        logger.info("Account generation tasks completed")

        # Create global cash system account
        if self.cash_account_id is None:
            try:
                self.cash_account_id = self._add_node(
                    NodeType.ACCOUNT,
                    {"address": {"country": "CASH"}},
                    {"start_balance": 0.0, "current_balance": 0.0, "currency": "EUR"},
                    creation_date=self.time_span["start_date"],
                )
                logger.info(
                    f"Created global cash system account: {self.cash_account_id}")
            except Exception as e:
                logger.error(
                    f"Error creating global cash system account: {str(e)}")

        logger.info("Entity and account generation complete")

    def select_fraudulent_entities(self):
        logger.info("Selecting fraudulent entities based on risk scores...")
        min_risk_score = self.fraud_selection_config.get(
            "min_risk_score_for_fraud_consideration", 0.30)
        base_prob = self.fraud_selection_config.get(
            "base_fraud_probability_if_considered", 0.20)

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

        logger.info(
            f"Considered {fraud_candidates} entities for fraud based on risk_score >= {min_risk_score}.")
        logger.info(f"Selected {selected_fraudulent} entities as fraudulent.")
        for nt, items in self.fraudulent_entities_map.items():
            if items:
                logger.info(f"  - {len(items)} {nt.value}(s)")

    def select_entities_for_pattern(self, pattern_name: str, num_entities_required: int) -> List[str]:
        """
        Selects entities for a given AML pattern.
        Prioritizes fraudulent entities (Individuals and Businesses).
        If not enough fraudulent entities are available, non-fraudulent ones are considered.
        The number of entities to select is determined by num_entities_required for each pattern.
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
            logger.warning(f"No entities available for pattern {pattern_name}")
            return []

        # Shuffle to ensure randomness if we have more than needed, then select
        random.shuffle(candidate_entities)
        return candidate_entities[:num_entities_required]

    def inject_aml_patterns(self):
        """Inject combined temporal and structural AML patterns using PatternManager."""
        logger.info(f"Injecting up to {self.num_aml_patterns} AML patterns...")
        if not self.pattern_manager:
            logger.error("PatternManager not initialized.")
            return

        fraudulent_edges = self.pattern_manager.inject_patterns()
        if fraudulent_edges:
            logger.info(
                f"PatternManager injected {len(fraudulent_edges)} new fraudulent edges.")
        else:
            logger.warning(
                "PatternManager did not inject any new fraudulent edges.")

        return

    def simulate_background_activity(self):
        logger.info("Simulating background financial activity...")
        pass

    def generate_graph(self):
        logger.info("Starting graph generation...")
        self.initialize_entities()
        self.select_fraudulent_entities()
        self.inject_aml_patterns()
        self.simulate_background_activity()
        logger.info(
            f"Graph generation complete. Total nodes: {self.num_of_nodes()}, Total edges: {self.num_of_edges()}")
        return self.graph
