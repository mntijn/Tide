import networkx as nx
import datetime
import csv
import logging
from dataclasses import asdict
from typing import List, Dict, Any, Optional, Tuple
from faker import Faker

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
    COUNTRY_CODES, COUNTRY_TO_CURRENCY, HIGH_RISK_BUSINESS_CATEGORIES, HIGH_RISK_COUNTRIES, HIGH_RISK_AGE_GROUPS, HIGH_RISK_OCCUPATIONS
)
from .utils.business import map_occupation_to_business_category, get_random_business_category
from .entities import Individual, Business, Institution, Account
from .patterns.manager import PatternManager
from .patterns.base import StructuralComponent, EntitySelection
from .utils.accounts import process_individual, process_business
from .utils.random_instance import random_instance
from .utils.faker_instance import reset_faker_seed
from .utils.threading import run_patterns_in_parallel
from .utils.entity_initialization import initialize_entities
from .utils.clustering import build_entity_clusters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphGenerator:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.random_seed: Optional[int] = params.get("random_seed")

        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.all_nodes: Dict[NodeType, List[str]] = {nt: [] for nt in NodeType}
        self.fraudulent_entities_map: Dict[NodeType, List[str]] = {
            nt: [] for nt in NodeType}

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
        self.individual_cash_accounts: Dict[str, str] = {}
        self.cash_account_id: Optional[str] = None
        self.random_instance = random_instance

        self.pattern_manager = PatternManager(self, self.params)
        self.entity_clusters: Dict[str, List[str]] = {}

        self.background_tx_rate_per_account_per_day = params.get(
            "transaction_rates", {}).get("per_account_per_day", 0.2)
        self.num_aml_patterns = params.get(
            "pattern_frequency", {}).get("num_illicit_patterns", 5)

        # Probability (0-1) that an individual with no occupation-based match will still start a high-risk business
        self.high_risk_business_probability = params.get(
            "high_risk_business_probability", 0.05
        )

        self.random_business_probability = params.get(
            "random_business_probability", 0.15
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

        if self.graph_scale.get("institutions_per_country", 0) == 0:
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

    def initialize_entities(self):
        """Initialize all entities using the helper function."""
        initialize_entities(self)

    def build_entity_clusters(self):
        """Build entity clusters using the helper function."""
        self.entity_clusters = build_entity_clusters(self)

    def inject_aml_patterns(self):
        """Inject AML patterns: each pattern builds its own transactions using the
        pre-built clusters and we simply merge the resulting edges into the main graph."""

        pattern_frequency_config = self.params.get("pattern_frequency", {})
        is_random = pattern_frequency_config.get("random", False)

        if not self.pattern_manager.patterns:
            logger.warning("No AML patterns are registered in PatternManager.")
            return

        total_edges_added = 0
        all_nodes_list = list(self.graph.nodes)

        if is_random:
            # Random mode: randomly select <num_illicit_patterns> patterns
            logger.info(
                f"Injecting {self.num_aml_patterns} AML patterns randomly...")
            available_patterns = list(self.pattern_manager.patterns.values())

            # Create pattern tasks for parallel execution
            pattern_tasks = []
            for i in range(self.num_aml_patterns):
                pattern_instance = self.random_instance.choice(
                    available_patterns)
                logger.info(
                    f"[Pattern] {i+1}/{self.num_aml_patterns}: {pattern_instance.pattern_name}")

                task = (pattern_instance.inject_pattern, (all_nodes_list,), i)
                pattern_tasks.append(task)

            pattern_results = run_patterns_in_parallel(pattern_tasks)

            # Merge all edges into the main graph
            for edges in pattern_results:
                for src, dest, attrs in edges:
                    self._add_edge(src, dest, attrs)
                total_edges_added += len(edges)

        else:
            # Fixed mode: use specific counts for each pattern type
            logger.info("Injecting AML patterns with specific counts...")

            config_to_pattern_mapping = self.pattern_manager.patterns

            # Collect all pattern tasks first
            pattern_tasks = []
            task_index = 0
            total_patterns_injected = 0

            config_keys = (sorted(pattern_frequency_config.keys()) if self.random_seed
                           else pattern_frequency_config.keys())

            for config_key in config_keys:
                count = pattern_frequency_config[config_key]
                if config_key in ["random", "num_illicit_patterns"]:
                    continue

                if isinstance(count, int) and count > 0:
                    pattern_instance = config_to_pattern_mapping.get(
                        config_key)
                    if pattern_instance:
                        logger.info(
                            f"Injecting {count} instances of {pattern_instance.pattern_name}")
                        for i in range(count):
                            task = (pattern_instance.inject_pattern,
                                    (all_nodes_list,), task_index)
                            pattern_tasks.append(task)
                            task_index += 1
                            total_patterns_injected += 1
                    else:
                        logger.warning(
                            f"Pattern for config key '{config_key}' not found. Available patterns: {list(config_to_pattern_mapping.keys())}")

            if total_patterns_injected == 0:
                logger.warning(
                    "No patterns were injected. Check your pattern_frequency configuration.")
                return

            pattern_results = run_patterns_in_parallel(pattern_tasks)

            for edges in pattern_results:
                for src, dest, attrs in edges:
                    self._add_edge(src, dest, attrs)
                total_edges_added += len(edges)

        logger.info(
            f"Done injecting AML patterns. Added {total_edges_added} fraudulent transaction edges.")

    def simulate_background_activity(self):
        """Generate realistic baseline transactions using the dedicated background pattern."""
        logger.info(
            "Simulating background financial activity via BackgroundActivityPattern…")

        try:
            from .patterns.background_activity import BackgroundActivityPattern

            bg_pattern = BackgroundActivityPattern(self, self.params)
            all_accounts = list(self.all_nodes.get(NodeType.ACCOUNT, []))
            bg_edges = bg_pattern.inject_pattern(all_accounts)
            for src, dest, attrs in bg_edges:
                self._add_edge(src, dest, attrs)
            logger.info(
                f"Background activity injected: {len(bg_edges)} edges.")
        except Exception as e:
            logger.error(f"Failed to inject background activity: {e}")

    def generate_graph(self):
        logger.info("Starting graph generation...")
        self.initialize_entities()
        self.build_entity_clusters()
        self.inject_aml_patterns()
        # self.simulate_background_activity()
        logger.info(
            f"Graph generation complete. Total nodes: {self.num_of_nodes()}, Total edges: {self.num_of_edges()}")
        return self.graph
