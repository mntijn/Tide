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

        # Track transaction history for each entity
        self.entity_transaction_history: Dict[str, datetime.datetime] = {}

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

        # PATTERN TRACKING: Track what patterns are actually injected
        self.injected_patterns = []

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

            # TRACK PATTERNS: Record what patterns were actually created
            pattern_index = 0
            for config_key in config_keys:
                count = pattern_frequency_config[config_key]
                if config_key in ["random", "num_illicit_patterns"]:
                    continue

                if isinstance(count, int) and count > 0:
                    pattern_instance = config_to_pattern_mapping.get(
                        config_key)
                    if pattern_instance:
                        for i in range(count):
                            if pattern_index < len(pattern_results):
                                edges = pattern_results[pattern_index]
                                if edges:  # Only track if pattern actually generated edges
                                    pattern_data = self._analyze_pattern_edges(
                                        pattern_instance.pattern_name, edges, pattern_instance)
                                    self.injected_patterns.append(pattern_data)
                                pattern_index += 1

            for edges in pattern_results:
                for src, dest, attrs in edges:
                    self._add_edge(src, dest, attrs)
                total_edges_added += len(edges)

        logger.info(
            f"Done injecting AML patterns. Added {total_edges_added} fraudulent transaction edges.")
        logger.info(
            f"Tracked {len(self.injected_patterns)} actual pattern instances.")

    def _analyze_pattern_edges(self, pattern_name: str, edges: List, pattern_instance) -> Dict:
        """Analyze pattern edges to extract key characteristics for tracking"""
        if not edges:
            return {}

        # Extract entities involved
        entities = set()
        transactions = []
        amounts = []
        countries = set()
        timestamps = []

        for src, dest, attrs in edges:
            entities.add(src)
            entities.add(dest)

            # Convert attrs to dict if it's a dataclass
            attrs_dict = attrs.__dict__ if hasattr(
                attrs, '__dict__') else attrs

            transactions.append({
                'src': src,
                'dest': dest,
                'amount': attrs_dict.get('amount', 0),
                'timestamp': attrs_dict.get('timestamp'),
                'transaction_type': str(attrs_dict.get('transaction_type', '')),
                'currency': attrs_dict.get('currency', 'EUR')
            })

            amounts.append(float(attrs_dict.get('amount', 0)))
            if attrs_dict.get('timestamp'):
                timestamps.append(attrs_dict['timestamp'])

        # Get countries for involved entities
        for entity_id in entities:
            if self.graph.has_node(entity_id):
                country = self.graph.nodes[entity_id].get('country_code')
                if country:
                    countries.add(country)

        # Calculate pattern characteristics
        total_amount = sum(amounts)
        time_span = None
        if len(timestamps) > 1:
            time_span = (max(timestamps) - min(timestamps)
                         ).total_seconds() / 3600  # hours

        pattern_data = {
            'pattern_type': pattern_name,
            'pattern_id': f"{pattern_name}_{len(self.injected_patterns) + 1}",
            'entities': list(entities),
            'transactions': transactions,
            'num_transactions': len(transactions),
            'total_amount': total_amount,
            'avg_amount': total_amount / len(amounts) if amounts else 0,
            'countries': list(countries),
            'time_span_hours': time_span,
            'start_time': min(timestamps) if timestamps else None,
            'end_time': max(timestamps) if timestamps else None,
        }

        # Add pattern-specific analysis
        if pattern_name == 'RepeatedOverseasTransfers':
            # Find source account and destination countries
            sources = set(tx['src'] for tx in transactions)
            if sources:
                # Assume single source for this pattern
                main_source = list(sources)[0]
                dest_countries = set()
                for tx in transactions:
                    if self.graph.has_node(tx['dest']):
                        dest_country = self.graph.nodes[tx['dest']].get(
                            'country_code')
                        if dest_country:
                            dest_countries.add(dest_country)

                pattern_data.update({
                    'source_account': main_source,
                    'destination_countries': list(dest_countries),
                    'num_overseas_destinations': len(dest_countries)
                })

        elif pattern_name == 'RapidFundMovement':
            # Analyze inflows vs outflows for central account
            entity_txs = {}
            for tx in transactions:
                if tx['src'] not in entity_txs:
                    entity_txs[tx['src']] = {'inflows': 0, 'outflows': 0}
                if tx['dest'] not in entity_txs:
                    entity_txs[tx['dest']] = {'inflows': 0, 'outflows': 0}

                entity_txs[tx['dest']]['inflows'] += 1
                entity_txs[tx['src']]['outflows'] += 1

            # Find the account with both inflows and outflows (central account)
            central_account = None
            for entity, counts in entity_txs.items():
                if counts['inflows'] > 0 and counts['outflows'] > 0:
                    central_account = entity
                    break

            pattern_data.update({
                'central_account': central_account,
                'movement_speed_hours': time_span
            })

        elif pattern_name == 'FrontBusinessActivity':
            # Find business entity and analyze deposit vs transfer flow
            business_entity = None
            deposits = [
                tx for tx in transactions if 'deposit' in tx['transaction_type'].lower()]
            transfers = [
                tx for tx in transactions if 'transfer' in tx['transaction_type'].lower()]

            # Find business among entities
            for entity_id in entities:
                if self.graph.has_node(entity_id):
                    node_type = str(self.graph.nodes[entity_id].get(
                        'node_type', '')).lower()
                    if 'business' in node_type:
                        business_entity = entity_id
                        break

            pattern_data.update({
                'business_entity': business_entity,
                'num_deposits': len(deposits),
                'num_transfers': len(transfers),
                'deposit_amount': sum(tx['amount'] for tx in deposits),
                'transfer_amount': sum(tx['amount'] for tx in transfers)
            })

        return pattern_data

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
