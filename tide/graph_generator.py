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
from .patterns.background.manager import BackgroundPatternManager
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
        # CRITICAL: Reset all random seeds IMMEDIATELY before anything else
        # This ensures determinism even when running multiple times in the same process
        from .utils.random_instance import reset_random_seed
        seed = params.get("random_seed")
        if seed is not None:
            reset_random_seed(seed)

        self.params = params
        self.random_seed: Optional[int] = seed

        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.all_nodes: Dict[NodeType, List[str]] = {nt: [] for nt in NodeType}

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
        self.background_pattern_manager = BackgroundPatternManager(
            self, self.params)
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
            "is_fraudulent": False,
            **common_attrs
        }
        if creation_date is not None:
            node_attributes_data["creation_date"] = creation_date

        final_node_attrs = asdict(NodeAttributes(**node_attributes_data))

        self.graph.add_node(node_id, **final_node_attrs,
                            **specific_attrs, **kwargs)
        self.all_nodes[node_type].append(node_id)

        # Add entities to legit cluster by default (skip accounts/institutions)
        if node_type in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
            if "legit" not in self.entity_clusters:
                self.entity_clusters["legit"] = []
            self.entity_clusters["legit"].append(node_id)

        return node_id

    def _add_edge(self, src_id: str, dest_id: str, attributes: EdgeAttributes):
        self.graph.add_edge(src_id, dest_id, **asdict(attributes))

    def initialize_entities(self):
        """Initialize all entities using the helper function."""
        initialize_entities(self)

    def build_entity_clusters(self):
        """Build entity clusters using the helper function."""
        from .utils.clustering import precompute_cluster_accounts

        # Keep the existing legit cluster that was built during entity creation
        existing_legit = self.entity_clusters.get("legit", [])
        self.entity_clusters = build_entity_clusters(self)
        # Restore the legit cluster
        self.entity_clusters["legit"] = existing_legit

        # Pre-compute account clusters for efficient pattern generation
        self.account_clusters = precompute_cluster_accounts(
            self, self.entity_clusters)

    def inject_aml_patterns(self):
        """Inject AML patterns: each pattern builds its own transactions using the
        pre-built clusters and we simply merge the resulting edges into the main graph."""

        pattern_frequency_config = self.params.get("pattern_frequency", {})
        is_random = pattern_frequency_config.get("random", False)

        if not self.pattern_manager.patterns:
            logger.warning("No AML patterns are registered in PatternManager.")
            return

        all_nodes_list = sorted(list(self.graph.nodes))
        pattern_tasks = []
        task_index_to_pattern: Dict[int, Any] = {}

        if is_random:
            # Random mode: randomly select <num_illicit_patterns> patterns
            logger.info(
                f"Injecting {self.num_aml_patterns} AML patterns randomly...")
            available_patterns = list(self.pattern_manager.patterns.values())

            for i in range(self.num_aml_patterns):
                pattern_instance = self.random_instance.choice(
                    available_patterns)

                task = (pattern_instance.inject_pattern, (all_nodes_list,), i)
                pattern_tasks.append(task)
                task_index_to_pattern[i] = pattern_instance
        else:
            logger.info("Injecting AML patterns with specific counts...")
            config_to_pattern_mapping = self.pattern_manager.patterns
            task_index = 0

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
                        for _ in range(count):
                            task = (pattern_instance.inject_pattern,
                                    (all_nodes_list,), task_index)
                            pattern_tasks.append(task)
                            task_index_to_pattern[task_index] = pattern_instance
                            task_index += 1
                    else:
                        logger.warning(
                            f"Pattern for config key '{config_key}' not found. Available patterns: {list(config_to_pattern_mapping.keys())}")

        if not pattern_tasks:
            logger.warning(
                "No patterns were injected. Check your pattern_frequency configuration.")
            return

        pattern_results = run_patterns_in_parallel(pattern_tasks)

        # Process results
        total_edges_added = 0
        fraudulent_cluster_set = set(
            self.entity_clusters.get("fraudulent", []))
        legit_cluster_set = set(self.entity_clusters.get("legit", []))

        for i, edges in enumerate(pattern_results):
            if not edges:
                continue

            # Mark nodes as fraudulent and update clusters
            fraudulent_nodes = {src for src, _, _ in edges}.union(
                {dest for _, dest, _ in edges})
            for node_id in fraudulent_nodes:
                if self.graph.has_node(node_id):
                    self.graph.nodes[node_id]['is_fraudulent'] = True
                    fraudulent_cluster_set.add(node_id)
                    # Remove from legit cluster if it was there
                    legit_cluster_set.discard(node_id)

            # Analyze and track the injected pattern
            pattern_instance = task_index_to_pattern[i]
            pattern_data = self._analyze_pattern_edges(
                pattern_instance.pattern_name, edges, pattern_instance)
            self.injected_patterns.append(pattern_data)

            # Add edges to the graph
            for src, dest, attrs in edges:
                self._add_edge(src, dest, attrs)
            total_edges_added += len(edges)

        # Update both fraudulent and legit clusters in entity_clusters
        self.entity_clusters["fraudulent"] = sorted(
            list(fraudulent_cluster_set))
        self.entity_clusters["legit"] = sorted(
            list(legit_cluster_set))

        logger.info(
            f"Done injecting AML patterns. Added {total_edges_added} fraudulent transaction edges.")
        logger.info(
            f"Tracked {len(self.injected_patterns)} actual pattern instances.")

        return total_edges_added

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

        if timestamps:
            transactions.sort(key=lambda x: x['timestamp'])
            timestamps.sort()

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

        elif pattern_name == 'UTurnTransactions':
            if transactions:
                source_account = transactions[0]['src']
                return_account = transactions[-1]['dest']

                # Collect all unique entities in the path
                path_entities = set()
                for tx in transactions:
                    path_entities.add(tx['src'])
                    path_entities.add(tx['dest'])

                intermediaries = list(
                    path_entities - {source_account, return_account})

                pattern_data.update({
                    'source_account': source_account,
                    'return_account': return_account,
                    'intermediary_accounts': intermediaries,
                    'num_intermediaries': len(intermediaries)
                })

        return pattern_data

    def expected_background_transactions(self) -> int:
        """Return the total number of background transactions that should be generated
        based on the configured `transaction_rates.per_account_per_day`.  The value
        is computed after entity creation so the number of accounts is known.
        """
        days = max(1, (self.time_span["end_date"] -
                   self.time_span["start_date"]).days)
        total_accounts = len(self.all_nodes.get(NodeType.ACCOUNT, []))
        return int(self.background_tx_rate_per_account_per_day * total_accounts * days)

    def simulate_background_activity(self, target_transaction_count: Optional[int] = None):
        """Generate realistic baseline transactions using the dedicated background patterns.

        Prior to running each pattern we allocate a transaction *budget* derived
        from the global `transaction_rates.per_account_per_day` so that the sum
        of all background patterns approximately matches the desired average
        rate.  Each pattern can still use its own logic but must not exceed the
        budget it receives via `set_tx_budget` (if it implements the method).
        """

        try:
            # ------------------------------------------------------------------
            # 1) Compute global budget based on configured rate and entity count
            # ------------------------------------------------------------------
            if target_transaction_count is not None:
                total_budget = target_transaction_count
                logger.info(
                    f"Using target transaction count for background activity: {total_budget:,}")
            else:
                total_budget = self.expected_background_transactions()
                logger.info(
                    f"Using rate-based count for background activity: {total_budget:,}")

            # ------------------------------------------------------------------
            # 2) Determine weights for each pattern from the configuration.
            #    Fallback: equal weight if not specified or sum == 0
            # ------------------------------------------------------------------
            bg_cfg = self.params.get("backgroundPatterns", {})
            pattern_weights = {}
            sum_weights = 0.0
            for pattern_name in self.background_pattern_manager.patterns.keys():
                # Config keys may be lowerCamelCase; fall back to exact key
                cfg_key = pattern_name[0].lower() + pattern_name[1:]
                weight = bg_cfg.get(cfg_key, {}).get("weight",
                                                     bg_cfg.get(pattern_name, {}).get("weight", 0.0))
                pattern_weights[pattern_name] = weight
                sum_weights += weight

            # If no weights defined, assign equal weight
            if sum_weights == 0.0:
                equal_w = 1.0 / len(pattern_weights)
                for k in pattern_weights:
                    pattern_weights[k] = equal_w
                sum_weights = 1.0

            # ------------------------------------------------------------------
            # 3) Allocate budget and inform patterns (if supported)
            # ------------------------------------------------------------------
            # Sort patterns by name to ensure deterministic execution order
            background_patterns = sorted(self.background_pattern_manager.patterns.items())

            for pattern_name, pattern_instance in background_patterns:
                weight = pattern_weights.get(pattern_name, 0.0)
                pattern_budget = int(total_budget * (weight / sum_weights))
                if hasattr(pattern_instance, "set_tx_budget"):
                    pattern_instance.set_tx_budget(pattern_budget)

            # ------------------------------------------------------------------
            # 4) Build function tasks for parallel execution
            # ------------------------------------------------------------------
            background_tasks = []
            task_index = 0

            for pattern_name, pattern_instance in background_patterns:
                task = (pattern_instance.inject_pattern_generator,
                        ([],), task_index)
                background_tasks.append(task)
                task_index += 1

            if not background_tasks:
                logger.warning("No background patterns available.")
                return

            background_results = run_patterns_in_parallel(background_tasks)

            # ------------------------------------------------------------------
            # 5) Merge results back into the master graph
            # ------------------------------------------------------------------
            total_bg_edges_added = 0
            for bg_edges_generator in background_results:
                if bg_edges_generator is None:
                    continue

                chunk_size = 10000
                chunk = []
                count = 0
                for src, dest, attrs in bg_edges_generator:
                    chunk.append(
                        (src, dest, (attrs.__dict__ if hasattr(attrs, "__dict__") else attrs)))
                    if len(chunk) >= chunk_size:
                        self.graph.add_edges_from(chunk)
                        chunk = []
                    count += 1

                if chunk:
                    self.graph.add_edges_from(chunk)

                total_bg_edges_added += count

        except Exception as e:
            logger.error(f"Failed to inject background activity: {e}")

    def generate_graph(self):
        logger.info("Starting graph generation...")
        self.initialize_entities()
        self.build_entity_clusters()
        fraud_edge_count = self.inject_aml_patterns() or 0

        # Determine background transaction cap based on target_fraud_ratio
        target_fraud_ratio = self.params.get("target_fraud_ratio")
        if target_fraud_ratio and target_fraud_ratio > 0 and fraud_edge_count > 0:
            # max_background = fraud / ratio - fraud
            # e.g., 2650 fraud, 2% ratio -> 2650/0.02 - 2650 = 129,850 background
            max_background = int(fraud_edge_count /
                                 target_fraud_ratio) - fraud_edge_count
            max_background = max(0, max_background)

            # Check if the target is achievable given graph capacity
            max_possible_background = self.expected_background_transactions()
            if max_background > max_possible_background:
                achievable_ratio = fraud_edge_count / \
                    (fraud_edge_count + max_possible_background)
                logger.warning(
                    f"Target fraud ratio {target_fraud_ratio:.4%} requires {max_background:,} background transactions, "
                    f"but graph capacity is only {max_possible_background:,}. "
                    f"Achievable ratio: {achievable_ratio:.4%} ({achievable_ratio/target_fraud_ratio:.1f}x target). "
                    f"Consider: reducing num_illicit_patterns, disabling layering, or increasing graph_scale."
                )

            logger.info(
                f"Target fraud ratio: {target_fraud_ratio:.4%} -> capping background to {max_background:,} transactions "
                f"(fraud transactions: {fraud_edge_count:,})"
            )
            self.simulate_background_activity(
                target_transaction_count=max_background)
        else:
            # No ratio cap; use raw rate-based count
            self.simulate_background_activity()

        # Calculate and report actual fraud ratio
        total_edges = self.num_of_edges()
        fraud_edges = sum(1 for _, _, attrs in self.graph.edges(data=True)
                          if attrs.get('is_fraudulent', False))
        actual_fraud_ratio = fraud_edges / total_edges if total_edges > 0 else 0

        logger.info(
            f"Graph generation complete. Total nodes: {self.num_of_nodes()}, Total edges: {self.num_of_edges()}")
        logger.info(
            f"Fraud statistics: {fraud_edges:,} fraudulent / {total_edges:,} total = {actual_fraud_ratio:.4%} fraud ratio")

        if target_fraud_ratio and target_fraud_ratio > 0:
            ratio_diff = abs(actual_fraud_ratio -
                             target_fraud_ratio) / target_fraud_ratio
            if ratio_diff > 0.1:  # More than 10% deviation
                logger.warning(
                    f"Actual fraud ratio ({actual_fraud_ratio:.4%}) deviates significantly from target ({target_fraud_ratio:.4%})")

        return self.graph
