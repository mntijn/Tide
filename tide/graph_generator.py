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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphGenerator:
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        # ------------------------------------------------------------------
        #  Reproducibility controls
        # ------------------------------------------------------------------
        self.random_seed: Optional[int] = params.get("random_seed")

        # ------------------------------------------------------------------
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
        logger.info("Starting entity generation...")

        # Reset random seeds if defined for reproducibility
        if self.random_seed:
            logger.info(
                f"Deterministic mode enabled with seed {self.random_seed}")
            self.random_instance.seed(self.random_seed)
            # Also reset global random module
            import random
            random.seed(self.random_seed)
            # Reset Faker instance
            reset_faker_seed()

        sim_start_date = self.time_span["start_date"]

        # Generate entity data sequentially
        logger.info("Generating institutions...")
        institutions_data = self.institution.generate_data()

        logger.info("Generating individuals...")
        individuals_data = self.individual.generate_data()

        # Create accounts for Institutions
        institution_countries = {}
        if institutions_data:
            for common_attrs, specific_attrs in institutions_data:
                try:
                    institution_id = self._add_node(NodeType.INSTITUTION, common_attrs,
                                                    specific_attrs, creation_date=None)
                    institution_countries[institution_id] = common_attrs["country_code"]
                except Exception as e:
                    logger.error(f"Error creating institution node: {str(e)}")

        all_institution_ids = self.all_nodes.get(NodeType.INSTITUTION, [])

        self.account = Account(
            self.params, all_institution_ids, institution_countries)

        # Create accounts for Individuals and create owned businesses
        num_owned_businesses_created = 0
        if individuals_data:
            for ind_common_attrs, ind_specific_attrs in individuals_data:
                try:
                    # Add the individual node
                    ind_id = self._add_node(
                        NodeType.INDIVIDUAL,
                        ind_common_attrs,
                        ind_specific_attrs,
                    )

                    # Determine if this individual's occupation suggests business ownership
                    occupation_str: str = ind_specific_attrs.get(
                        "occupation", "")
                    suggested_category = map_occupation_to_business_category(
                        occupation_str)

                    # Create business if occupation suggests one, or randomly
                    should_create_business = (suggested_category is not None or
                                              self.random_instance.random() < self.random_business_probability)

                    if should_create_business:
                        # Use suggested category or random one
                        business_category = suggested_category or get_random_business_category()

                        business_data_tuple = self.business.generate_age_consistent_business_for_individual(
                            individual_age_group=ind_specific_attrs["age_group"],
                            sim_start_date=sim_start_date,
                            business_category_override=business_category,
                            owner_occupation=occupation_str,
                            owner_risk_score=ind_common_attrs.get(
                                "risk_score", 0.0),
                            owner_country=ind_common_attrs.get("country_code"),
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

            individual_ids = self.all_nodes.get(NodeType.INDIVIDUAL, [])
            if not individual_ids:
                logger.warning(
                    "No individuals available to own extra businesses. Skipping override.")
            else:
                for i in range(additional_needed):
                    try:
                        owner_id = self.random_instance.choice(individual_ids)
                        owner_data = self.graph.nodes[owner_id]

                        owner_age_group = owner_data.get("age_group")

                        bus_tuple = self.business.generate_age_consistent_business_for_individual(
                            individual_age_group=owner_age_group,
                            sim_start_date=sim_start_date,
                            owner_occupation=owner_data.get("occupation", ""),
                            owner_risk_score=owner_data.get("risk_score", 0.0),
                            owner_country=owner_data.get("country_code"),
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

        # Generate accounts sequentially
        logger.info("Generating accounts...")
        individual_ids = self.all_nodes.get(NodeType.INDIVIDUAL, [])
        business_ids = self.all_nodes.get(NodeType.BUSINESS, [])

        # Process all individuals
        for ind_id in individual_ids:
            process_individual(self, ind_id, None, sim_start_date)

        # Process all businesses
        for bus_id in business_ids:
            process_business(self, bus_id, None, sim_start_date)

        logger.info("Account generation completed")

        # Create global cash system account
        if self.cash_account_id is None:
            try:
                self.cash_account_id = self._add_node(
                    NodeType.ACCOUNT,
                    {"country_code": "CASH"},
                    {"start_balance": 0.0, "current_balance": 0.0, "currency": "EUR"},
                    creation_date=self.time_span["start_date"],
                )
                logger.info(
                    f"Created global cash system account: {self.cash_account_id}")
            except Exception as e:
                logger.error(
                    f"Error creating global cash system account: {str(e)}")

        logger.info("Entity and account generation complete")

    def build_entity_clusters(self):
        """Builds entity clusters for pattern targeting:

        Pattern Source Selection:
        1. super_high_risk/high_risk_score: Entities with 3+ risk factors
        2. offshore_candidates: Financial sophistication for international operations
        3. structuring_candidates: Usually executors, not sources

        Other Roles:
        - intermediaries: Money mules/fronts (often unwitting participants)
        - Basic single-factor clusters: Country, business category, age, occupation
        """
        logger.info("Building entity clusters...")
        min_risk_score = self.fraud_selection_config.get(
            "min_risk_score_for_fraud_consideration", 0.30)

        # Build clusters using a more sophisticated approach that recognizes overlapping risk factors
        clusters: Dict[str, List[str]] = {
            # Basic single-factor clusters
            "high_risk_countries": [],
            "high_risk_business_categories": [],
            "high_risk_age_groups": [],
            "high_risk_occupations": [],
            "high_risk_score": [],

            # Composite clusters for entities with multiple risk factors
            "super_high_risk": [],  # Multiple high-risk factors
            # Potential intermediaries
            "intermediaries": [],
            "offshore_candidates": [],  # Likely to have offshore connections
            "structuring_candidates": [],  # Likely to engage in structuring
            "fraudulent": [],
        }

        # Single pass through all nodes to build all clusters
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get("node_type")
            country = data.get("country_code")
            risk_score = data.get("risk_score", 0.0)

            # Handle None risk_score values explicitly
            if risk_score is None:
                risk_score = 0.0

            # Skip accounts for most clusters (focus on individuals/businesses)
            if node_type == NodeType.ACCOUNT:
                continue

            risk_factors = []  # Track all risk factors for this entity

            # Check individual risk factors
            if country in HIGH_RISK_COUNTRIES:
                clusters["high_risk_countries"].append(node_id)
                risk_factors.append("high_risk_country")

            if node_type == NodeType.BUSINESS and data.get("is_high_risk_category", False):
                clusters["high_risk_business_categories"].append(node_id)
                risk_factors.append("high_risk_business")

            age_group = data.get("age_group")
            if age_group and str(age_group).upper() in HIGH_RISK_AGE_GROUPS:
                clusters["high_risk_age_groups"].append(node_id)
                risk_factors.append("high_risk_age")

            occupation = data.get("occupation")
            if occupation and occupation in HIGH_RISK_OCCUPATIONS:
                clusters["high_risk_occupations"].append(node_id)
                risk_factors.append("high_risk_occupation")

            if risk_score >= min_risk_score:
                clusters["high_risk_score"].append(node_id)
                risk_factors.append("high_risk_score")

            # Build composite clusters based on multiple risk factors

            # Super high risk: entities with 3+ risk factors
            if len(risk_factors) >= 3:
                clusters["super_high_risk"].append(node_id)

            # Intermediaries: comprehensive criteria for potential money mules/intermediaries
            is_intermediary = False

            # Young adults (18-24) are often recruited as intermediaries
            if age_group and str(age_group).upper() == "EIGHTEEN_TO_TWENTY_FOUR":
                is_intermediary = True

            # Elderly (65+) can be vulnerable to being used as intermediaries
            elif age_group and str(age_group).upper() == "SIXTY_FIVE_PLUS":
                is_intermediary = True

            # High-risk occupations with access to financial systems
            elif occupation and occupation in HIGH_RISK_OCCUPATIONS:
                is_intermediary = True

            # Individuals in high-risk countries (easier to use as intermediaries)
            elif node_type == NodeType.INDIVIDUAL and country in HIGH_RISK_COUNTRIES:
                is_intermediary = True

            # Small businesses in high-risk categories (often used as fronts)
            elif (node_type == NodeType.BUSINESS and
                  data.get("is_high_risk_category", False) and
                  data.get("number_of_employees", 0) <= 10):
                is_intermediary = True

            if is_intermediary:
                clusters["intermediaries"].append(node_id)

            # Offshore candidates: entities likely to have or use offshore accounts
            is_offshore_candidate = False

            # High-paid occupations often have offshore accounts
            if occupation and occupation in ["Banker", "Investment banker", "Financial trader",
                                             "Lawyer", "Chartered accountant", "IT consultant"]:
                is_offshore_candidate = True

            # Businesses in high-risk categories often use offshore structures
            elif (node_type == NodeType.BUSINESS and
                  data.get("business_category") in ["Private Banking", "Investment Banking",
                                                    "Trust Services", "Currency Exchange"]):
                is_offshore_candidate = True

            # Entities from high-risk countries
            elif country in HIGH_RISK_COUNTRIES:
                is_offshore_candidate = True

            # High overall risk score
            elif risk_score >= 0.7:
                is_offshore_candidate = True

            if is_offshore_candidate:
                clusters["offshore_candidates"].append(node_id)

            # Structuring candidates: entities likely to engage in transaction structuring
            is_structuring_candidate = False

            # Cash-intensive businesses
            cash_businesses = ["Casinos", "Currency Exchange", "Check Cashing Services",
                               "Convenience Stores", "Gas Stations", "Bars and Nightclubs",
                               "Pawn Shops", "Laundromats"]
            if (node_type == NodeType.BUSINESS and
                    data.get("business_category") in cash_businesses):
                is_structuring_candidate = True

            # Individuals with financial backgrounds who know the rules
            elif (node_type == NodeType.INDIVIDUAL and occupation and
                  any(keyword in occupation.lower() for keyword in
                      ["bank", "financial", "accountant", "trader"])):
                is_structuring_candidate = True

            # Multiple risk factors (sophisticated actors)
            elif len(risk_factors) >= 2:
                is_structuring_candidate = True

            if is_structuring_candidate:
                clusters["structuring_candidates"].append(node_id)

        # Store the clusters
        self.entity_clusters = clusters

        # Log cluster sizes for debugging
        cluster_summary = [(k, len(v)) for k, v in clusters.items() if v]
        logger.info(f"Built entity clusters: {cluster_summary}")

        # Log overlap statistics for insight
        total_entities = len([n for n, d in self.graph.nodes(data=True)
                              if d.get("node_type") in [NodeType.INDIVIDUAL, NodeType.BUSINESS]])
        super_high_risk_count = len(clusters["super_high_risk"])
        if total_entities > 0:
            logger.info(f"Super high-risk entities: {super_high_risk_count}/{total_entities} "
                        f"({super_high_risk_count/total_entities*100:.1f}%)")

    def inject_aml_patterns(self):
        """Inject AML patterns: each pattern builds its own transactions using the
        pre-built clusters and we simply merge the resulting edges into the main graph."""

        pattern_frequency_config = self.params.get("pattern_frequency", {})
        is_random = pattern_frequency_config.get("random", False)

        if not self.pattern_manager.patterns:
            logger.warning("No AML patterns are registered in PatternManager.")
            return

        total_edges_added = 0

        if is_random:
            # Random mode: randomly select num_illicit_patterns patterns
            logger.info(
                f"Injecting {self.num_aml_patterns} AML patterns randomly...")
            available_patterns = list(self.pattern_manager.patterns.values())

            for i in range(self.num_aml_patterns):
                pattern_instance = self.random_instance.choice(
                    available_patterns)
                logger.info(
                    f"[Pattern] {i+1}/{self.num_aml_patterns}: {pattern_instance.pattern_name}")

                new_edges = pattern_instance.inject_pattern(
                    list(self.graph.nodes))
                for src, dest, attrs in new_edges:
                    self._add_edge(src, dest, attrs)
                total_edges_added += len(new_edges)
        else:
            # Fixed mode: use specific counts for each pattern type
            logger.info("Injecting AML patterns with specific counts...")

            # Map config keys to pattern names (handle variations in naming)
            config_to_pattern_mapping = {}
            for pattern_name, pattern_instance in self.pattern_manager.patterns.items():
                # Create multiple possible config keys for each pattern
                normalized_name = pattern_name.lower().replace("_", "")
                config_to_pattern_mapping[normalized_name] = pattern_instance

                # Also try without common suffixes
                if normalized_name.endswith("_pattern"):
                    config_to_pattern_mapping[normalized_name[:-8]
                                              ] = pattern_instance

            total_patterns_injected = 0
            # Only sort config keys in deterministic mode (when seed is defined)
            config_keys = (sorted(pattern_frequency_config.keys()) if self.random_seed
                           else pattern_frequency_config.keys())

            for config_key in config_keys:
                count = pattern_frequency_config[config_key]
                # Skip non-pattern config keys
                if config_key in ["random", "num_illicit_patterns"]:
                    continue

                if isinstance(count, int) and count > 0:
                    # Normalize the config key
                    normalized_config_key = config_key.lower().replace(
                        "_", "")

                    pattern_instance = config_to_pattern_mapping.get(
                        normalized_config_key)
                    if pattern_instance:
                        logger.info(
                            f"Injecting {count} instances of {pattern_instance.pattern_name}")
                        for i in range(count):
                            new_edges = pattern_instance.inject_pattern(
                                list(self.graph.nodes))
                            for src, dest, attrs in new_edges:
                                self._add_edge(src, dest, attrs)
                            total_edges_added += len(new_edges)
                            total_patterns_injected += 1
                    else:
                        logger.warning(
                            f"Pattern for config key '{config_key}' not found. Available patterns: {list(config_to_pattern_mapping.keys())}")

            if total_patterns_injected == 0:
                logger.warning(
                    "No patterns were injected. Check your pattern_frequency configuration.")

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
