import datetime
from typing import List, Dict, Any, Tuple, Optional, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..datastructures.enums import (
    NodeType, EdgeType, TransactionType
)
from ..datastructures.attributes import (
    TransactionAttributes, NodeAttributes
)
from ..utils.currency_conversion import (
    convert_currency, generate_structured_amounts as generate_structured_amounts_util
)

from ..utils.random_instance import random_instance

try:
    from forex_python.converter import CurrencyRates
except ImportError:
    print("Warning: forex-python not available. Currency conversion will be disabled.")
    CurrencyRates = None


@dataclass
class EntitySelection:
    """Represents a selection of entities for a pattern"""
    central_entities: List[str]  # Main actors (the individual, front business)
    # Supporting entities (overseas accounts, multiple accounts)
    peripheral_entities: List[str]


@dataclass
class TransactionSequence:
    """Represents a sequence of transactions with timing"""
    transactions: List[Tuple[str, str, TransactionAttributes]
                       ]
    sequence_name: str
    start_time: datetime.datetime
    duration: datetime.timedelta


class PatternInjector:
    """Base pattern injector with common functionality"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        self.graph = graph_generator.graph
        self.graph_generator = graph_generator
        self.params = params
        self.time_span = params.get("time_span", {})

    @property
    @abstractmethod
    def num_required_entities(self) -> int:
        """Minimum number of entities required for this pattern."""
        pass

    def _create_transaction_edge(self,
                                 src_id: str,
                                 dest_id: str,
                                 timestamp: datetime.datetime,
                                 amount: float,
                                 transaction_type: TransactionType = TransactionType.TRANSFER,
                                 is_fraudulent: bool = True) -> TransactionAttributes:
        """Helper to create transaction attributes"""
        # Prefer currency of the source account; fall back to destination or EUR
        currency = "EUR"
        if self.graph.has_node(src_id):
            currency = self.graph.nodes[src_id].get("currency", currency)
        elif self.graph.has_node(dest_id):
            currency = self.graph.nodes[dest_id].get("currency", currency)

        # Compute time since previous transaction for this entity
        time_since_previous = None
        if hasattr(self.graph_generator, 'entity_transaction_history'):
            last_tx_time = self.graph_generator.entity_transaction_history.get(
                src_id)
            if last_tx_time is not None:
                time_since_previous = timestamp - last_tx_time

            # Update the history
            self.graph_generator.entity_transaction_history[src_id] = timestamp

        return TransactionAttributes(
            timestamp=timestamp,
            amount=amount,
            currency=currency,
            transaction_type=transaction_type,
            is_fraudulent=is_fraudulent,
            time_since_previous_transaction=time_since_previous
        )


class StructuralComponent(ABC):
    """Base class for structural components of patterns"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        self.graph = graph_generator.graph
        self.graph_generator = graph_generator
        self.params = params

    @property
    @abstractmethod
    def num_required_entities(self) -> int:
        """Minimum number of entities this structural component needs."""
        pass

    @abstractmethod
    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        """Select entities based on structural requirements"""
        pass

    def filter_entities_by_criteria(self, entities: List[str], criteria: Dict[str, Any]) -> List[str]:
        """Helper to filter entities based on various criteria"""
        filtered = []
        for entity_id in entities:
            attrs = self.graph.nodes[entity_id]

            # Check node type
            if criteria.get("node_type") and attrs.get("node_type") != criteria["node_type"]:
                continue

            # Check high-risk geography
            if criteria.get("high_risk_geography"):
                address = attrs.get("address", {})
                if address.get("country") not in self.params.get("high_risk_countries", []):
                    continue

            # Check business category risk
            if criteria.get("high_risk_business"):
                if not attrs.get("is_high_risk_category", False):
                    continue

            # Check if entity has multiple accounts (optimized)
            if criteria.get("multiple_accounts"):
                min_accounts = criteria.get("min_accounts", 2)
                owned_accounts = self._get_owned_accounts(entity_id)
                if len(owned_accounts) < min_accounts:
                    continue

            # Check for overseas connections
            if criteria.get("overseas_connections"):
                entity_country = attrs.get("address", {}).get("country")
                if not self._has_overseas_connections(entity_id, entity_country):
                    continue

            filtered.append(entity_id)

        return filtered

    def _get_owned_accounts(self, entity_id: str) -> List[str]:
        """Efficiently get all accounts owned by an entity (individual or business)"""
        owned_accounts = []

        # Direct accounts
        for neighbor_id in self.graph.neighbors(entity_id):
            if self.graph.nodes[neighbor_id].get("node_type") == NodeType.ACCOUNT:
                owned_accounts.append(neighbor_id)

        # If entity is an individual, also check business accounts they own
        if self.graph.nodes[entity_id].get("node_type") == NodeType.INDIVIDUAL:
            for neighbor_id in self.graph.neighbors(entity_id):
                if self.graph.nodes[neighbor_id].get("node_type") == NodeType.BUSINESS:
                    # Get accounts of owned businesses
                    for business_neighbor in self.graph.neighbors(neighbor_id):
                        if self.graph.nodes[business_neighbor].get("node_type") == NodeType.ACCOUNT:
                            owned_accounts.append(business_neighbor)

        return list(set(owned_accounts))  # Remove duplicates

    def _has_overseas_connections(self, entity_id: str, entity_country: str) -> bool:
        """Check if entity has connections to different countries"""
        if not entity_country:
            return False

        for neighbor_id in self.graph.neighbors(entity_id):
            neighbor_country = self.graph.nodes[neighbor_id].get(
                "address", {}).get("country")
            if neighbor_country and neighbor_country != entity_country:
                return True
        return False

    # ------------------------------------------------------------------
    #  Convenience for fast entity retrieval
    # ------------------------------------------------------------------
    def get_cluster(self, name: str) -> List[str]:
        """Return pre-computed entity cluster from the parent graph generator (if available)."""
        if hasattr(self.graph_generator, "entity_clusters"):
            return self.graph_generator.entity_clusters.get(name, [])
        return []

    def get_combined_clusters(self, cluster_names: List[str], deduplicate: bool = True) -> List[str]:
        """Get entities from multiple clusters combined."""
        combined = []
        for cluster_name in cluster_names:
            combined.extend(self.get_cluster(cluster_name))

        if deduplicate:
            return list(set(combined))
        return combined

    def get_high_risk_entities(self, include_super_high_risk: bool = True) -> List[str]:
        """Get all high-risk entities across different risk factors."""
        clusters_to_combine = [
            "high_risk_countries",
            "high_risk_business_categories",
            "high_risk_age_groups",
            "high_risk_occupations",
            "high_risk_score"
        ]

        if include_super_high_risk:
            clusters_to_combine.append("super_high_risk")

        return self.get_combined_clusters(clusters_to_combine, deduplicate=True)

    def get_high_risk_individuals(self, cluster_names: List[str] = None, max_entities: int = None) -> List[str]:
        """Efficiently get high-risk individuals from specified clusters.

        Args:
            cluster_names: List of cluster names to search. Defaults to common high-risk clusters.
            max_entities: Maximum number of entities to return (for performance)

        Returns:
            List of individual entity IDs, prioritized by risk level
        """
        if cluster_names is None:
            cluster_names = ["structuring_candidates",
                             "intermediaries", "super_high_risk", "high_risk_score"]

        potential_individuals = []
        for cluster_name in cluster_names:
            # Get entities from existing clusters and filter to individuals
            cluster_entities = self.get_cluster(cluster_name)
            cluster_individuals = [e for e in cluster_entities
                                   if self.graph.nodes[e].get("node_type") == NodeType.INDIVIDUAL]
            potential_individuals.extend(cluster_individuals)

            # Early stopping for performance
            if max_entities and len(potential_individuals) > max_entities:
                break

        # Remove duplicates while preserving priority order
        return list(dict.fromkeys(potential_individuals))

    def get_entities_with_multiple_risk_factors(self) -> List[str]:
        """Get entities that appear in multiple risk clusters (indicating compound risk)."""
        risk_clusters = [
            "high_risk_countries",
            "high_risk_business_categories",
            "high_risk_age_groups",
            "high_risk_occupations",
            "high_risk_score"
        ]

        entity_risk_count = {}
        for cluster_name in risk_clusters:
            for entity_id in self.get_cluster(cluster_name):
                entity_risk_count[entity_id] = entity_risk_count.get(
                    entity_id, 0) + 1

        # Return entities with 2+ risk factors
        return [entity_id for entity_id, count in entity_risk_count.items() if count >= 2]

    def prioritize_by_risk_factors(self, entities: List[str]) -> List[str]:
        """Sort entities by number of risk factors (highest risk first)."""
        risk_clusters = [
            "high_risk_countries",
            "high_risk_business_categories",
            "high_risk_age_groups",
            "high_risk_occupations",
            "high_risk_score"
        ]

        entity_risk_count = {}
        for entity_id in entities:
            count = 0
            for cluster_name in risk_clusters:
                if entity_id in self.get_cluster(cluster_name):
                    count += 1
            entity_risk_count[entity_id] = count

        # Sort by risk count (descending), then by entity_id for stability
        return sorted(entities, key=lambda x: (-entity_risk_count.get(x, 0), x))


class TemporalComponent(ABC):
    """Base class for temporal components of patterns"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        self.graph = graph_generator.graph
        self.graph_generator = graph_generator
        self.params = params
        self.time_span = params.get("time_span", {})

    @abstractmethod
    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        """Generate transaction sequences based on temporal pattern"""
        pass

    def _get_owned_accounts(self, entity_id: str) -> List[str]:
        """Get all accounts owned by an entity (individual or business)"""
        owned_accounts = []

        # Direct accounts
        for neighbor_id in self.graph.neighbors(entity_id):
            if self.graph.nodes[neighbor_id].get("node_type") == NodeType.ACCOUNT:
                owned_accounts.append(neighbor_id)

        # If entity is an individual, also check business accounts they own
        if self.graph.nodes[entity_id].get("node_type") == NodeType.INDIVIDUAL:
            for neighbor_id in self.graph.neighbors(entity_id):
                if self.graph.nodes[neighbor_id].get("node_type") == NodeType.BUSINESS:
                    # Get accounts of owned businesses
                    for business_neighbor in self.graph.neighbors(neighbor_id):
                        if self.graph.nodes[business_neighbor].get("node_type") == NodeType.ACCOUNT:
                            owned_accounts.append(business_neighbor)

        return list(set(owned_accounts))  # Remove duplicates

    def generate_timestamps(self,
                            start_time: datetime.datetime,
                            pattern_type: str,
                            count: int) -> List[datetime.datetime]:
        """Generate timestamps based on different temporal patterns"""

        if pattern_type == "high_frequency":
            # Burst of transactions in short time window
            timestamps = []
            for i in range(count):
                offset_minutes = random_instance.randint(
                    0, 1440)  # Within 24 hours
                timestamps.append(
                    start_time + datetime.timedelta(minutes=offset_minutes))
            return sorted(timestamps)

        elif pattern_type == "synchronized":
            # The temporal component using this will handle synchronization across entities.
            timestamps = []
            for i in range(count):
                offset_minutes = random_instance.randint(
                    0, 1440)  # Within 24 hours
                timestamps.append(
                    start_time + datetime.timedelta(minutes=offset_minutes))
            return sorted(timestamps)

        elif pattern_type == "periodic":
            # Regular intervals over longer period
            timestamps = []
            current_time = start_time
            for i in range(count):
                timestamps.append(current_time)
                days_offset = random_instance.choice(
                    [7, 14, 30])  # Weekly/monthly
                current_time += datetime.timedelta(days=days_offset)
            return timestamps

        elif pattern_type == "immediate_followup":
            # Quick sequence after initial trigger
            timestamps = []
            for i in range(count):
                offset_minutes = random_instance.randint(
                    0, 30)  # Within 30 minutes
                timestamps.append(
                    start_time + datetime.timedelta(minutes=offset_minutes))
            return sorted(timestamps)

        else:
            # Random spread
            timestamps = []
            for i in range(count):
                offset_hours = random_instance.randint(0, 168)  # Within a week
                timestamps.append(
                    start_time + datetime.timedelta(hours=offset_hours))
            return sorted(timestamps)

    def convert_currency(self, amount: float, from_currency: str, to_currency: str,
                         transaction_date: datetime.datetime = None) -> float:
        """Convert amount from one currency to another using the utility module.

        Args:
            amount: Amount to convert
            from_currency: Source currency code (e.g., 'EUR', 'GBP')
            to_currency: Target currency code (e.g., 'USD')
            transaction_date: Date for historical rates (unused - using current rates)

        Returns:
            Converted amount in target currency
        """
        return convert_currency(amount, from_currency, to_currency)

    def generate_structured_amounts(self, count: int, base_amount: float = None,
                                    target_currency: str = None,
                                    transaction_date: datetime.datetime = None) -> List[float]:
        """Generate amounts that may be structured to avoid thresholds"""

        # Get reporting threshold and currency from config
        reporting_threshold = self.params.get("reporting_threshold", 10000)
        reporting_currency = self.params.get("reporting_currency", "USD")

        # Use the utility function for structured amount generation
        return generate_structured_amounts_util(
            count=count,
            reporting_threshold=reporting_threshold,
            reporting_currency=reporting_currency,
            target_currency=target_currency,
            base_amount=base_amount
        )


class CompositePattern(PatternInjector):
    """Base class for patterns composed of structural and temporal components"""

    def __init__(self,
                 structural_component: StructuralComponent,
                 temporal_component: TemporalComponent,
                 graph_generator,
                 params: Dict[str, Any]):
        super().__init__(graph_generator, params)
        self.structural = structural_component
        self.temporal = temporal_component

    @property
    def num_required_entities(self) -> int:
        """Number of entities required, delegated to the structural component."""
        return self.structural.num_required_entities

    def inject_pattern(self, entities: List[str]) -> List[Tuple[str, str, TransactionAttributes]]:
        """Generate all edges for this pattern without mutating the master graph.
        The caller (GraphGenerator) decides when to merge the resulting sub-graph."""

        generated_edges: List[Tuple[str, str, TransactionAttributes]] = []

        try:
            # Entity selection
            entity_selection = self.structural.select_entities(entities)
            if not entity_selection.central_entities:
                return generated_edges
            print(f"entities done for {self.__class__.__name__}")
            # Build sequences
            for sequence in self.temporal.generate_transaction_sequences(entity_selection):
                print(
                    f"Generated sequence {sequence.sequence_name}, for pattern {self.__class__.__name__}")
                generated_edges.extend(sequence.transactions)

        except Exception as e:
            print(f"Failed to build pattern {self.__class__.__name__}: {e}")

        return generated_edges

    @property
    def pattern_name(self) -> str:
        """Return a descriptive name for this pattern"""
        return f"{self.structural.__class__.__name__}_{self.temporal.__class__.__name__}"
