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
from ..utils.random_instance import random_instance


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

        return TransactionAttributes(
            timestamp=timestamp,
            amount=amount,
            currency=currency,
            transaction_type=transaction_type,
            is_fraudulent=is_fraudulent
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

            # Check if entity has multiple accounts
            if criteria.get("multiple_accounts"):
                # Count accounts owned by this entity
                account_count = sum(1 for node_id in self.graph.nodes
                                    if self.graph.nodes[node_id].get("node_type") == NodeType.ACCOUNT
                                    and any(self.graph.has_edge(entity_id, node_id) for _ in [1]))
                if account_count < criteria.get("min_accounts", 2):
                    continue

            # Check for overseas connections
            if criteria.get("overseas_connections"):
                entity_country = attrs.get("address", {}).get("country")
                has_overseas = any(
                    self.graph.nodes[neighbor].get("address", {}).get(
                        "country") != entity_country
                    for neighbor in self.graph.neighbors(entity_id)
                )
                if not has_overseas:
                    continue

            filtered.append(entity_id)

        return filtered

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
                offset_hours = random_instance.randint(
                    0, 168)  # Within a week
                timestamps.append(
                    start_time + datetime.timedelta(hours=offset_hours))
            return sorted(timestamps)

    def generate_structured_amounts(self, count: int, base_amount: float = None, target_currency: str = None) -> List[float]:
        """Generate amounts that may be structured to avoid thresholds"""
        if base_amount is None:
            reporting_threshold = self.params.get("reporting_threshold", 10000)
            base_amount = reporting_threshold * \
                round(random_instance.uniform(0.7, 0.95), 2)

        amounts = []
        for i in range(count):
            # Add variation to avoid exact patterns
            variation = round(random_instance.uniform(-base_amount *
                              0.15, base_amount * 0.15))
            amounts.append(max(100, base_amount + variation))

        return amounts


class CompositePattern(PatternInjector):
    """Base class for composite patterns that combine structural and temporal components."""

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
        """Minimum number of entities required for this pattern."""
        return self.structural.num_required_entities

    def inject_pattern(self, entities: List[str]) -> Tuple[List[Tuple[str, str, TransactionAttributes]], EntitySelection]:
        """Injects the pattern and returns the generated edges and selected entities."""
        entity_selection = self.structural.select_entities(entities)

        if not entity_selection.central_entities and not entity_selection.peripheral_entities:
            return [], entity_selection

        # Generate transaction sequences based on the selected entities
        sequences = self.temporal.generate_transaction_sequences(
            entity_selection)

        # Collect all transaction edges from all sequences
        all_edges = []
        for seq in sequences:
            all_edges.extend(seq.transactions)

        return all_edges, entity_selection

    @property
    def pattern_name(self) -> str:
        """Name of the pattern, must be overridden."""
        raise NotImplementedError
