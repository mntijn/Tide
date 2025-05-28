import random
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
        currency = "EUR"  # Default currency
        src_node_exists = self.graph.has_node(src_id)
        dest_node_exists = self.graph.has_node(dest_id)

        # Check source account for currency
        if src_node_exists and self.graph.nodes[src_id].get("node_type") == NodeType.ACCOUNT and "currency" in self.graph.nodes[src_id]:
            currency = self.graph.nodes[src_id]["currency"]
        # Else, check destination account for currency
        elif dest_node_exists and self.graph.nodes[dest_id].get("node_type") == NodeType.ACCOUNT and "currency" in self.graph.nodes[dest_id]:
            currency = self.graph.nodes[dest_id]["currency"]
        # Fallback to source node if not an account but has currency
        elif src_node_exists and "currency" in self.graph.nodes[src_id]:
            currency = self.graph.nodes[src_id]["currency"]
        # Fallback to destination node if not an account but has currency
        elif dest_node_exists and "currency" in self.graph.nodes[dest_id]:
            currency = self.graph.nodes[dest_id]["currency"]

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
    def select_entities_from_pools(self, fraudulent_pool: List[str], non_fraudulent_pool: List[str]) -> Optional[EntitySelection]:
        """
        Selects entities for the pattern from the provided pools
        """
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
            timestamps = []
            for i in range(count):
                offset_minutes = random.randint(0, 1440)  # Within 24 hours
                timestamps.append(
                    start_time + datetime.timedelta(minutes=offset_minutes))
            return sorted(timestamps)

        elif pattern_type == "synchronized":
            # The temporal component using this will handle synchronization across entities.
            timestamps = []
            for i in range(count):
                offset_minutes = random.randint(0, 1440)  # Within 24 hours
                timestamps.append(
                    start_time + datetime.timedelta(minutes=offset_minutes))
            return sorted(timestamps)

        elif pattern_type == "periodic":
            # Regular intervals over longer period
            timestamps = []
            current_time = start_time
            for i in range(count):
                timestamps.append(current_time)
                days_offset = random.choice([7, 14, 30])  # Weekly/monthly
                current_time += datetime.timedelta(days=days_offset)
            return timestamps

        elif pattern_type == "immediate_followup":
            # Quick sequence after initial trigger
            timestamps = []
            for i in range(count):
                offset_minutes = random.randint(0, 30)  # Within 30 minutes
                timestamps.append(
                    start_time + datetime.timedelta(minutes=offset_minutes))
            return sorted(timestamps)

        else:
            # Random spread
            timestamps = []
            for i in range(count):
                offset_hours = random.randint(0, 168)  # Within a week
                timestamps.append(
                    start_time + datetime.timedelta(hours=offset_hours))
            return sorted(timestamps)

    def generate_structured_amounts(self, count: int, base_amount: float = None) -> List[float]:
        """Generate amounts that may be structured to avoid thresholds"""
        if base_amount is None:
            reporting_threshold = self.params.get("reporting_threshold", 10000)
            base_amount = reporting_threshold * \
                round(random.uniform(0.7, 0.95), 2)

        amounts = []
        for i in range(count):
            # Add variation to avoid exact patterns
            variation = round(random.uniform(-base_amount *
                              0.15, base_amount * 0.15))
            amounts.append(max(100, base_amount + variation))

        return amounts


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
        self.graph_generator = graph_generator

    @property
    def num_required_entities(self) -> int:
        """Number of entities required, delegated to the structural component."""
        return self.structural.num_required_entities

    def inject_pattern_with_selection(self, entity_selection: Optional[EntitySelection]) -> List[Tuple[str, str, TransactionAttributes]]:
        """Main method to inject the complete pattern using a pre-selected EntitySelection."""
        fraudulent_edges = []

        if not entity_selection or not entity_selection.central_entities:
            return fraudulent_edges

        try:
            # Generate transaction sequences based on temporal pattern
            transaction_sequences = self.temporal.generate_transaction_sequences(
                entity_selection)

            # Execute all transactions
            for sequence in transaction_sequences:
                for src, dest, tx_attrs in sequence.transactions:
                    self.graph_generator._add_edge(src, dest, tx_attrs)
                    fraudulent_edges.append((src, dest, tx_attrs))

            return fraudulent_edges

        except Exception as e:
            print(
                f"Failed to inject {self.__class__.__name__} with pre-selected entities: {e}")
            return fraudulent_edges

    @property
    def pattern_name(self) -> str:
        """Return a descriptive name for this pattern"""
        return f"{self.structural.__class__.__name__}_{self.temporal.__class__.__name__}"
