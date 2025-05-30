from ..utils.random_instance import random_instance
import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType


class RepeatedOverseasTransfersStructural(StructuralComponent):
    """
    Selects:
    - An individual or business entity (central).
    - One of their domestic accounts (peripheral).
    - Multiple overseas accounts (peripheral).
    """

    @property
    def num_required_entities(self) -> int:
        # Central entity (Ind/Bus) + 1 source account + min overseas accounts
        # The pattern config specifies min_overseas_entities
        # So, 1 central entity is the core requirement here.
        return 1

    def _filter_entities_by_type(self, entities: List[str]) -> List[str]:
        """Helper to filter entities to only individuals and businesses."""
        filtered = []
        for entity_id in entities:
            try:
                node_type = self.graph.nodes[entity_id].get("node_type")
                if node_type in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
                    filtered.append(entity_id)
            except KeyError:
                continue  # Skip entities that don't exist in graph
        return filtered

    def _is_account_owned_by_entity(self, account_id: str) -> bool:
        """Check if account is owned by an individual or business entity."""
        try:
            return any(
                self.graph.nodes.get(owner_id, {}).get("node_type") in [
                    NodeType.INDIVIDUAL, NodeType.BUSINESS]
                for owner_id in self.graph.predecessors(account_id)
            )
        except KeyError:
            return False

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        central_owner_id = None
        source_account_id = None
        overseas_account_ids = []

        pattern_config = self.params.get("repeatedOverseas", {})
        min_overseas_entities = pattern_config.get("min_overseas_entities", 2)
        max_overseas_entities = pattern_config.get("max_overseas_entities", 5)

        # Build prioritized list of potential source entities
        potential_source_entities = []

        # Priority 1: offshore_candidates and super_high_risk clusters
        for cluster_name in ["offshore_candidates", "super_high_risk"]:
            cluster_entities = self.get_cluster(cluster_name)
            print(f"Cluster {cluster_name}: {cluster_entities}")
            filtered_entities = self._filter_entities_by_type(cluster_entities)
            print(f"Filtered entities: {filtered_entities}")
            print("Difference: ", set(cluster_entities) - set(filtered_entities))
            potential_source_entities.extend(filtered_entities)
            if len(potential_source_entities) > 20:  # Gather a pool
                break

        # Priority 2: high_risk_countries if pool is small
        if len(potential_source_entities) < 10:
            high_risk_entities = self.get_cluster("high_risk_countries")
            filtered_entities = self._filter_entities_by_type(
                high_risk_entities)
            potential_source_entities.extend(filtered_entities)

        potential_source_entities = list(dict.fromkeys(
            potential_source_entities))  # Deduplicate

        # Fallback if specific clusters are empty or yield too few candidates
        if not potential_source_entities:
            raise ValueError(
                "No potential source entities found. Please check the graph configuration.")

        random_instance.shuffle(potential_source_entities)

        for entity_id in potential_source_entities:
            try:
                entity_node_data = self.graph.nodes[entity_id]
                entity_country = entity_node_data.get("country_code")

                # Find domestic accounts owned by this entity
                owned_domestic_accounts = []
                for acc_id in self.graph.neighbors(entity_id):
                    try:
                        acc_data = self.graph.nodes[acc_id]
                        if (acc_data.get("node_type") == NodeType.ACCOUNT and
                                acc_data.get("country_code") == entity_country):
                            owned_domestic_accounts.append(acc_id)
                    except KeyError:
                        continue

                if not owned_domestic_accounts:
                    continue

                source_account_id = random_instance.choice(
                    owned_domestic_accounts)

                # Search for overseas destination accounts
                high_risk_candidates = list(
                    self.get_cluster("high_risk_countries"))
                potential_overseas_accounts = []

                for acc_id in high_risk_candidates:
                    try:
                        acc_node_data = self.graph.nodes[acc_id]
                        acc_country = acc_node_data.get("country_code")

                        if (acc_country and acc_country != entity_country and
                                self._is_account_owned_by_entity(acc_id)):
                            potential_overseas_accounts.append(acc_id)
                    except KeyError:
                        continue

                if len(potential_overseas_accounts) >= min_overseas_entities:
                    central_owner_id = entity_id
                    num_to_select = random_instance.randint(
                        min_overseas_entities,
                        min(len(potential_overseas_accounts),
                            max_overseas_entities)
                    )
                    overseas_account_ids = random_instance.sample(
                        potential_overseas_accounts, num_to_select
                    )
                    break

            except KeyError:
                continue  # Skip entities with missing data

        if not central_owner_id or not source_account_id or not overseas_account_ids:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        return EntitySelection(
            central_entities=[central_owner_id],
            peripheral_entities=[source_account_id] + overseas_account_ids,
        )


class FrequentOrPeriodicTransfersTemporal(TemporalComponent):
    """
    Temporal: High frequency (burst) or periodic transfers.
    Amounts are often structured below reporting thresholds.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []
        if not entity_selection.central_entities or not entity_selection.peripheral_entities or len(entity_selection.peripheral_entities) < 2:
            return sequences

        # First peripheral is source, rest are destinations
        source_account_id = entity_selection.peripheral_entities[0]
        overseas_account_ids = entity_selection.peripheral_entities[1:]

        if not overseas_account_ids:
            return sequences

        # Load parameters from YAML config
        pattern_config = self.params.get("repeatedOverseas", {})
        tx_params = pattern_config.get("transaction_params", {})

        min_tx = tx_params.get("min_transactions", 10)
        max_tx = tx_params.get("max_transactions", 30)
        amount_range = tx_params.get("transfer_amount_range", [5000, 20000])
        interval_days_options = tx_params.get(
            "transfer_interval_days", [7, 14, 30])

        temporal_type = random_instance.choice(["high_frequency", "periodic"])
        num_transactions = random_instance.randint(min_tx, max_tx)

        # Calculate start time
        time_span_days = (
            self.time_span["end_date"] - self.time_span["start_date"]).days
        start_day_offset_range = max(
            0, time_span_days - 30) if time_span_days >= 30 else max(0, time_span_days - 1)

        base_start_time = self.time_span["start_date"] + datetime.timedelta(
            days=random_instance.randint(0, start_day_offset_range)
        )

        # Generate timestamps based on temporal type
        if temporal_type == "periodic" and interval_days_options:
            timestamps = []
            current_time = base_start_time
            for _ in range(num_transactions):
                if current_time > self.time_span["end_date"]:
                    break
                timestamps.append(current_time)
                current_time += datetime.timedelta(
                    days=random_instance.choice(interval_days_options))
            num_transactions = len(timestamps)
        else:
            timestamps = self.generate_timestamps(
                base_start_time, temporal_type, num_transactions)

        # Generate amounts (use EUR as default since we're dealing with multiple overseas accounts)
        amounts = self.generate_structured_amounts(
            count=num_transactions,
            base_amount=round(random_instance.uniform(
                amount_range[0], amount_range[1]), 2),
            target_currency="EUR"
        )

        # Create transactions
        transactions_for_sequence = []
        duration = timestamps[-1] - timestamps[0] if len(
            timestamps) > 1 else datetime.timedelta(days=0)

        for i in range(min(num_transactions, len(timestamps))):
            destination_account_id = random_instance.choice(
                overseas_account_ids)
            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=source_account_id,
                dest_id=destination_account_id,
                timestamp=timestamps[i],
                amount=amounts[i],
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=True
            )
            transactions_for_sequence.append(
                (source_account_id, destination_account_id, tx_attrs))

        if transactions_for_sequence:
            sequences.append(
                TransactionSequence(
                    transactions=transactions_for_sequence,
                    sequence_name=f"repeated_transfers_{temporal_type}",
                    start_time=timestamps[0] if timestamps else base_start_time,
                    duration=duration
                )
            )
        return sequences


class RepeatedOverseasTransfersPattern(CompositePattern):
    """Injects repeated overseas transfers pattern"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = RepeatedOverseasTransfersStructural(
            graph_generator, params)
        temporal_component = FrequentOrPeriodicTransfersTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "RepeatedOverseasTransfers"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
