import datetime
import numpy as np
from typing import List, Dict, Any, Tuple

from ..base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ...datastructures.enums import NodeType, TransactionType
from ...datastructures.attributes import TransactionAttributes
from ...utils.random_instance import random_instance


class BackgroundActivityStructural(StructuralComponent):
    """Structural component: treat all supplied accounts as actors for daily activity."""

    @property
    def num_required_entities(self) -> int:
        return 1  # At least one account

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        # We expect available_entities to be account IDs. Use them directly.
        if not available_entities:
            return EntitySelection(central_entities=[], peripheral_entities=[])
        return EntitySelection(central_entities=available_entities, peripheral_entities=[])


class DailyRandomTransfersTemporal(TemporalComponent):
    """Temporal component: scatter random low-value transactions across the timespan."""

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        all_accounts = entity_selection.central_entities
        if not all_accounts:
            return sequences

        start_date: datetime.datetime = self.time_span["start_date"]
        end_date: datetime.datetime = self.time_span["end_date"]
        total_days = max(1, (end_date - start_date).days)
        total_seconds = total_days * 86400  # days to seconds

        tx_rate = self.params.get("transaction_rates", {}).get(
            "per_account_per_day", 0.2)

        # Calculate total expected transactions across all accounts
        total_expected_txs = int(
            tx_rate * total_days * len(all_accounts) * random_instance.uniform(0.5, 1.5))
        if total_expected_txs == 0:
            return sequences

        # Generate all timestamps using random_instance for reproducibility
        random_seconds = [random_instance.randint(0, total_seconds - 1)
                          for _ in range(total_expected_txs)]
        timestamps = [
            start_date + datetime.timedelta(seconds=s) for s in random_seconds]

        # Generate all amounts using random_instance for reproducibility
        amount_range = self.params.get(
            "background_amount_range", [10.0, 500.0])
        amounts = [round(random_instance.uniform(amount_range[0], amount_range[1]), 2)
                   for _ in range(total_expected_txs)]

        # Pre-generate all source-destination pairs using random_instance
        src_indices = [random_instance.randint(0, len(all_accounts) - 1)
                       for _ in range(total_expected_txs)]
        dest_indices = [random_instance.randint(0, len(all_accounts) - 1)
                        for _ in range(total_expected_txs)]

        # Ensure src != dest by adjusting dest indices where they match
        for i in range(total_expected_txs):
            if src_indices[i] == dest_indices[i]:
                dest_indices[i] = (dest_indices[i] + 1) % len(all_accounts)

        # Convert indices to actual account IDs
        src_accounts = [all_accounts[i] for i in src_indices]
        dest_accounts = [all_accounts[i] for i in dest_indices]

        # Create all transactions in one go
        transactions: List[Tuple[str, str, TransactionAttributes]] = []
        pattern_injector = PatternInjector(self.graph_generator, self.params)

        for i in range(total_expected_txs):
            tx_attrs = pattern_injector._create_transaction_edge(
                src_id=src_accounts[i],
                dest_id=dest_accounts[i],
                timestamp=timestamps[i],
                amount=float(amounts[i]),
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=False,
            )
            transactions.append((src_accounts[i], dest_accounts[i], tx_attrs))

        # Create a single sequence for all transactions
        if transactions:
            sequence_name = "background_activity_all"
            timestamps_sorted = sorted(timestamps)
            sequences.append(
                TransactionSequence(
                    transactions=transactions,
                    sequence_name=sequence_name,
                    start_time=timestamps_sorted[0],
                    duration=timestamps_sorted[-1] - timestamps_sorted[0],
                )
            )

        return sequences


class BackgroundActivityPattern(CompositePattern):
    """Composite pattern wrapper for baseline activity."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = BackgroundActivityStructural(
            graph_generator, params)
        temporal_component = DailyRandomTransfersTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "BackgroundActivity"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
