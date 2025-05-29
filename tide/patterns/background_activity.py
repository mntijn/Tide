import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType
from ..datastructures.attributes import TransactionAttributes
from ..utils.random_instance import random_instance


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

        tx_rate = self.params.get("transaction_rates", {}).get(
            "per_account_per_day", 0.2)

        for acc_id in all_accounts:
            # Slight randomness around the configured rate
            expected_txs = int(tx_rate * total_days *
                               random_instance.uniform(0.5, 1.5))
            if expected_txs == 0:
                continue

            transactions: List[Tuple[str, str, TransactionAttributes]] = []
            timestamps: List[datetime.datetime] = []
            for _ in range(expected_txs):
                # Pick a random moment within time span
                random_day_offset = random_instance.randint(0, total_days)
                random_second = random_instance.randint(0, 86_399)
                ts = start_date + \
                    datetime.timedelta(days=random_day_offset,
                                       seconds=random_second)
                timestamps.append(ts)

                # Pick a random destination account (excluding self)
                # Ensure there is more than one account to pick from to avoid infinite loop
                if len(all_accounts) <= 1:
                    continue  # Cannot make a transfer

                dest_id = random_instance.choice(all_accounts)
                while dest_id == acc_id:
                    dest_id = random_instance.choice(all_accounts)

                # Use background_amount_range from params (loaded from graph.yaml)
                amount_range = self.params.get("background_amount_range", [
                                               10.0, 500.0])  # Default if not in params
                amount = round(random_instance.uniform(
                    amount_range[0], amount_range[1]), 2)

                tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                    src_id=acc_id,
                    dest_id=dest_id,
                    timestamp=ts,
                    amount=amount,
                    transaction_type=TransactionType.TRANSFER,
                    is_fraudulent=False,
                )
                transactions.append((acc_id, dest_id, tx_attrs))

            if transactions:
                sequence_name = f"background_activity_{acc_id}"
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
