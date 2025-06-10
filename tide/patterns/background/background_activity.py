import datetime
from typing import List, Dict, Any, Tuple

from ..base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ...datastructures.enums import NodeType, TransactionType
from ...datastructures.attributes import TransactionAttributes
from ...utils.random_instance import random_instance


class RandomTransactionsStructural(StructuralComponent):
    """Structural component: treat all supplied accounts as actors for daily activity."""

    @property
    def num_required_entities(self) -> int:
        return 1

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        random_config = self.params.get("background_activity", {}).get(
            "patterns", {}).get("random_transfers", {})
        if not random_config.get("enabled", False):
            return EntitySelection(central_entities=[], peripheral_entities=[])

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
        total_seconds = total_days * 86400

        tx_rate = self.params.get("transaction_rates", {}).get(
            "per_account_per_day", 0.2)

        total_expected_txs = int(
            tx_rate * total_days * len(all_accounts) * random_instance.uniform(0.5, 1.5))
        if total_expected_txs == 0:
            return sequences

        random_seconds = [random_instance.randint(0, total_seconds - 1)
                          for _ in range(total_expected_txs)]
        timestamps = [
            start_date + datetime.timedelta(seconds=s) for s in random_seconds]

        amount_range = self.params.get(
            "background_amount_range", [10.0, 500.0])
        high_value_amount_range = self.params.get(
            "high_value_amount_range", [5000.0, 50000.0])
        high_value_ratio = self.params.get(
            "high_transaction_amount_ratio", 0.05)

        amounts = []
        for _ in range(total_expected_txs):
            if random_instance.random() < high_value_ratio:
                amount = random_instance.uniform(
                    high_value_amount_range[0], high_value_amount_range[1])
            else:
                amount = random_instance.uniform(
                    amount_range[0], amount_range[1])
            amounts.append(round(amount, 2))

        src_indices = [random_instance.randint(0, len(all_accounts) - 1)
                       for _ in range(total_expected_txs)]
        dest_indices = [random_instance.randint(0, len(all_accounts) - 1)
                        for _ in range(total_expected_txs)]

        for i in range(total_expected_txs):
            if src_indices[i] == dest_indices[i]:
                dest_indices[i] = (dest_indices[i] + 1) % len(all_accounts)

        src_accounts = [all_accounts[i] for i in src_indices]
        dest_accounts = [all_accounts[i] for i in dest_indices]

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

        if transactions:
            sequence_name = "random_background_activity"
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


class RandomTransfersPattern(CompositePattern):
    """Composite pattern wrapper for baseline activity."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = RandomTransactionsStructural(
            graph_generator, params)
        temporal_component = DailyRandomTransfersTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "RandomTransfers"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
