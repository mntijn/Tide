import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType
from ..datastructures.attributes import TransactionAttributes
from ..utils.random_instance import random_instance


class FraudulentEntitiesStructural(StructuralComponent):
    """Structural component: selects a subset of fraudulent entities."""

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        cover_config = self.params.get("background_activity", {}).get(
            "patterns", {}).get("cover_traffic", {})
        if not cover_config.get("enabled", False):
            return EntitySelection(central_entities=[], peripheral_entities=[])

        participation_rate = cover_config.get("participation_rate", 0.8)

        # Get all entities that participated in a fraudulent transaction
        fraudulent_entities = self.graph_generator.entity_clusters.get(
            "fraudulent", [])

        if not fraudulent_entities:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        num_to_select = int(len(fraudulent_entities) * participation_rate)
        selected_entities = random_instance.sample(
            fraudulent_entities, num_to_select)

        return EntitySelection(central_entities=selected_entities, peripheral_entities=[])


class CoverTransactionsTemporal(TemporalComponent):
    """Temporal component: adds extra daily transactions for selected entities."""

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        if not entity_selection.central_entities:
            return sequences

        cover_config = self.params.get("background_activity", {}).get(
            "patterns", {}).get("cover_traffic", {})
        extra_tx_rate = cover_config.get("extra_tx_rate_per_day", 0.5)
        amount_range = self.params.get(
            "background_amount_range", [10.0, 500.0])

        all_accounts = self.graph_generator.all_nodes.get(NodeType.ACCOUNT, [])
        if len(all_accounts) < 2:
            return sequences

        transactions: List[Tuple[str, str, TransactionAttributes]] = []
        pattern_injector = PatternInjector(self.graph_generator, self.params)

        total_days = max(
            1, (self.time_span["end_date"] - self.time_span["start_date"]).days)
        num_extra_txs = int(
            len(entity_selection.central_entities) * extra_tx_rate * total_days)

        for _ in range(num_extra_txs):
            src_account = random_instance.choice(
                entity_selection.central_entities)

            # Ensure the entity is an account, or find an account for it
            if self.graph.nodes[src_account].get("node_type") != NodeType.ACCOUNT:
                owned_accounts = self._get_owned_accounts(src_account)
                if not owned_accounts:
                    continue
                src_account = random_instance.choice(owned_accounts)

            dest_account = random_instance.choice(all_accounts)
            while dest_account == src_account:
                dest_account = random_instance.choice(all_accounts)

            timestamp = self.time_span["start_date"] + datetime.timedelta(
                seconds=random_instance.randint(0, total_days * 86400 - 1)
            )
            amount = random_instance.uniform(amount_range[0], amount_range[1])

            tx_attrs = pattern_injector._create_transaction_edge(
                src_id=src_account,
                dest_id=dest_account,
                timestamp=timestamp,
                amount=round(amount, 2),
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=False,
            )
            transactions.append((src_account, dest_account, tx_attrs))

        if transactions:
            sequences.append(TransactionSequence(
                transactions=transactions,
                sequence_name="cover_transactions",
                start_time=self.time_span["start_date"],
                duration=self.time_span["end_date"] -
                self.time_span["start_date"],
            ))
        return sequences


class CoverTransactionsPattern(CompositePattern):
    """Adds extra non-fraudulent transactions for fraudulent entities."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = FraudulentEntitiesStructural(
            graph_generator, params)
        temporal_component = CoverTransactionsTemporal(graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "CoverTransactions"

    @property
    def num_required_entities(self) -> int:
        return 0
