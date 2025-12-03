import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType
from ..datastructures.attributes import TransactionAttributes
from ..utils.random_instance import random_instance


class SynchronisedTransactionsStructural(StructuralComponent):
    """
    Structural: Multiple seemingly unrelated entities that will perform
    coordinated transactions to a common recipient.
    """

    @property
    def num_required_entities(self) -> int:
        # Need at least 3 entities: 2 coordinating depositors + 1 recipient
        return 3

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        pattern_config = self.params.get("pattern_config", {}).get(
            "synchronisedTransactions", {})
        min_coordinating_entities = pattern_config.get(
            "min_coordinating_entities", 3)
        max_coordinating_entities = pattern_config.get(
            "max_coordinating_entities", 8)

        # First, find a suitable recipient (could be individual or business)
        recipient_id = None
        recipient_account_id = None

        # Get all individuals and businesses from pre-computed all_nodes
        all_individuals = list(
            self.graph_generator.all_nodes.get(NodeType.INDIVIDUAL, []))
        all_businesses = list(
            self.graph_generator.all_nodes.get(NodeType.BUSINESS, []))
        all_entities = all_individuals + all_businesses

        # Use mixed selection for recipients: ~65% high-risk, ~35% general population
        recipient_clusters = ["super_high_risk",
                              "offshore_candidates", "high_risk_countries"]
        potential_recipients = self.get_mixed_risk_entities(
            high_risk_clusters=recipient_clusters,
            fallback_pool=all_entities,
            num_needed=max(20, len(all_entities) // 10),
            high_risk_ratio=0.65
        )

        from .base import deduplicate_preserving_order
        potential_recipients = deduplicate_preserving_order(
            potential_recipients)
        random_instance.shuffle(potential_recipients)

        # Find a recipient with at least one account
        for entity_id in potential_recipients:
            # Get accounts owned by this entity
            owned_accounts = [
                n for n in self.graph.neighbors(entity_id)
                if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
            ]

            if owned_accounts:
                recipient_id = entity_id
                recipient_account_id = random_instance.choice(owned_accounts)
                break

        if not recipient_id or not recipient_account_id:
            # Fallback: pick any entity with an account
            random_instance.shuffle(all_entities)

            for entity_id in all_entities:
                owned_accounts = [
                    n for n in self.graph.neighbors(entity_id)
                    if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
                ]
                if owned_accounts:
                    recipient_id = entity_id
                    recipient_account_id = random_instance.choice(
                        owned_accounts)
                    break

        if not recipient_id:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Now find coordinating entities (individuals who will make deposits)
        coordinating_entities = []
        coordinating_accounts = []

        # Use mixed selection for coordinators: ~65% high-risk, ~35% general population
        coordinator_clusters = ["intermediaries",
                                "high_risk_countries", "high_risk_age_groups"]
        potential_coordinators = self.get_mixed_risk_entities(
            high_risk_clusters=coordinator_clusters,
            fallback_pool=all_individuals,
            num_needed=max(max_coordinating_entities *
                           2, len(all_individuals) // 5),
            high_risk_ratio=0.65,
            node_type_filter=NodeType.INDIVIDUAL
        )

        random_instance.shuffle(potential_coordinators)

        for ind_id in potential_coordinators:
            if ind_id == recipient_id:  # Skip the recipient
                continue

            # Get accounts owned by this individual
            owned_accounts = [
                n for n in self.graph.neighbors(ind_id)
                if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
            ]

            if owned_accounts:
                coordinating_entities.append(ind_id)
                coordinating_accounts.append(
                    random_instance.choice(owned_accounts))

                if len(coordinating_entities) >= max_coordinating_entities:
                    break

        if len(coordinating_entities) < min_coordinating_entities:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Trim to desired number
        num_to_use = random_instance.randint(min_coordinating_entities,
                                             min(len(coordinating_entities), max_coordinating_entities))
        coordinating_entities = coordinating_entities[:num_to_use]
        coordinating_accounts = coordinating_accounts[:num_to_use]

        return EntitySelection(
            # The common recipient account
            central_entities=[recipient_account_id],
            peripheral_entities=coordinating_accounts  # The coordinating depositor accounts
        )


class SynchronisedTransactionsTemporal(TemporalComponent):
    """
    Temporal: Multiple entities perform deposits at nearly the same time,
    followed by rapid transfer of aggregated funds.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []

        if not entity_selection.central_entities or not entity_selection.peripheral_entities:
            return sequences

        recipient_account_id = entity_selection.central_entities[0]
        coordinating_accounts = entity_selection.peripheral_entities

        pattern_config = self.params.get("pattern_config", {}).get(
            "synchronisedTransactions", {})
        tx_params = pattern_config.get("transaction_params", {})

        deposit_amount_range = tx_params.get(
            "deposit_amount_range", [2000, 9500])
        sync_window_hours = tx_params.get("sync_window_hours", 2)
        transfer_delay_hours = tx_params.get("transfer_delay_hours", [1, 6])

        # Calculate start time
        time_span_days = (
            self.time_span["end_date"] - self.time_span["start_date"]).days
        start_day_offset = random_instance.randint(
            0, max(0, time_span_days - 1))
        base_start_time = self.time_span["start_date"] + \
            datetime.timedelta(days=start_day_offset)

        # Generate synchronized deposit timestamps
        sync_base_time = base_start_time + \
            datetime.timedelta(
                hours=random_instance.randint(9, 15))  # Business hours

        deposit_transactions = []
        total_deposited = 0

        # Get cash account
        cash_account_id = self.graph_generator.cash_account_id
        if not cash_account_id:
            return sequences

        # Each coordinating account makes a cash deposit within the sync window
        for account_id in coordinating_accounts:
            # Small random offset within sync window
            offset_minutes = random_instance.randint(0, sync_window_hours * 60)
            deposit_time = sync_base_time + \
                datetime.timedelta(minutes=offset_minutes)

            # Get account currency for structuring
            account_currency = self.graph.nodes[account_id].get(
                "currency", "EUR")

            # Generate structured amount
            amounts = self.generate_structured_amounts(
                count=1,
                base_amount=random_instance.uniform(
                    deposit_amount_range[0], deposit_amount_range[1]),
                target_currency=account_currency
            )
            deposit_amount = amounts[0]

            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=cash_account_id,
                dest_id=account_id,
                timestamp=deposit_time,
                amount=deposit_amount,
                transaction_type=TransactionType.DEPOSIT,
                is_fraudulent=True
            )

            deposit_transactions.append(
                (cash_account_id, account_id, tx_attrs))
            total_deposited += deposit_amount

        # Create deposit sequence
        if deposit_transactions:
            deposit_times = [tx[2].timestamp for tx in deposit_transactions]
            sequences.append(TransactionSequence(
                transactions=deposit_transactions,
                sequence_name="synchronized_deposits",
                start_time=min(deposit_times),
                duration=max(deposit_times) - min(deposit_times)
            ))

        # Now create rapid transfers to recipient
        last_deposit_time = max(
            [tx[2].timestamp for tx in deposit_transactions])
        transfer_start_time = last_deposit_time + datetime.timedelta(
            hours=random_instance.uniform(
                transfer_delay_hours[0], transfer_delay_hours[1])
        )

        transfer_transactions = []

        # Each account transfers to the recipient
        for i, account_id in enumerate(coordinating_accounts):
            transfer_time = transfer_start_time + \
                datetime.timedelta(minutes=random_instance.randint(0, 30))

            # Transfer most of what was deposited
            transfer_amount = deposit_transactions[i][2].amount * \
                random_instance.uniform(0.85, 0.95)

            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=account_id,
                dest_id=recipient_account_id,
                timestamp=transfer_time,
                amount=transfer_amount,
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=True
            )

            transfer_transactions.append(
                (account_id, recipient_account_id, tx_attrs))

        # Create transfer sequence
        if transfer_transactions:
            transfer_times = [tx[2].timestamp for tx in transfer_transactions]
            sequences.append(TransactionSequence(
                transactions=transfer_transactions,
                sequence_name="synchronized_transfers",
                start_time=min(transfer_times),
                duration=max(transfer_times) - min(transfer_times)
            ))

        return sequences


class SynchronisedTransactionsPattern(CompositePattern):
    """Injects synchronized transactions pattern"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = SynchronisedTransactionsStructural(
            graph_generator, params)
        temporal_component = SynchronisedTransactionsTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "SynchronisedTransactions"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
