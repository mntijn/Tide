import random
import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..data_structures import NodeType, TransactionType, TransactionAttributes


class IndividualWithMultipleAccountsStructural(StructuralComponent):
    """
    Structural: An individual with multiple accounts (personal and/or business).
    Focuses on identifying an individual who owns several accounts.
    """

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        central_individual_id = None
        peripheral_account_ids = []

        individual_ids = self.filter_entities_by_criteria(
            available_entities, {"node_type": NodeType.INDIVIDUAL}
        )
        random.shuffle(individual_ids)

        # Default for now
        min_accounts_for_pattern = 2

        for ind_id in individual_ids:
            owned_accounts = set()
            for acc_node in self.graph.neighbors(ind_id):
                if self.graph.nodes[acc_node].get("node_type") == NodeType.ACCOUNT:
                    owned_accounts.add(acc_node)

            for owned_node_id in self.graph.neighbors(ind_id):
                if self.graph.nodes[owned_node_id].get("node_type") == NodeType.BUSINESS:
                    for acc_node in self.graph.neighbors(owned_node_id):
                        if self.graph.nodes[acc_node].get("node_type") == NodeType.ACCOUNT:
                            owned_accounts.add(acc_node)

            if len(owned_accounts) >= min_accounts_for_pattern:
                central_individual_id = ind_id
                peripheral_account_ids = list(owned_accounts)
                break

        if not central_individual_id or not peripheral_account_ids:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        return EntitySelection(
            central_entities=peripheral_account_ids,
            peripheral_entities=[central_individual_id],
        )


class RapidInflowOutflowTemporal(TemporalComponent):
    """
    Temporal: High volume of incoming transfers, then high frequency of cash withdrawals.
    Withdrawals can be across multiple banks and structured.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []
        if not entity_selection.central_entities:
            return sequences

        individual_accounts = entity_selection.central_entities
        primary_inflow_account = random.choice(individual_accounts)

        # Default values for pattern parameters
        min_inflows = 15
        max_inflows = 30
        min_withdrawals = 20
        max_withdrawals = 50

        num_incoming_transfers = random.randint(min_inflows, max_inflows)
        num_withdrawals = random.randint(min_withdrawals, max_withdrawals)

        all_accounts_in_graph = [
            node_id for node_id, data in self.graph.nodes(data=True)
            if data.get("node_type") == NodeType.ACCOUNT
        ]
        external_source_accounts = [
            acc_id for acc_id in all_accounts_in_graph
            if acc_id not in individual_accounts
        ]

        if not external_source_accounts:
            print("Warning: No external accounts for RapidInflowOutflowTemporal inflows.")
            return sequences

        if (self.time_span["end_date"] - self.time_span["start_date"]).days < 10:
            start_day_offset_range_inflow = max(
                0, (self.time_span["end_date"] - self.time_span["start_date"]).days - 1)
        else:
            start_day_offset_range_inflow = (
                self.time_span["end_date"] - self.time_span["start_date"]).days - 10

        inflow_start_time = self.time_span["start_date"] + datetime.timedelta(
            days=random.randint(0, start_day_offset_range_inflow)
        )
        inflow_timestamps = self.generate_timestamps(
            inflow_start_time, "high_frequency", num_incoming_transfers)
        inflow_amounts = self.generate_structured_amounts(
            num_incoming_transfers, base_amount=random.uniform(500, 5000))

        inflow_transactions = []
        inflow_duration = datetime.timedelta(days=0)
        if inflow_timestamps:
            inflow_duration = inflow_timestamps[-1] - inflow_timestamps[0]

        for i in range(min(num_incoming_transfers, len(inflow_timestamps))):
            source_external_account = random.choice(external_source_accounts)
            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=source_external_account,
                dest_id=primary_inflow_account,
                timestamp=inflow_timestamps[i],
                amount=inflow_amounts[i],
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=True
            )
            inflow_transactions.append(
                (source_external_account, primary_inflow_account, tx_attrs))

        if inflow_transactions:
            sequences.append(TransactionSequence(
                transactions=inflow_transactions,
                sequence_name="rapid_inflows",
                start_time=inflow_timestamps[0] if inflow_timestamps else inflow_start_time,
                duration=inflow_duration
            ))

        last_inflow_time = inflow_timestamps[-1] if inflow_timestamps else inflow_start_time
        withdrawal_start_time = last_inflow_time + \
            datetime.timedelta(hours=random.uniform(1, 24))

        withdrawal_timestamps = self.generate_timestamps(
            withdrawal_start_time, "high_frequency", num_withdrawals)
        withdrawal_amounts = self.generate_structured_amounts(num_withdrawals)

        withdrawal_transactions = []
        withdrawal_duration = datetime.timedelta(days=0)
        if withdrawal_timestamps:
            withdrawal_duration = withdrawal_timestamps[-1] - \
                withdrawal_timestamps[0]

        # Locate the central individual to obtain their cash account
        central_individual_id = entity_selection.peripheral_entities[
            0] if entity_selection.peripheral_entities else None
        cash_account_id = None
        if central_individual_id:
            for neigh in self.graph.neighbors(central_individual_id):
                if self.graph.nodes[neigh].get("is_cash_account"):
                    cash_account_id = neigh
                    break

        for i in range(min(num_withdrawals, len(withdrawal_timestamps))):
            account_for_withdrawal = random.choice(individual_accounts)
            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=account_for_withdrawal,
                dest_id=cash_account_id,
                timestamp=withdrawal_timestamps[i],
                amount=withdrawal_amounts[i],
                transaction_type=TransactionType.WITHDRAWAL,
                is_fraudulent=True
            )
            withdrawal_transactions.append(
                (account_for_withdrawal, cash_account_id, tx_attrs))

        if withdrawal_transactions:
            sequences.append(TransactionSequence(
                transactions=withdrawal_transactions,
                sequence_name="rapid_withdrawals",
                start_time=withdrawal_timestamps[0] if withdrawal_timestamps else withdrawal_start_time,
                duration=withdrawal_duration
            ))
        return sequences


class RapidFundMovementStructural(StructuralComponent):
    """
    Selects entities for rapid fund movement:
    - A source account (individual or business).
    - One or more intermediary accounts.
    - A destination account.
    """
    @property
    def num_required_entities(self) -> int:
        return 3  # Source, at least one intermediary, destination

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        central_individual_id = None
        peripheral_account_ids = []

        individual_ids = self.filter_entities_by_criteria(
            available_entities, {"node_type": NodeType.INDIVIDUAL}
        )
        random.shuffle(individual_ids)

        # Default for now
        min_accounts_for_pattern = 2

        for ind_id in individual_ids:
            owned_accounts = set()
            for acc_node in self.graph.neighbors(ind_id):
                if self.graph.nodes[acc_node].get("node_type") == NodeType.ACCOUNT:
                    owned_accounts.add(acc_node)

            for owned_node_id in self.graph.neighbors(ind_id):
                if self.graph.nodes[owned_node_id].get("node_type") == NodeType.BUSINESS:
                    for acc_node in self.graph.neighbors(owned_node_id):
                        if self.graph.nodes[acc_node].get("node_type") == NodeType.ACCOUNT:
                            owned_accounts.add(acc_node)

            if len(owned_accounts) >= min_accounts_for_pattern:
                central_individual_id = ind_id
                peripheral_account_ids = list(owned_accounts)
                break

        if not central_individual_id or not peripheral_account_ids:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        return EntitySelection(
            central_entities=peripheral_account_ids,
            peripheral_entities=[central_individual_id],
        )


class RapidFundMovementPattern(CompositePattern):
    """Injects rapid fund movement pattern"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = RapidFundMovementStructural(
            graph_generator, params)
        temporal_component = RapidInflowOutflowTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "RapidFundMovement"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
