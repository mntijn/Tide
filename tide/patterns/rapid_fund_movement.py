import random
import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType
from ..datastructures.attributes import TransactionAttributes


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
        if not entity_selection.central_entities or not entity_selection.peripheral_entities:
            return sequences

        central_individual_id = entity_selection.central_entities[0]
        individual_accounts = entity_selection.peripheral_entities

        if not individual_accounts:  # Need accounts to operate on
            return sequences

        primary_inflow_account = random.choice(individual_accounts)

        # Load parameters from YAML config
        pattern_config = self.params.get(
            "pattern_config", {}).get("rapidMovement", {})
        tx_params = pattern_config.get("transaction_params", {})
        inflow_params = tx_params.get("inflows", {})
        withdrawal_params = tx_params.get("withdrawals", {})

        min_inflows = inflow_params.get("min_inflows", 15)
        max_inflows = inflow_params.get("max_inflows", 30)
        inflow_amount_range = inflow_params.get("amount_range", [500, 5000])

        min_withdrawals = withdrawal_params.get("min_withdrawals", 20)
        max_withdrawals = withdrawal_params.get("max_withdrawals", 50)
        withdrawal_amount_range = withdrawal_params.get(
            "amount_range", [100, 2000])

        # inflow_to_withdrawal_delay_hours is also in YAML, e.g. [1, 24]
        delay_hours_range = tx_params.get(
            "inflow_to_withdrawal_delay", [1, 24])

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

        # Get the currency of the primary inflow account for proper structuring
        inflow_account_currency = self.graph.nodes[primary_inflow_account].get(
            "currency", "EUR")

        inflow_amounts = self.generate_structured_amounts(
            count=num_incoming_transfers,
            base_amount=round(random.uniform(
                inflow_amount_range[0], inflow_amount_range[1]), 2),
            target_currency=inflow_account_currency
        )

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
            datetime.timedelta(hours=random.uniform(
                delay_hours_range[0], delay_hours_range[1]))

        withdrawal_timestamps = self.generate_timestamps(
            withdrawal_start_time, "high_frequency", num_withdrawals)

        # Get the currency from the individual accounts for proper structuring
        # Sample an account to determine currency (assuming all accounts have similar currencies)
        sample_account = random.choice(individual_accounts)
        account_currency = self.graph.nodes[sample_account].get(
            "currency", "EUR")

        withdrawal_amounts = self.generate_structured_amounts(
            count=num_withdrawals,
            base_amount=round(random.uniform(
                withdrawal_amount_range[0], withdrawal_amount_range[1]), 2),
            target_currency=account_currency
        )

        withdrawal_transactions = []
        withdrawal_duration = datetime.timedelta(days=0)
        if withdrawal_timestamps:
            withdrawal_duration = withdrawal_timestamps[-1] - \
                withdrawal_timestamps[0]

        # Locate the central individual to obtain their cash account
        # central_individual_id = entity_selection.peripheral_entities[
        #     0] if entity_selection.peripheral_entities else None # This was incorrect

        # Use the global cash account ID from graph_generator
        cash_account_id = self.graph_generator.cash_account_id

        if not cash_account_id:
            print(
                "Warning: Global cash_account_id not found for RapidInflowOutflowTemporal withdrawals.")
            # Potentially fall back or skip: for now, if no cash account, cannot make withdrawals.
            # This could happen if cash_account_id was not created in GraphGenerator.
            # We might need to ensure it always exists or handle this more gracefully.
            # If we proceed without it, _create_transaction_edge might fail or use a default.
            # Let's assume for now that if it's None, we can't make these specific withdrawals.
            # However, the pattern might still want to create other types of outflows if applicable.
            # For this pattern, withdrawal to cash is key.
            if inflow_transactions and not withdrawal_transactions:  # if only inflows happened
                return sequences  # Or return only inflow sequence
            return []  # No transactions can be made if cash account is missing for withdrawal step

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
        # An individual and at least N accounts for them to use.
        # The config "min_accounts_for_pattern" implies the N.
        return 1  # The individual is the key central entity.

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        central_individual_id = None
        peripheral_account_ids = []

        # Configurable minimum number of accounts for this pattern
        pattern_config = self.params.get("rapidMovement", {})
        min_accounts_for_pattern = pattern_config.get(
            "min_accounts_for_pattern", 2)

        potential_individuals = []
        # Prioritize individuals from relevant clusters
        # Order: structuring_candidates -> intermediaries -> super_high_risk -> high_risk_score -> all individuals
        for cluster_name in ["structuring_candidates", "intermediaries", "super_high_risk", "high_risk_score"]:
            cluster_individuals = self.filter_entities_by_criteria(
                self.get_cluster(cluster_name), {
                    "node_type": NodeType.INDIVIDUAL}
            )
            potential_individuals.extend(cluster_individuals)
            if len(potential_individuals) > 20:  # Gather a decent pool before stopping early
                break

        # Remove duplicates maintaining order somewhat by creating a set then list
        potential_individuals = list(dict.fromkeys(potential_individuals))

        # Fallback to all individuals if clusters yield too few
        if len(potential_individuals) < 5:
            all_individuals_in_graph = self.filter_entities_by_criteria(
                available_entities, {"node_type": NodeType.INDIVIDUAL}
            )
            potential_individuals.extend(all_individuals_in_graph)
            potential_individuals = list(dict.fromkeys(
                potential_individuals))  # Deduplicate again

        random.shuffle(potential_individuals)

        for ind_id in potential_individuals:
            owned_accounts = set()
            # Direct accounts of the individual
            for acc_node in self.graph.neighbors(ind_id):
                if self.graph.nodes[acc_node].get("node_type") == NodeType.ACCOUNT:
                    owned_accounts.add(acc_node)

            # Accounts of businesses owned by the individual
            for owned_business_id in self.graph.neighbors(ind_id):
                if self.graph.nodes[owned_business_id].get("node_type") == NodeType.BUSINESS:
                    # Check if the edge from ind_id to owned_business_id is 'OWNS'
                    # This requires checking edge attributes, assuming 'OWNS' implies ownership relation.
                    # For simplicity, we assume any business neighbor is owned for now.
                    # A more robust check would be: self.graph.edges[ind_id, owned_business_id].get('edge_type') == EdgeType.OWNERSHIP
                    for acc_node in self.graph.neighbors(owned_business_id):
                        if self.graph.nodes[acc_node].get("node_type") == NodeType.ACCOUNT:
                            owned_accounts.add(acc_node)

            if len(owned_accounts) >= min_accounts_for_pattern:
                central_individual_id = ind_id
                peripheral_account_ids = list(owned_accounts)
                break

        if not central_individual_id or not peripheral_account_ids:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        return EntitySelection(
            central_entities=[central_individual_id],  # Individual is central
            peripheral_entities=peripheral_account_ids,  # Their accounts are peripheral
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
