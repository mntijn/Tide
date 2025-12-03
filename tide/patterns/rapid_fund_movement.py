from ..utils.random_instance import random_instance
import datetime
from typing import List, Dict, Any, Tuple
import logging

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType
from ..datastructures.attributes import TransactionAttributes
from ..utils.constants import HIGH_RISK_COUNTRIES

logger = logging.getLogger(__name__)


class IndividualWithMultipleAccountsStructural(StructuralComponent):
    """
    Structural: An individual with multiple accounts (personal and/or business) as receiver,
    and multiple overseas entities from high-risk jurisdictions as senders.
    """

    @property
    def num_required_entities(self) -> int:
        return 1  # One individual with multiple accounts

    def _is_high_risk_jurisdiction(self, country_code: str) -> bool:
        """Check if a country code represents a high-risk jurisdiction."""
        return country_code in HIGH_RISK_COUNTRIES

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        # Load pattern configuration
        pattern_config = self.params.get(
            "pattern_config", {}).get("rapidMovement", {})

        min_accounts_for_pattern = pattern_config.get(
            "min_accounts_for_pattern", 2)

        # Get sender configuration from inflow params (each sender typically sends one inflow)
        tx_params = pattern_config.get("transaction_params", {})
        inflow_params = tx_params.get("inflows", {})
        min_senders = inflow_params.get("min_inflows", 2)
        max_senders = inflow_params.get("max_inflows", 7)
        max_sender_entities = pattern_config.get("max_sender_entities", 20)

        logger.debug(
            f"RapidFundMovement: min_accounts_for_pattern: {min_accounts_for_pattern}, min_senders: {min_senders}, max_senders: {max_senders}")

        logger.debug(
            f"RapidFundMovement: Looking for individual with {min_accounts_for_pattern}+ accounts, {min_senders}-{max_senders} senders from max {max_sender_entities} entities")

        # Get all individuals from pre-computed all_nodes (efficient!)
        all_individuals = list(
            self.graph_generator.all_nodes.get(NodeType.INDIVIDUAL, []))

        # Use mixed selection: ~65% high-risk, ~35% general population
        # This prevents country_code from being perfectly predictive of fraud
        receiver_clusters = ["super_high_risk", "high_risk_score",
                             "structuring_candidates", "high_risk_countries"]
        potential_receivers = self.get_mixed_risk_entities(
            high_risk_clusters=receiver_clusters,
            fallback_pool=all_individuals,
            num_needed=max(25, len(all_individuals) // 10),
            high_risk_ratio=0.65,
            node_type_filter=NodeType.INDIVIDUAL
        )

        logger.debug(
            f"RapidFundMovement: Selected {len(potential_receivers)} potential receivers (mixed risk)")

        random_instance.shuffle(potential_receivers)

        # Find an individual with multiple accounts
        selected_individual = None
        for individual_id in potential_receivers:
            owned_accounts = self._get_owned_accounts(individual_id)

            if len(owned_accounts) >= min_accounts_for_pattern:
                selected_individual = individual_id
                logger.debug(
                    f"RapidFundMovement: Selected individual {individual_id} with {len(owned_accounts)} accounts")
                break

        if not selected_individual:
            logger.warning(
                "RapidFundMovement: No individual found with sufficient accounts")
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Get individual's accounts to exclude them from sender selection
        individual_accounts = self._get_owned_accounts(selected_individual)

        # Get all entities (excluding the selected individual) from pre-computed all_nodes
        all_individuals = list(
            self.graph_generator.all_nodes.get(NodeType.INDIVIDUAL, []))
        all_businesses = list(
            self.graph_generator.all_nodes.get(NodeType.BUSINESS, []))
        all_entities = [e for e in (
            all_individuals + all_businesses) if e != selected_individual]

        # Use mixed selection for senders: ~65% from high-risk, ~35% general
        sender_clusters = ["high_risk_countries",
                           "offshore_candidates", "super_high_risk"]
        mixed_sender_entities = self.get_mixed_risk_entities(
            high_risk_clusters=sender_clusters,
            fallback_pool=all_entities,
            num_needed=max_sender_entities,
            high_risk_ratio=0.65
        )

        logger.debug(
            f"RapidFundMovement: Selected {len(mixed_sender_entities)} sender entities (mixed risk)")

        # Get accounts owned by these entities
        from .base import deduplicate_preserving_order
        potential_sender_accounts = []
        for entity_id in mixed_sender_entities:
            entity_accounts = self._get_owned_accounts(entity_id)
            potential_sender_accounts.extend(entity_accounts)

        # Remove any accounts owned by the selected individual
        potential_sender_accounts = [acc for acc in potential_sender_accounts
                                     if acc not in individual_accounts]

        # Remove duplicates
        potential_sender_accounts = deduplicate_preserving_order(
            potential_sender_accounts)

        logger.debug(
            f"RapidFundMovement: Found {len(potential_sender_accounts)} potential sender accounts")

        if not potential_sender_accounts:
            logger.warning("RapidFundMovement: No sender accounts found")
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Select sender accounts based on configuration
        num_senders = random_instance.randint(min_senders, max_senders)
        random_instance.shuffle(potential_sender_accounts)
        selected_senders = potential_sender_accounts[:min(
            num_senders, len(potential_sender_accounts))]

        logger.debug(
            f"RapidFundMovement: Selected {len(selected_senders)} sender accounts")

        # Only include the central individual and sender accounts in the entity selection
        # The individual's accounts will be used in the temporal component but aren't part of the entity selection
        return EntitySelection(
            central_entities=[selected_individual],
            peripheral_entities=selected_senders
        )


class RapidInflowOutflowTemporal(TemporalComponent):
    """
    Temporal: High volume of incoming transfers, then high frequency of cash withdrawals.
    Withdrawals can be across multiple banks and structured.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []
        if not entity_selection.central_entities or not entity_selection.peripheral_entities:
            logger.debug(
                f"RapidFundMovement: Missing entities - central: {len(entity_selection.central_entities)}, peripheral: {len(entity_selection.peripheral_entities)}")
            return sequences

        central_individual_id = entity_selection.central_entities[0]
        overseas_sender_accounts = entity_selection.peripheral_entities

        # Get the individual's owned accounts for receiving funds and withdrawals
        individual_accounts = self._get_owned_accounts(central_individual_id)
        if not individual_accounts:  # Need accounts to operate on
            logger.warning(
                f"RapidFundMovement: Individual {central_individual_id} has no accounts")
            return sequences

        logger.debug(
            f"RapidFundMovement: Generating transactions for individual {central_individual_id} with {len(individual_accounts)} accounts and {len(overseas_sender_accounts)} senders")

        # Select accounts that will receive inflows
        receiving_accounts = individual_accounts.copy()

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

        # inflow_to_withdrawal_delay_hours is also in YAML, e.g. [1, 24]
        delay_hours_range = tx_params.get(
            "inflow_to_withdrawal_delay", [1, 24])

        num_incoming_transfers = random_instance.randint(
            min_inflows, max_inflows)
        num_withdrawals = random_instance.randint(
            min_withdrawals, max_withdrawals)

        # Use the overseas sender accounts from entity selection
        if not overseas_sender_accounts:
            logger.warning(
                "RapidFundMovement: No overseas sender accounts for RapidInflowOutflowTemporal inflows.")
            return sequences

        if (self.time_span["end_date"] - self.time_span["start_date"]).days < 10:
            start_day_offset_range_inflow = max(
                0, (self.time_span["end_date"] - self.time_span["start_date"]).days - 1)
        else:
            start_day_offset_range_inflow = (
                self.time_span["end_date"] - self.time_span["start_date"]).days - 10

        inflow_start_time = self.time_span["start_date"] + datetime.timedelta(
            days=random_instance.randint(0, start_day_offset_range_inflow)
        )
        inflow_timestamps = self.generate_timestamps(
            inflow_start_time, "high_frequency", num_incoming_transfers)

        # Get currency for structuring (sample from receiving accounts)
        sample_account = random_instance.choice(receiving_accounts)
        inflow_account_currency = self.graph.nodes[sample_account].get(
            "currency", "EUR")

        inflow_amounts = self.generate_structured_amounts(
            count=num_incoming_transfers,
            base_amount=round(random_instance.uniform(
                inflow_amount_range[0], inflow_amount_range[1]), 2),
            target_currency=inflow_account_currency
        )

        # Cap inflow amounts to respect the configured range
        inflow_amounts = [min(max(amount, inflow_amount_range[0]), inflow_amount_range[1])
                          for amount in inflow_amounts]

        inflow_transactions = []
        inflow_duration = datetime.timedelta(days=0)
        if inflow_timestamps:
            inflow_duration = inflow_timestamps[-1] - inflow_timestamps[0]

        # Track which accounts receive money and how much
        account_balances = {acc: 0.0 for acc in receiving_accounts}

        for i in range(min(num_incoming_transfers, len(inflow_timestamps))):
            # Select from overseas sender accounts (entity selection peripheral entities)
            source_overseas_account = random_instance.choice(
                overseas_sender_accounts)

            # Select destination account from receiving accounts (distribute across accounts)
            dest_account = random_instance.choice(receiving_accounts)

            # Track the money going into this account
            account_balances[dest_account] += inflow_amounts[i]

            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=source_overseas_account,
                dest_id=dest_account,
                timestamp=inflow_timestamps[i],
                amount=inflow_amounts[i],
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=True
            )
            inflow_transactions.append(
                (source_overseas_account, dest_account, tx_attrs))

        if inflow_transactions:
            sequences.append(TransactionSequence(
                transactions=inflow_transactions,
                sequence_name="rapid_inflows",
                start_time=inflow_timestamps[0] if inflow_timestamps else inflow_start_time,
                duration=inflow_duration
            ))

        last_inflow_time = inflow_timestamps[-1] if inflow_timestamps else inflow_start_time
        withdrawal_start_time = last_inflow_time + \
            datetime.timedelta(hours=random_instance.uniform(
                delay_hours_range[0], delay_hours_range[1]))

        withdrawal_timestamps = self.generate_timestamps(
            withdrawal_start_time, "high_frequency", num_withdrawals)

        # Calculate total inflow amount for proper outflow ratio
        total_inflow = sum(inflow_amounts)

        # Get validation parameters to determine target outflow ratio
        validation_params = pattern_config.get('validation_params', {})
        outflow_ratio_range = validation_params.get(
            'outflow_ratio_range', [0.85, 0.95])
        target_outflow_ratio = random_instance.uniform(
            outflow_ratio_range[0], outflow_ratio_range[1])

        # Calculate target total withdrawal amount based on inflow
        target_total_withdrawal = total_inflow * target_outflow_ratio

        # Generate structured withdrawal amounts (below reporting thresholds)
        # Use a base amount that's reasonable for the withdrawal range
        base_withdrawal_amount = (
            inflow_amount_range[0] + inflow_amount_range[1]) / 2

        withdrawal_amounts = self.generate_structured_amounts(
            count=num_withdrawals,
            base_amount=base_withdrawal_amount,
            target_currency=inflow_account_currency
        )

        # Scale all amounts proportionally to match the target total
        current_total = sum(withdrawal_amounts)
        if current_total > 0:
            scale_factor = target_total_withdrawal / current_total
            withdrawal_amounts = [
                amount * scale_factor for amount in withdrawal_amounts]

        # Round all amounts
        withdrawal_amounts = [round(amount, 2)
                              for amount in withdrawal_amounts]

        withdrawal_transactions = []
        withdrawal_duration = datetime.timedelta(days=0)
        if withdrawal_timestamps:
            withdrawal_duration = withdrawal_timestamps[-1] - \
                withdrawal_timestamps[0]

        # Use the global cash account ID from graph_generator
        cash_account_id = self.graph_generator.cash_account_id

        if not cash_account_id:
            logger.warning(
                "RapidFundMovement: Global cash_account_id not found for RapidInflowOutflowTemporal withdrawals.")
            if inflow_transactions and not withdrawal_transactions:  # if only inflows happened
                return sequences  # Or return only inflow sequence
            return []  # No transactions can be made if cash account is missing for withdrawal step

        # Only withdraw from accounts that actually received money
        accounts_with_money = [
            acc for acc, balance in account_balances.items() if balance > 0]

        if not accounts_with_money:
            logger.warning(
                "RapidFundMovement: No accounts received money, cannot generate withdrawals")
            return sequences

        for i in range(min(num_withdrawals, len(withdrawal_timestamps))):
            # Only withdraw from accounts that received inflows
            account_for_withdrawal = random_instance.choice(
                accounts_with_money)

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

        logger.debug(
            f"RapidFundMovement: Generated {len(sequences)} sequences with {sum(len(seq.transactions) for seq in sequences)} total transactions")
        return sequences


class RapidFundMovementPattern(CompositePattern):
    """Injects rapid fund movement pattern"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = IndividualWithMultipleAccountsStructural(
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
