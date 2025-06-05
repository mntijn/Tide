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

        logger.info(
            f"RapidFundMovement: min_accounts_for_pattern: {min_accounts_for_pattern}, min_senders: {min_senders}, max_senders: {max_senders}")

        logger.info(
            f"RapidFundMovement: Looking for individual with {min_accounts_for_pattern}+ accounts, {min_senders}-{max_senders} senders from max {max_sender_entities} entities")

        # Select high-risk individual as receiver from the most risky clusters
        receiver_clusters = ["super_high_risk",
                             "high_risk_score", "structuring_candidates"]
        potential_receivers = self.get_high_risk_individuals(
            cluster_names=receiver_clusters, max_entities=25)

        logger.debug(
            f"RapidFundMovement: Found {len(potential_receivers)} potential receivers from clusters")

        # Fallback to all individuals if clusters yield too few
        if len(potential_receivers) < 5:
            all_individuals = [e for e in available_entities
                               if self.graph.nodes[e].get("node_type") == NodeType.INDIVIDUAL]
            potential_receivers.extend(all_individuals)
            potential_receivers = list(dict.fromkeys(potential_receivers))
            logger.debug(
                f"RapidFundMovement: Expanded to {len(potential_receivers)} total potential receivers")

        random_instance.shuffle(potential_receivers)

        # Find an individual with multiple accounts
        selected_individual = None
        for individual_id in potential_receivers:
            owned_accounts = self._get_owned_accounts(individual_id)

            if len(owned_accounts) >= min_accounts_for_pattern:
                selected_individual = individual_id
                logger.info(
                    f"RapidFundMovement: Selected individual {individual_id} with {len(owned_accounts)} accounts")
                break

        if not selected_individual:
            logger.info(
                "RapidFundMovement: No individual found with sufficient accounts")
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Get individual's accounts to exclude them from sender selection
        individual_accounts = self._get_owned_accounts(selected_individual)

        # Select overseas sender entities from high-risk clusters
        sender_entity_clusters = ["high_risk_countries", "offshore_candidates"]
        potential_sender_entities = self.get_combined_clusters(
            sender_entity_clusters)

        logger.debug(
            f"RapidFundMovement: Found {len(potential_sender_entities)} potential sender entities from clusters")

        # Limit the number of sender entities to prevent too many accounts
        random_instance.shuffle(potential_sender_entities)
        limited_sender_entities = potential_sender_entities[:max_sender_entities]

        logger.debug(
            f"RapidFundMovement: Limited to {len(limited_sender_entities)} sender entities")

        # Get accounts owned by these limited high-risk entities
        potential_sender_accounts = []
        for entity_id in limited_sender_entities:
            entity_accounts = self._get_owned_accounts(entity_id)
            potential_sender_accounts.extend(entity_accounts)

        # Remove any accounts owned by the selected individual
        potential_sender_accounts = [acc for acc in potential_sender_accounts
                                     if acc not in individual_accounts]

        # Remove duplicates
        potential_sender_accounts = list(set(potential_sender_accounts))

        logger.debug(
            f"RapidFundMovement: Found {len(potential_sender_accounts)} potential sender accounts from high-risk entities")

        # If we don't have enough from clusters, expand to all accounts from high-risk countries
        if len(potential_sender_accounts) < min_senders:
            all_accounts = [e for e in available_entities
                            if self.graph.nodes[e].get("node_type") == NodeType.ACCOUNT]

            high_risk_accounts_added = 0
            for account_id in all_accounts:
                if account_id in individual_accounts:
                    continue

                account_country = self.graph.nodes[account_id].get(
                    "country_code", "US")
                if self._is_high_risk_jurisdiction(account_country):
                    potential_sender_accounts.append(account_id)
                    high_risk_accounts_added += 1

            logger.debug(
                f"RapidFundMovement: Added {high_risk_accounts_added} more accounts from high-risk countries")

        # Remove duplicates again
        potential_sender_accounts = list(set(potential_sender_accounts))

        if not potential_sender_accounts:
            logger.info("RapidFundMovement: No sender accounts found")
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Select sender accounts based on configuration
        num_senders = random_instance.randint(min_senders, max_senders)
        random_instance.shuffle(potential_sender_accounts)
        selected_senders = potential_sender_accounts[:min(
            num_senders, len(potential_sender_accounts))]

        logger.info(
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

    def _get_owned_accounts(self, entity_id: str) -> List[str]:
        """Get all accounts owned by an entity (individual or business)"""
        owned_accounts = []

        # Direct accounts
        for neighbor_id in self.graph.neighbors(entity_id):
            if self.graph.nodes[neighbor_id].get("node_type") == NodeType.ACCOUNT:
                owned_accounts.append(neighbor_id)

        # If entity is an individual, also check business accounts they own
        if self.graph.nodes[entity_id].get("node_type") == NodeType.INDIVIDUAL:
            for neighbor_id in self.graph.neighbors(entity_id):
                if self.graph.nodes[neighbor_id].get("node_type") == NodeType.BUSINESS:
                    # Get accounts of owned businesses
                    for business_neighbor in self.graph.neighbors(neighbor_id):
                        if self.graph.nodes[business_neighbor].get("node_type") == NodeType.ACCOUNT:
                            owned_accounts.append(business_neighbor)

        return list(set(owned_accounts))  # Remove duplicates

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []
        if not entity_selection.central_entities or not entity_selection.peripheral_entities:
            logger.info(
                f"RapidFundMovement: Missing entities - central: {len(entity_selection.central_entities)}, peripheral: {len(entity_selection.peripheral_entities)}")
            return sequences

        central_individual_id = entity_selection.central_entities[0]
        overseas_sender_accounts = entity_selection.peripheral_entities

        # Get the individual's owned accounts for receiving funds and withdrawals
        individual_accounts = self._get_owned_accounts(central_individual_id)
        if not individual_accounts:  # Need accounts to operate on
            logger.info(
                f"RapidFundMovement: Individual {central_individual_id} has no accounts")
            return sequences

        logger.info(
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
            logger.info(
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
            logger.info(
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

        logger.info(
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
