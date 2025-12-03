import datetime
from typing import List, Dict, Any, Tuple
import logging

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType
from ..datastructures.attributes import TransactionAttributes
from ..utils.random_instance import random_instance

logger = logging.getLogger(__name__)


class UTurnTransactionsStructural(StructuralComponent):
    """
    Structural: An originator, intermediary accounts (often in high-risk jurisdictions),
    and accounts linked to the originator for funds to return to.
    """

    @property
    def num_required_entities(self) -> int:
        # Need at least 1 originator + 2 intermediaries
        return 3

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        logger.debug(
            "Starting entity selection for U-Turn Transactions pattern")
        pattern_config = self.params.get(
            "pattern_config", {}).get("uTurnTransactions", {})
        min_intermediaries = pattern_config.get("min_intermediaries", 2)
        max_intermediaries = pattern_config.get("max_intermediaries", 5)
        logger.debug(
            f"Pattern config - min_intermediaries: {min_intermediaries}, max_intermediaries: {max_intermediaries}")

        # Find originator (individual or business with multiple accounts)
        originator_id = None
        originator_accounts = []
        return_account_id = None

        # Get all individuals and businesses from pre-computed all_nodes
        all_individuals = list(
            self.graph_generator.all_nodes.get(NodeType.INDIVIDUAL, []))
        all_businesses = list(
            self.graph_generator.all_nodes.get(NodeType.BUSINESS, []))
        all_entities = all_individuals + all_businesses

        # Use mixed selection: ~65% high-risk, ~35% general population
        originator_clusters = ["super_high_risk", "high_risk_score",
                               "structuring_candidates", "high_risk_countries"]
        mixed_originators = self.get_mixed_risk_entities(
            high_risk_clusters=originator_clusters,
            fallback_pool=all_entities,
            num_needed=max(30, len(all_entities) // 10),
            high_risk_ratio=0.65
        )

        # Find entities with multiple accounts
        potential_originators = []
        for entity_id in mixed_originators:
            owned_accounts = sorted([
                n for n in self.graph.neighbors(entity_id)
                if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
            ])
            if len(owned_accounts) >= 2:  # Need at least 2 accounts
                potential_originators.append((entity_id, owned_accounts))
                logger.debug(
                    f"Found potential originator {entity_id} with {len(owned_accounts)} accounts")

        # Shuffle and try to find suitable originator
        random_instance.shuffle(potential_originators)

        for entity_id, accounts in potential_originators:
            # Use one account for sending, another for receiving
            if len(accounts) >= 2:
                originator_id = entity_id
                # First for sending, second for receiving
                originator_accounts = accounts[:2]
                return_account_id = accounts[1]
                break

        if not originator_id:
            logger.debug(
                "No suitable originator found in mixed selection, trying all entities fallback")
            # Fallback: any entity with 2+ accounts
            random_instance.shuffle(all_entities)

            for entity_id in all_entities:
                owned_accounts = sorted([
                    n for n in self.graph.neighbors(entity_id)
                    if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
                ])
                if len(owned_accounts) >= 2:
                    originator_id = entity_id
                    originator_accounts = owned_accounts[:2]
                    return_account_id = owned_accounts[1]
                    logger.debug(
                        f"Selected fallback originator {originator_id} with accounts {originator_accounts}")
                    break

        if not originator_id:
            logger.warning(
                "Failed to find any suitable originator, returning empty selection")
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Find intermediary accounts using mixed selection
        intermediary_accounts = []
        originator_country = self.graph.nodes[originator_id].get(
            "country_code")

        # Get all entities (excluding originator) for intermediary selection
        all_other_entities = [e for e in all_entities if e != originator_id]

        # Use mixed selection for intermediaries: ~65% from high-risk, ~35% general
        intermediary_clusters = ["high_risk_countries", "offshore_candidates"]
        mixed_intermediary_entities = self.get_mixed_risk_entities(
            high_risk_clusters=intermediary_clusters,
            fallback_pool=all_other_entities,
            num_needed=max_intermediaries * 3,  # Get more, then filter
            high_risk_ratio=0.65
        )

        # Get accounts from these entities (excluding originator's country)
        high_risk_accounts = []
        for entity_id in mixed_intermediary_entities:
            entity_accounts = self._get_owned_accounts(entity_id)
            for acc_id in entity_accounts:
                acc_country = self.graph.nodes[acc_id].get("country_code")
                if acc_country and acc_country != originator_country:
                    high_risk_accounts.append(acc_id)

        # Deduplicate and shuffle
        potential_intermediaries = sorted(list(set(high_risk_accounts)))
        random_instance.shuffle(potential_intermediaries)

        # Select intermediaries
        if len(potential_intermediaries) >= min_intermediaries:
            num_to_select = random_instance.randint(min_intermediaries,
                                                    min(len(potential_intermediaries), max_intermediaries))
            intermediary_accounts = potential_intermediaries[:num_to_select]
        else:
            intermediary_accounts = []

        if len(intermediary_accounts) < min_intermediaries:
            logger.debug(
                "Not enough cross-country intermediaries found, trying fallback with all accounts")
            # Fallback: allow same-country intermediaries if no cross-country found
            all_accounts = list(
                self.graph_generator.all_nodes.get(NodeType.ACCOUNT, []))
            potential_intermediaries = [
                acc for acc in all_accounts if acc not in originator_accounts]
            # Exclude originator's own accounts
            potential_intermediaries = [
                acc for acc in potential_intermediaries if acc not in originator_accounts]
            random_instance.shuffle(potential_intermediaries)

            if len(potential_intermediaries) >= min_intermediaries:
                num_to_select = random_instance.randint(
                    min_intermediaries, min(len(potential_intermediaries), max_intermediaries))
                intermediary_accounts = potential_intermediaries[:num_to_select]
            else:
                intermediary_accounts = []

        if len(intermediary_accounts) < min_intermediaries:
            logger.warning(
                f"Failed to find enough intermediaries (found {len(intermediary_accounts)}, need {min_intermediaries})")
            return EntitySelection(central_entities=[], peripheral_entities=[])

        return EntitySelection(
            # Send from first, return to second
            central_entities=[originator_accounts[0], return_account_id],
            peripheral_entities=intermediary_accounts
        )


class UTurnTransactionsTemporal(TemporalComponent):
    """
    Temporal: Funds flow out through intermediaries and return (partially) to originator.
    International transfers may take 1-5 business days.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []

        if not entity_selection.central_entities or len(entity_selection.central_entities) < 2:
            return sequences

        source_account_id = entity_selection.central_entities[0]
        return_account_id = entity_selection.central_entities[1]
        intermediary_accounts = entity_selection.peripheral_entities

        if not intermediary_accounts:
            return sequences

        pattern_config = self.params.get(
            "pattern_config", {}).get("uTurnTransactions", {})
        tx_params = pattern_config.get("transaction_params", {})
        time_variability = tx_params.get("time_variability", {
            "business_hours": [9, 16],
            "include_minutes": True,
            "include_hours": True
        })

        initial_amount_range = tx_params.get(
            "initial_amount_range", [10000, 100000])
        return_percentage_range = tx_params.get(
            "return_percentage_range", [0.7, 0.9])
        international_delay_days = tx_params.get(
            "international_delay_days", [1, 5])

        # Get time variability parameters
        business_hours = time_variability.get("business_hours", [9, 16])
        include_minutes = time_variability.get("include_minutes", True)
        include_hours = time_variability.get("include_hours", True)

        # Calculate start time
        time_span_days = (
            self.time_span["end_date"] - self.time_span["start_date"]).days
        # Need enough time for full cycle
        max_start_offset = max(0, time_span_days -
                               len(intermediary_accounts) * 5 - 10)
        start_day_offset = random_instance.randint(0, max_start_offset)

        # Start with a random time of day within business hours
        start_hour = random_instance.randint(
            business_hours[0], business_hours[1])
        start_minute = random_instance.randint(0, 59) if include_minutes else 0
        current_time = self.time_span["start_date"] + \
            datetime.timedelta(days=start_day_offset,
                               hours=start_hour, minutes=start_minute)

        # Initial amount
        initial_amount = random_instance.uniform(
            initial_amount_range[0], initial_amount_range[1])
        current_amount = initial_amount

        all_transactions = []

        # Create path through intermediaries
        path = [source_account_id] + intermediary_accounts

        # Add transactions along the path
        for i in range(len(path) - 1):
            src = path[i]
            dest = path[i + 1]

            # International transfer delay with configurable time variability
            base_delay_days = random_instance.uniform(
                international_delay_days[0], international_delay_days[1] - 0.1)  # Subtract 0.1 to avoid hitting exactly 5.0

            # Add random hours if enabled
            delay_hours = random_instance.randint(
                0, 23) if include_hours else 0
            # Add random minutes if enabled
            delay_minutes = random_instance.randint(
                0, 59) if include_minutes else 0

            # Ensure total delay doesn't exceed the maximum configured days
            total_delay_hours = base_delay_days * 24 + delay_hours
            max_allowed_hours = international_delay_days[1] * 24
            if total_delay_hours > max_allowed_hours:
                # Adjust to stay within bounds
                total_delay_hours = max_allowed_hours
                base_delay_days = total_delay_hours // 24
                delay_hours = total_delay_hours % 24

            # For boundary cases, ensure we're strictly less than the max
            total_delay_seconds = base_delay_days * 24 * \
                3600 + delay_hours * 3600 + delay_minutes * 60
            max_delay_seconds = international_delay_days[1] * 24 * 3600
            if total_delay_seconds >= max_delay_seconds:
                # Reduce by a small amount to stay strictly under the boundary
                total_delay_seconds = max_delay_seconds - 60  # 1 minute less
                base_delay_days = total_delay_seconds // (24 * 3600)
                remaining_seconds = total_delay_seconds % (24 * 3600)
                delay_hours = remaining_seconds // 3600
                delay_minutes = (remaining_seconds % 3600) // 60

            current_time += datetime.timedelta(
                days=base_delay_days,
                hours=delay_hours,
                minutes=delay_minutes
            )

            # Small fee/loss at each hop
            fee_percentage = random_instance.uniform(0.01, 0.03)
            current_amount *= (1 - fee_percentage)

            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=src,
                dest_id=dest,
                timestamp=current_time,
                amount=round(current_amount, 2),
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=True
            )

            all_transactions.append((src, dest, tx_attrs))

        # Return transaction from last intermediary to return account
        if intermediary_accounts:
            last_intermediary = intermediary_accounts[-1]

            # Delay before return with configurable time variability
            base_delay_days = random_instance.uniform(
                international_delay_days[0], international_delay_days[1] - 0.1)  # Subtract 0.1 to avoid hitting exactly 5.0

            # Add random hours if enabled
            delay_hours = random_instance.randint(
                0, 23) if include_hours else 0
            # Add random minutes if enabled
            delay_minutes = random_instance.randint(
                0, 59) if include_minutes else 0

            # Ensure total delay doesn't exceed the maximum configured days
            total_delay_hours = base_delay_days * 24 + delay_hours
            max_allowed_hours = international_delay_days[1] * 24
            if total_delay_hours > max_allowed_hours:
                # Adjust to stay within bounds
                total_delay_hours = max_allowed_hours
                base_delay_days = total_delay_hours // 24
                delay_hours = total_delay_hours % 24

            # For boundary cases, ensure we're strictly less than the max
            total_delay_seconds = base_delay_days * 24 * \
                3600 + delay_hours * 3600 + delay_minutes * 60
            max_delay_seconds = international_delay_days[1] * 24 * 3600
            if total_delay_seconds >= max_delay_seconds:
                # Reduce by a small amount to stay strictly under the boundary
                total_delay_seconds = max_delay_seconds - 60  # 1 minute less
                base_delay_days = total_delay_seconds // (24 * 3600)
                remaining_seconds = total_delay_seconds % (24 * 3600)
                delay_hours = remaining_seconds // 3600
                delay_minutes = (remaining_seconds % 3600) // 60

            current_time += datetime.timedelta(
                days=base_delay_days,
                hours=delay_hours,
                minutes=delay_minutes
            )

            # Return percentage of funds
            return_percentage = random_instance.uniform(
                return_percentage_range[0], return_percentage_range[1])
            return_amount = current_amount * return_percentage

            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=last_intermediary,
                dest_id=return_account_id,
                timestamp=current_time,
                amount=round(return_amount, 2),
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=True
            )

            all_transactions.append(
                (last_intermediary, return_account_id, tx_attrs))

        if all_transactions:
            start_time = all_transactions[0][2].timestamp
            end_time = all_transactions[-1][2].timestamp
            sequences.append(TransactionSequence(
                transactions=all_transactions,
                sequence_name="u_turn_cycle",
                start_time=start_time,
                duration=end_time - start_time
            ))
        return sequences


class UTurnTransactionsPattern(CompositePattern):
    """Injects U-Turn transactions pattern"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = UTurnTransactionsStructural(
            graph_generator, params)
        temporal_component = UTurnTransactionsTemporal(graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "UTurnTransactions"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
