from ..utils.random_instance import random_instance
import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType
from ..utils.amount_interleaving import generate_fraud_with_camouflage
import logging


logger = logging.getLogger(__name__)


class RepeatedOverseasTransfersStructural(StructuralComponent):
    """
    Selects:
    - A high-risk individual (central) - always an individual.
    - One of their domestic accounts (peripheral).
    - Multiple overseas accounts of individuals or businesses, prioritizing high-risk jurisdictions (peripheral).
    """

    @property
    def num_required_entities(self) -> int:
        return 1

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        central_owner_id = None
        source_account_id = None
        overseas_account_ids = []

        pattern_config = self.params.get(
            "pattern_config", {}).get("repeatedOverseas", {})
        min_overseas_entities = pattern_config.get("min_overseas_entities", 2)
        max_overseas_entities = pattern_config.get("max_overseas_entities", 5)

        # Get all individuals from pre-computed all_nodes
        all_individuals = list(
            self.graph_generator.all_nodes.get(NodeType.INDIVIDUAL, []))

        # Use mixed selection: ~40% high-risk, ~60% general population
        # Reduced from 65% to prevent risk_score from being predictive of fraud
        source_clusters = ["offshore_candidates",
                           "super_high_risk", "high_risk_countries"]
        potential_source_entities = self.get_mixed_risk_entities(
            high_risk_clusters=source_clusters,
            fallback_pool=all_individuals,
            num_needed=max(20, len(all_individuals) // 10),
            # high_risk_ratio read from fraud_selection_config in graph config
            node_type_filter=NodeType.INDIVIDUAL
        )

        if not potential_source_entities:
            # Fallback to all individuals if mixed selection fails
            potential_source_entities = all_individuals.copy()
            random_instance.shuffle(potential_source_entities)

        for entity_id in potential_source_entities:
            try:
                entity_node_data = self.graph.nodes[entity_id]
                entity_type = entity_node_data.get("node_type")

                # Only consider individuals for the central entity
                if entity_type != NodeType.INDIVIDUAL:
                    continue

                entity_country = entity_node_data.get("country_code")
                if not entity_country:
                    continue

                # Find domestic accounts owned by this entity
                owned_domestic_accounts = []
                for acc_id in sorted(self.graph.neighbors(entity_id)):
                    try:
                        acc_data = self.graph.nodes[acc_id]
                        if (acc_data.get("node_type") == NodeType.ACCOUNT and
                                acc_data.get("country_code") == entity_country):
                            owned_domestic_accounts.append(acc_id)
                    except KeyError:
                        continue

                if not owned_domestic_accounts:
                    continue

                source_account_id = random_instance.choice(
                    owned_domestic_accounts)

                # Search for overseas destination accounts using mixed selection
                # Get all entities (excluding source) from pre-computed all_nodes
                all_individuals_dest = list(
                    self.graph_generator.all_nodes.get(NodeType.INDIVIDUAL, []))
                all_businesses_dest = list(
                    self.graph_generator.all_nodes.get(NodeType.BUSINESS, []))
                all_other_entities = [e for e in (
                    all_individuals_dest + all_businesses_dest) if e != entity_id]

                # Use mixed selection: ~40% from high-risk countries, ~60% general
                # Reduced from 65% to prevent risk_score from being predictive of fraud
                dest_clusters = ["high_risk_countries", "offshore_candidates"]
                mixed_dest_entities = self.get_mixed_risk_entities(
                    high_risk_clusters=dest_clusters,
                    fallback_pool=all_other_entities,
                    num_needed=max_overseas_entities * 3  # Get more, then filter
                )

                # Get overseas accounts from these entities
                potential_overseas_accounts = []
                for owner_id in mixed_dest_entities:
                    try:
                        owner_country = self.graph.nodes[owner_id].get(
                            "country_code")
                        if not owner_country or owner_country == entity_country:
                            continue

                        owned_accounts = self._get_owned_accounts(owner_id)
                        for acc_id in owned_accounts:
                            acc_country = self.graph.nodes[acc_id].get(
                                "country_code")
                            if acc_country and acc_country != entity_country:
                                potential_overseas_accounts.append(acc_id)
                    except KeyError:
                        continue

                # If still not enough, expand to any overseas account
                if len(potential_overseas_accounts) < min_overseas_entities:
                    # Sort nodes for deterministic order
                    all_account_ids = [nid for nid, ndata in sorted(self.graph.nodes(
                        data=True)) if ndata.get("node_type") == NodeType.ACCOUNT]
                    random_instance.shuffle(all_account_ids)

                    for acc_id in all_account_ids:
                        if len(potential_overseas_accounts) >= max_overseas_entities * 2:
                            break
                        if acc_id in potential_overseas_accounts:
                            continue

                        try:
                            acc_node_data = self.graph.nodes[acc_id]
                            acc_country = acc_node_data.get("country_code")
                            if acc_country and acc_country != entity_country:
                                for owner_id in sorted(self.graph.predecessors(acc_id)):
                                    owner_data = self.graph.nodes[owner_id]
                                    if owner_data.get("node_type") in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
                                        potential_overseas_accounts.append(
                                            acc_id)
                                        break
                        except KeyError:
                            continue

                if len(potential_overseas_accounts) < min_overseas_entities:
                    continue  # Not enough suitable overseas accounts, try next source

                central_owner_id = entity_id
                num_to_select = random_instance.randint(
                    min_overseas_entities,
                    min(len(potential_overseas_accounts),
                        max_overseas_entities)
                )
                overseas_account_ids = random_instance.sample(
                    potential_overseas_accounts, num_to_select
                )
                break  # Found a valid set of entities

            except KeyError:
                continue  # Skip entities with missing data

        if not central_owner_id or not source_account_id or not overseas_account_ids:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        return EntitySelection(
            central_entities=[central_owner_id],
            peripheral_entities=[source_account_id] + overseas_account_ids,
        )


class FrequentOrPeriodicTransfersTemporal(TemporalComponent):
    """
    Generates transactions in two phases:
    1. Initial structured cash deposits INTO the source account.
    2. A sequence of transfers FROM the source account to overseas accounts,
       following either a high-frequency or periodic pattern.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []
        if not entity_selection.central_entities or not entity_selection.peripheral_entities or len(entity_selection.peripheral_entities) < 2:
            return sequences

        central_owner_id = entity_selection.central_entities[0]
        source_account_id = entity_selection.peripheral_entities[0]
        overseas_account_ids = entity_selection.peripheral_entities[1:]

        if not overseas_account_ids:
            return sequences

        pattern_config = self.params.get(
            "pattern_config", {}).get("repeatedOverseas", {})

        # --- Phase 1: Initial Funding (Cash Deposits) ---
        deposit_params = pattern_config.get("deposit_params", {})
        num_deposits = random_instance.randint(
            deposit_params.get("min_deposits", 5),
            deposit_params.get("max_deposits", 15)
        )
        deposit_amount_range = deposit_params.get(
            "amount_range", [8500, 9900])

        time_span_days = (
            self.time_span["end_date"] - self.time_span["start_date"]).days
        start_day_offset_range = max(
            0, time_span_days - 45) if time_span_days > 45 else 0

        deposit_start_time = self.time_span["start_date"] + datetime.timedelta(
            days=random_instance.randint(0, start_day_offset_range)
        )

        deposit_timestamps = self.generate_timestamps(
            deposit_start_time, "high_frequency", num_deposits
        )
        if not deposit_timestamps:
            return []  # Not enough time to even make deposits

        # NEW: Use camouflage amounts to lower ROC-AUC
        # Some deposits should look like average user deposits
        camouflage_probability = deposit_params.get(
            "camouflage_probability", 0.30)

        if camouflage_probability > 0:
            deposit_amounts = generate_fraud_with_camouflage(
                count=len(deposit_timestamps),
                camouflage_probability=camouflage_probability,
                small_amount_range=(100.0, 500.0),
                medium_amount_range=(500.0, 3000.0),
                high_amount_range=(
                    deposit_amount_range[0], deposit_amount_range[1]),
            )
        else:
            deposit_amounts = [round(random_instance.uniform(
                deposit_amount_range[0], deposit_amount_range[1]), 2)
                for _ in range(len(deposit_timestamps))]

        deposit_transactions = []
        for i in range(len(deposit_timestamps)):
            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=central_owner_id,
                dest_id=source_account_id,
                timestamp=deposit_timestamps[i],
                amount=deposit_amounts[i],
                transaction_type=TransactionType.DEPOSIT,
                is_fraudulent=True
            )
            deposit_transactions.append(
                (central_owner_id, source_account_id, tx_attrs))

        total_deposited = sum(deposit_amounts)
        if deposit_transactions:
            sequences.append(
                TransactionSequence(
                    transactions=deposit_transactions,
                    sequence_name="initial_cash_deposits",
                    start_time=deposit_timestamps[0],
                    duration=deposit_timestamps[-1] - deposit_timestamps[0]
                )
            )
        else:
            return []  # Should not happen if timestamps were generated, but as a safeguard

        # --- Phase 2: Overseas Transfers ---
        tx_params = pattern_config.get("transaction_params", {})
        min_tx = tx_params.get("min_transactions", 10)
        max_tx = tx_params.get("max_transactions", 30)
        amount_range = tx_params.get("transfer_amount_range", [5000, 20000])
        interval_days_options = tx_params.get(
            "transfer_interval_days", [7, 14, 30])

        temporal_type = random_instance.choice(
            ["periodic"])
        num_transactions = random_instance.randint(min_tx, max_tx)

        # Transfers start after a short delay following the last deposit
        transfer_start_delay = datetime.timedelta(
            days=random_instance.randint(1, 5))
        base_start_time = deposit_timestamps[-1] + transfer_start_delay

        if base_start_time >= self.time_span["end_date"]:
            return sequences  # Not enough time for transfers, return only deposits

        transfer_timestamps = []
        if temporal_type == "periodic" and interval_days_options:
            interval_days = random_instance.choice(interval_days_options)
            current_time = base_start_time
            for _ in range(num_transactions):
                if current_time > self.time_span["end_date"]:
                    break
                transfer_timestamps.append(current_time)
                current_time += datetime.timedelta(days=interval_days)
        else:
            transfer_timestamps = self.generate_timestamps(
                base_start_time, temporal_type, num_transactions)

        if not transfer_timestamps:
            return sequences  # Not enough time for transfers

        actual_num_transactions = len(transfer_timestamps)

        # NEW: Use camouflage for transfer amounts too
        transfer_camouflage_prob = tx_params.get(
            "camouflage_probability", 0.25)

        if transfer_camouflage_prob > 0:
            raw_amounts = generate_fraud_with_camouflage(
                count=actual_num_transactions,
                camouflage_probability=transfer_camouflage_prob,
                small_amount_range=(100.0, 600.0),
                medium_amount_range=(600.0, 2500.0),
                high_amount_range=(amount_range[0], amount_range[1]),
            )
        else:
            raw_amounts = self.generate_structured_amounts(
                count=actual_num_transactions,
                base_amount=round(random_instance.uniform(
                    amount_range[0], amount_range[1]), 2),
                target_currency="EUR"
            )
            raw_amounts = [min(max(amount, amount_range[0]), amount_range[1])
                           for amount in raw_amounts]

        # Scale transfer amounts to match total_deposited (optionally allow a small leakage, e.g. 98%)
        transfer_leakage_ratio = pattern_config.get(
            "transfer_leakage_ratio", 0.98)
        target_total_transfer = total_deposited * transfer_leakage_ratio
        current_total = sum(raw_amounts)
        if current_total > 0:
            scale_factor = target_total_transfer / current_total
            amounts = [round(amount * scale_factor, 2)
                       for amount in raw_amounts]
            # Ensure all scaled amounts stay within the configured range
            amounts = [min(max(amount, amount_range[0]), amount_range[1])
                       for amount in amounts]
        else:
            amounts = raw_amounts

        # Create PatternInjector for layering
        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Build exclusion list
        exclude_accounts = [source_account_id] + overseas_account_ids

        transactions_for_sequence = []
        for i in range(actual_num_transactions):
            destination_account_id = random_instance.choice(
                overseas_account_ids)

            # === ADD LAYERING HOPS ===
            layering_txs, final_time, final_amount = self.generate_layering_transactions(
                source_account=source_account_id,
                dest_account=destination_account_id,
                amount=amounts[i],
                start_time=transfer_timestamps[i],
                pattern_injector=pattern_injector,
                exclude_accounts=exclude_accounts
            )

            # Add layering transactions
            transactions_for_sequence.extend(layering_txs)

            # Determine actual source for final transfer
            if layering_txs:
                actual_src = layering_txs[-1][1]
                final_timestamp = final_time
                tx_amount = final_amount
            else:
                actual_src = source_account_id
                final_timestamp = transfer_timestamps[i]
                tx_amount = amounts[i]

            tx_attrs = pattern_injector._create_transaction_edge(
                src_id=actual_src,
                dest_id=destination_account_id,
                timestamp=final_timestamp,
                amount=tx_amount,
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=True
            )
            transactions_for_sequence.append(
                (actual_src, destination_account_id, tx_attrs))

        if transactions_for_sequence:
            # Get actual start and end times
            start_time = transactions_for_sequence[0][2].timestamp
            end_time = transactions_for_sequence[-1][2].timestamp
            sequences.append(
                TransactionSequence(
                    transactions=transactions_for_sequence,
                    sequence_name=f"repeated_transfers_{temporal_type}",
                    start_time=start_time,
                    duration=end_time - start_time
                )
            )
        return sequences


class RepeatedOverseasTransfersPattern(CompositePattern):
    """Injects repeated overseas transfers pattern"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = RepeatedOverseasTransfersStructural(
            graph_generator, params)
        temporal_component = FrequentOrPeriodicTransfersTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "RepeatedOverseasTransfers"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
