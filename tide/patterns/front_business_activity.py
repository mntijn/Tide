from ..utils.random_instance import random_instance
import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType
from ..datastructures.attributes import TransactionAttributes
from ..utils.amount_interleaving import generate_fraud_with_camouflage


class FrontBusinessStructural(StructuralComponent):
    """
    Structural: A business with multiple bank accounts at multiple banks.
    Connects to overseas business accounts.
    """

    @property
    def num_required_entities(self) -> int:
        return 2  # Business and an individual

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        central_business_id = None
        business_accounts_ids = []
        overseas_business_accounts_ids = []

        # Get all businesses from pre-computed all_nodes (efficient!)
        all_businesses = list(
            self.graph_generator.all_nodes.get(NodeType.BUSINESS, []))

        # Use mixed selection: ~40% high-risk, ~60% general population
        # Reduced from 65% to prevent risk_score from being predictive of fraud
        front_business_clusters = ["super_high_risk", "offshore_candidates",
                                   "high_risk_business_categories", "high_risk_countries"]
        potential_front_businesses = self.get_mixed_risk_entities(
            high_risk_clusters=front_business_clusters,
            fallback_pool=all_businesses,
            num_needed=max(20, len(all_businesses) // 5),
            # high_risk_ratio read from fraud_selection_config in graph config
            node_type_filter=NodeType.BUSINESS
        )

        if not potential_front_businesses:
            potential_front_businesses = all_businesses.copy()

        random_instance.shuffle(potential_front_businesses)

        # Default values for pattern parameters
        pattern_config = self.params.get(
            "pattern_config", {}).get("frontBusiness", {})
        min_bus_accounts = pattern_config.get(
            "min_accounts_for_front_business", 2)
        num_front_business_accounts_to_use = pattern_config.get(
            "num_front_business_accounts_to_use", 3)
        min_overseas_dest_bus_accounts = pattern_config.get(
            "min_overseas_destination_accounts", 2)
        max_overseas_bus_accounts_for_front = pattern_config.get(
            "max_overseas_destination_accounts_for_front", 4)

        for bus_id in potential_front_businesses:
            owned_accounts_data = [
                n for n in sorted(self.graph.neighbors(bus_id))
                if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
            ]

            if len(owned_accounts_data) >= min_bus_accounts:
                potential_dest_accounts = []
                # Search all accounts in the graph for overseas destinations
                all_graph_accounts = self.graph_generator.all_nodes.get(
                    NodeType.ACCOUNT, [])
                business_country = self.graph.nodes[bus_id].get(
                    "country_code")  # Note: updated to use country_code instead of address.country

                for acc_id in all_graph_accounts:
                    if acc_id in owned_accounts_data:  # Exclude accounts owned by the front business itself
                        continue

                    acc_node_data = self.graph.nodes[acc_id]
                    acc_country = acc_node_data.get(
                        "country_code")  # Updated field name

                    # Ensure account has a country and it's different from the front business's country
                    if acc_country and acc_country != business_country:
                        # Check if this account is owned by any business
                        is_owned_by_business = False
                        for owner_id in sorted(self.graph.predecessors(acc_id)):
                            if self.graph.nodes[owner_id].get("node_type") == NodeType.BUSINESS:
                                is_owned_by_business = True
                                break
                        if is_owned_by_business:
                            potential_dest_accounts.append(acc_id)

                if len(potential_dest_accounts) >= min_overseas_dest_bus_accounts:
                    central_business_id = bus_id
                    business_accounts_ids = random_instance.sample(owned_accounts_data, k=min(
                        len(owned_accounts_data), num_front_business_accounts_to_use))

                    num_overseas_to_select = random_instance.randint(
                        min_overseas_dest_bus_accounts,
                        min(len(potential_dest_accounts),
                            max_overseas_bus_accounts_for_front)
                    )
                    overseas_business_accounts_ids = random_instance.sample(
                        potential_dest_accounts, num_overseas_to_select)
                    break

        if not central_business_id or not business_accounts_ids or not overseas_business_accounts_ids:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        return EntitySelection(
            central_entities=[central_business_id],  # Business ID is central
            peripheral_entities=business_accounts_ids + overseas_business_accounts_ids,
        )


class FrequentCashDepositsAndOverseasTransfersTemporal(TemporalComponent):
    """
    Temporal: Frequent large cash deposits to business accounts,
    followed by immediate transfers to overseas business accounts.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []
        if not entity_selection.central_entities or not entity_selection.peripheral_entities:
            return sequences

        front_business_id = entity_selection.central_entities[0]

        front_business_accounts = []
        overseas_dest_accounts = []
        for acc_id in entity_selection.peripheral_entities:
            is_owned_by_front_business = any(
                owner == front_business_id for owner in self.graph.predecessors(acc_id))
            if is_owned_by_front_business:
                front_business_accounts.append(acc_id)
            else:
                overseas_dest_accounts.append(acc_id)

        if not front_business_accounts or not overseas_dest_accounts:
            return sequences

        pattern_config = self.params.get(
            "pattern_config", {}).get("frontBusiness", {})
        tx_params = pattern_config.get("transaction_params", {})
        min_deposits = tx_params.get("min_deposits", 5)
        max_deposits = tx_params.get("max_deposits", 15)
        deposit_amount_range = tx_params.get(
            "deposit_amount_range", [15000, 75000])

        num_cycles = random_instance.randint(min_deposits, max_deposits)

        # Account for layering time in start offset
        layering_config = self.params.get("layering_config", {})
        max_extra_hops = layering_config.get(
            "max_extra_hops", 5) if layering_config.get("enabled", False) else 0

        day_span = (self.time_span["end_date"] -
                    self.time_span["start_date"]).days
        start_day_offset = random_instance.randint(
            0, max(0, day_span - (num_cycles * 2) - max_extra_hops * 3))
        current_time = self.time_span["start_date"] + \
            datetime.timedelta(days=start_day_offset)

        # Create PatternInjector for layering
        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Build exclusion list
        exclude_accounts = front_business_accounts + overseas_dest_accounts

        # NEW: Camouflage probability for front business
        # Some deposits should be smaller to look like normal business operations
        camouflage_probability = tx_params.get("camouflage_probability", 0.25)

        # Pre-generate deposit amounts with camouflage
        if camouflage_probability > 0:
            deposit_amounts = generate_fraud_with_camouflage(
                count=num_cycles,
                camouflage_probability=camouflage_probability,
                # Small cash like average user
                small_amount_range=(50.0, 300.0),
                medium_amount_range=(300.0, 2000.0),  # Medium deposits
                high_amount_range=(
                    deposit_amount_range[0], deposit_amount_range[1]),
            )
        else:
            deposit_amounts = [
                round(random_instance.uniform(
                    deposit_amount_range[0], deposit_amount_range[1]), 2)
                for _ in range(num_cycles)
            ]

        all_txs = []
        for i in range(num_cycles):
            # --- Deposit ---
            deposit_time = current_time
            target_deposit_account = random_instance.choice(
                front_business_accounts)
            deposit_amount = deposit_amounts[i]

            deposit_tx = pattern_injector._create_transaction_edge(
                src_id=self.graph_generator.cash_account_id,
                dest_id=target_deposit_account,
                timestamp=deposit_time,
                amount=deposit_amount,
                transaction_type=TransactionType.DEPOSIT,
                is_fraudulent=True
            )
            all_txs.append(
                (self.graph_generator.cash_account_id, target_deposit_account, deposit_tx))

            # --- Transfer with LAYERING ---
            delay_hours = random_instance.uniform(0.5, 6)
            transfer_time = deposit_time + \
                datetime.timedelta(hours=delay_hours)
            transfer_amount = round(
                deposit_amount * random_instance.uniform(0.8, 1.0), 2)

            source_transfer_account = random_instance.choice(
                front_business_accounts)
            dest_transfer_account = random_instance.choice(
                overseas_dest_accounts)

            # === ADD LAYERING HOPS ===
            layering_txs, final_time, final_amount = self.generate_layering_transactions(
                source_account=source_transfer_account,
                dest_account=dest_transfer_account,
                amount=transfer_amount,
                start_time=transfer_time,
                pattern_injector=pattern_injector,
                exclude_accounts=exclude_accounts
            )

            # Add layering transactions
            all_txs.extend(layering_txs)

            # Determine actual source for final transfer
            if layering_txs:
                actual_src = layering_txs[-1][1]
                transfer_time = final_time
                transfer_amount = final_amount
            else:
                actual_src = source_transfer_account

            transfer_tx = pattern_injector._create_transaction_edge(
                src_id=actual_src,
                dest_id=dest_transfer_account,
                timestamp=transfer_time,
                amount=transfer_amount,
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=True
            )
            all_txs.append(
                (actual_src, dest_transfer_account, transfer_tx))

            # --- Advance time for the next cycle ---
            current_time += datetime.timedelta(
                days=random_instance.uniform(0.5, 2.0))

        if all_txs:
            # Sort transactions by time to ensure proper sequence ordering
            all_txs.sort(key=lambda x: x[2].timestamp)
            start_time = all_txs[0][2].timestamp
            end_time = all_txs[-1][2].timestamp
            sequences.append(TransactionSequence(
                transactions=all_txs,
                sequence_name="front_business_deposit_transfer_pairs",
                start_time=start_time,
                duration=end_time - start_time
            ))

        return sequences


class FrontBusinessPattern(CompositePattern):
    """Injects front business activity pattern"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = FrontBusinessStructural(graph_generator, params)
        temporal_component = FrequentCashDepositsAndOverseasTransfersTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "FrontBusinessActivity"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
