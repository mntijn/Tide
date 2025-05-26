import random
import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..data_structures import NodeType, TransactionType, TransactionAttributes


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

        potential_front_businesses = self.filter_entities_by_criteria(
            available_entities,
            {"node_type": NodeType.BUSINESS}
        )
        random.shuffle(potential_front_businesses)

        # If no businesses in available_entities, consider all businesses in graph
        if not potential_front_businesses:
            potential_front_businesses = list(
                self.graph_generator.all_nodes.get(NodeType.BUSINESS, []))
            random.shuffle(potential_front_businesses)

        # Default values for pattern parameters
        min_bus_accounts = self.params.get("front_business_pattern", {}).get(
            "min_accounts_for_front_business", 2)
        num_front_business_accounts_to_use = self.params.get(
            "front_business_pattern", {}).get("num_front_business_accounts_to_use", 3)
        min_overseas_dest_bus_accounts = self.params.get(
            "front_business_pattern", {}).get("min_overseas_destination_accounts", 2)
        max_overseas_bus_accounts_for_front = self.params.get(
            "front_business_pattern", {}).get("max_overseas_destination_accounts_for_front", 4)

        for bus_id in potential_front_businesses:
            owned_accounts_data = [
                n for n in self.graph.neighbors(bus_id)
                if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
            ]

            if len(owned_accounts_data) >= min_bus_accounts:
                potential_dest_accounts = []
                # Search all accounts in the graph for overseas destinations
                all_graph_accounts = self.graph_generator.all_nodes.get(
                    NodeType.ACCOUNT, [])
                business_country = self.graph.nodes[bus_id].get(
                    "address", {}).get("country")

                for acc_id in all_graph_accounts:
                    if acc_id in owned_accounts_data:  # Exclude accounts owned by the front business itself
                        continue

                    acc_node_data = self.graph.nodes[acc_id]
                    acc_country = acc_node_data.get(
                        "address", {}).get("country")

                    # Ensure account has a country and it's different from the front business's country
                    if acc_country and acc_country != business_country:
                        # Check if this account is owned by any business
                        is_owned_by_business = False
                        for owner_id in self.graph.predecessors(acc_id):
                            if self.graph.nodes[owner_id].get("node_type") == NodeType.BUSINESS:
                                is_owned_by_business = True
                                break
                        if is_owned_by_business:
                            potential_dest_accounts.append(acc_id)

                if len(potential_dest_accounts) >= min_overseas_dest_bus_accounts:
                    central_business_id = bus_id
                    business_accounts_ids = random.sample(owned_accounts_data, k=min(
                        len(owned_accounts_data), num_front_business_accounts_to_use))

                    num_overseas_to_select = random.randint(
                        min_overseas_dest_bus_accounts,
                        min(len(potential_dest_accounts),
                            max_overseas_bus_accounts_for_front)
                    )
                    overseas_business_accounts_ids = random.sample(
                        potential_dest_accounts, num_overseas_to_select)
                    break

        if not central_business_id or not business_accounts_ids or not overseas_business_accounts_ids:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        return EntitySelection(
            central_entities=business_accounts_ids,
            peripheral_entities=overseas_business_accounts_ids,
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

        front_business_accounts = entity_selection.central_entities
        overseas_dest_accounts = entity_selection.peripheral_entities

        # Default values for pattern parameters
        min_deposit_cycles = 5
        max_deposit_cycles = 15

        num_deposit_cycles = random.randint(
            min_deposit_cycles, max_deposit_cycles)

        if (self.time_span["end_date"] - self.time_span["start_date"]).days < 15:
            start_day_offset_range = max(
                0, (self.time_span["end_date"] - self.time_span["start_date"]).days - 1)
        else:
            start_day_offset_range = (
                self.time_span["end_date"] - self.time_span["start_date"]).days - 15

        base_start_time = self.time_span["start_date"] + datetime.timedelta(
            days=random.randint(0, start_day_offset_range)
        )

        current_time = base_start_time

        for i in range(num_deposit_cycles):
            if not front_business_accounts:
                break  # Cannot proceed without accounts
            target_deposit_account = random.choice(front_business_accounts)
            num_deposits_in_cycle = random.randint(1, 3)

            deposit_timestamps = self.generate_timestamps(
                current_time, "high_frequency", num_deposits_in_cycle)
            deposit_amounts = self.generate_structured_amounts(
                num_deposits_in_cycle, base_amount=random.uniform(10000, 50000))

            deposit_txs_this_cycle = []
            for j in range(min(num_deposits_in_cycle, len(deposit_timestamps))):
                tx_attrs_deposit = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                    src_id=self.graph_generator.cash_account_id,
                    dest_id=target_deposit_account,
                    timestamp=deposit_timestamps[j],
                    amount=deposit_amounts[j],
                    transaction_type=TransactionType.DEPOSIT,
                    is_fraudulent=True
                )
                deposit_txs_this_cycle.append(
                    (self.graph_generator.cash_account_id, target_deposit_account, tx_attrs_deposit))

            last_deposit_time_this_cycle = current_time  # Fallback
            if deposit_txs_this_cycle:
                last_deposit_time_this_cycle = deposit_timestamps[-1]
                sequences.append(TransactionSequence(
                    transactions=deposit_txs_this_cycle,
                    sequence_name=f"cash_deposits_cycle_{i+1}",
                    start_time=deposit_timestamps[0],
                    duration=deposit_timestamps[-1] - deposit_timestamps[0]
                ))

            if not overseas_dest_accounts:  # Cannot proceed with transfers if no overseas accounts
                current_time = last_deposit_time_this_cycle + \
                    datetime.timedelta(days=random.uniform(1, 7))
                continue

            transfer_start_time = last_deposit_time_this_cycle + \
                datetime.timedelta(hours=random.uniform(0.5, 6))
            num_overseas_transfers_in_cycle = random.randint(
                1, len(overseas_dest_accounts))

            transfer_timestamps = self.generate_timestamps(
                transfer_start_time, "immediate_followup", num_overseas_transfers_in_cycle)

            total_deposited_this_cycle = sum(
                amt for _, _, attrs in deposit_txs_this_cycle for amt in [attrs.amount])
            base_transfer_amount = (total_deposited_this_cycle / num_overseas_transfers_in_cycle * random.uniform(0.8, 1.0)
                                    if total_deposited_this_cycle > 0 and num_overseas_transfers_in_cycle > 0
                                    else random.uniform(5000, 20000))
            transfer_amounts = self.generate_structured_amounts(
                num_overseas_transfers_in_cycle, base_amount=base_transfer_amount)

            transfer_txs_this_cycle = []
            source_for_transfer_account = target_deposit_account

            for k in range(min(num_overseas_transfers_in_cycle, len(transfer_timestamps))):
                if not overseas_dest_accounts:
                    break
                dest_overseas_account = random.choice(overseas_dest_accounts)
                tx_attrs_transfer = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                    src_id=source_for_transfer_account,
                    dest_id=dest_overseas_account,
                    timestamp=transfer_timestamps[k],
                    amount=transfer_amounts[k],
                    transaction_type=TransactionType.TRANSFER,
                    is_fraudulent=True
                )
                transfer_txs_this_cycle.append(
                    (source_for_transfer_account, dest_overseas_account, tx_attrs_transfer))

            last_tx_time_this_cycle = last_deposit_time_this_cycle
            if transfer_txs_this_cycle:
                last_tx_time_this_cycle = transfer_timestamps[-1]
                sequences.append(TransactionSequence(
                    transactions=transfer_txs_this_cycle,
                    sequence_name=f"overseas_transfers_cycle_{i+1}",
                    start_time=transfer_timestamps[0],
                    duration=transfer_timestamps[-1] - transfer_timestamps[0]
                ))

            current_time = last_tx_time_this_cycle + \
                datetime.timedelta(days=random.uniform(1, 7))

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
