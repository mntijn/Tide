import random
import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..data_structures import NodeType, TransactionType, TransactionAttributes


class RepeatedOverseasTransfersStructural(StructuralComponent):
    """
    Selects:
    - An individual or business entity.
    - An existing overseas account (or creates one if none suitable).
    """

    @property
    def num_required_entities(self) -> int:
        return 2

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        central_individual_id = None
        central_individual_account_id = None
        peripheral_overseas_account_ids = []

        potential_individuals = self.filter_entities_by_criteria(
            available_entities, {"node_type": NodeType.INDIVIDUAL}
        )
        random.shuffle(potential_individuals)

        if not potential_individuals:
            potential_individuals = list(
                self.graph_generator.all_nodes.get(NodeType.INDIVIDUAL, []))
            random.shuffle(potential_individuals)

        min_overseas_entities = self.params.get(
            "repeated_overseas_transfers_pattern", {}).get("min_overseas_entities", 2)
        max_overseas_entities = self.params.get(
            "repeated_overseas_transfers_pattern", {}).get("max_overseas_entities", 5)

        for ind_id in potential_individuals:
            # Get accounts owned by this individual from the main graph
            owned_accounts = [
                n for n in self.graph.neighbors(ind_id)
                if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
            ]
            if not owned_accounts:
                continue

            current_central_individual_account_id = random.choice(
                owned_accounts)

            # Search all accounts in the graph for suitable overseas destinations
            all_graph_accounts = self.graph_generator.all_nodes.get(
                NodeType.ACCOUNT, [])

            individual_node_data = self.graph.nodes[ind_id]
            individual_country = individual_node_data.get(
                "address", {}).get("country")
            high_risk_countries_list = self.params.get(
                "high_risk_countries", [])

            potential_overseas_dest_accounts = []
            for acc_id in all_graph_accounts:
                if acc_id in owned_accounts:
                    continue

                acc_node_data = self.graph.nodes[acc_id]
                acc_country = acc_node_data.get("address", {}).get("country")

                # Ensure account has a country and it's different
                if acc_country and acc_country != individual_country:
                    if not high_risk_countries_list or acc_country in high_risk_countries_list:
                        # Check if this account is owned by an Individual or Business
                        is_owned_by_entity = False
                        for owner_id in self.graph.predecessors(acc_id):
                            owner_node_type = self.graph.nodes[owner_id].get(
                                "node_type")
                            if owner_node_type in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
                                is_owned_by_entity = True
                                break
                        if is_owned_by_entity:
                            potential_overseas_dest_accounts.append(acc_id)

            if len(potential_overseas_dest_accounts) >= min_overseas_entities:
                central_individual_id = ind_id
                central_individual_account_id = current_central_individual_account_id

                num_to_select = random.randint(
                    min_overseas_entities,
                    min(len(potential_overseas_dest_accounts),
                        max_overseas_entities)
                )
                peripheral_overseas_account_ids = random.sample(
                    potential_overseas_dest_accounts, num_to_select
                )
                break

        if not central_individual_id or not central_individual_account_id or not peripheral_overseas_account_ids:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        return EntitySelection(
            central_entities=[central_individual_account_id],
            peripheral_entities=peripheral_overseas_account_ids,
        )


class FrequentOrPeriodicTransfersTemporal(TemporalComponent):
    """
    Temporal: High frequency (burst) or periodic transfers.
    Amounts are often structured below reporting thresholds.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []
        if not entity_selection.central_entities or not entity_selection.peripheral_entities:
            return sequences

        source_account_id = entity_selection.central_entities[0]
        overseas_account_ids = entity_selection.peripheral_entities

        temporal_type = random.choice(["high_frequency", "periodic"])
        num_transactions = 0
        if temporal_type == "high_frequency":
            num_transactions = random.randint(
                10, 20)  # Default: 10-20 transactions
        else:  # periodic
            num_transactions = random.randint(
                5, 10)   # Default: 5-10 transactions

        if (self.time_span["end_date"] - self.time_span["start_date"]).days < 30:
            start_day_offset_range = max(
                0, (self.time_span["end_date"] - self.time_span["start_date"]).days - 1)
        else:
            start_day_offset_range = (
                self.time_span["end_date"] - self.time_span["start_date"]).days - 30

        base_start_time = self.time_span["start_date"] + datetime.timedelta(
            days=random.randint(0, start_day_offset_range)
        )

        timestamps = self.generate_timestamps(
            base_start_time, temporal_type, num_transactions)
        amounts = self.generate_structured_amounts(num_transactions)

        transactions_for_sequence = []
        duration = datetime.timedelta(days=0)
        if timestamps:
            duration = timestamps[-1] - timestamps[0]

        for i in range(min(num_transactions, len(timestamps))):
            destination_account_id = random.choice(overseas_account_ids)
            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=source_account_id,
                dest_id=destination_account_id,
                timestamp=timestamps[i],
                amount=amounts[i],
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=True
            )
            transactions_for_sequence.append(
                (source_account_id, destination_account_id, tx_attrs))

        if transactions_for_sequence:
            sequences.append(
                TransactionSequence(
                    transactions=transactions_for_sequence,
                    sequence_name=f"repeated_transfers_{temporal_type}",
                    start_time=timestamps[0] if timestamps else base_start_time,
                    duration=duration
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
