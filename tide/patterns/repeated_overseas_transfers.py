import random
import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..data_structures import NodeType, TransactionType, TransactionAttributes


class IndividualToOverseasStructural(StructuralComponent):
    """
    An individual connected to multiple overseas entities (individuals or businesses).
    The individual's account and the overseas entities' accounts are involved.
    """

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        central_individual_id = None
        peripheral_overseas_account_ids = []
        entity_roles = {}

        # Find an individual
        individual_ids = self.filter_entities_by_criteria(
            available_entities, {"node_type": NodeType.INDIVIDUAL}
        )
        random.shuffle(individual_ids)

        # Default for now
        min_overseas_entities = 2
        max_overseas_entities = 5

        for ind_id in individual_ids:
            owned_accounts = [
                n for n in self.graph.neighbors(ind_id)
                if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
            ]
            if not owned_accounts:
                continue

            central_individual_account_id = random.choice(owned_accounts)
            central_individual_id = ind_id
            entity_roles[central_individual_account_id] = "source_individual_account"
            entity_roles[ind_id] = "source_individual"

            all_accounts = [
                e for e in available_entities
                if self.graph.nodes[e].get("node_type") == NodeType.ACCOUNT
            ]

            individual_country = self.graph.nodes[ind_id].get(
                "address", {}).get("country")
            high_risk_countries_list = self.params.get(
                "high_risk_countries", [])

            potential_overseas_dest_accounts = []
            for acc_id in all_accounts:
                if acc_id == central_individual_account_id:
                    continue

                acc_country = self.graph.nodes[acc_id].get(
                    "address", {}).get("country")
                if acc_country != individual_country:
                    if not high_risk_countries_list or acc_country in high_risk_countries_list:
                        # If high_risk_countries_list is empty, consider all overseas
                        potential_overseas_dest_accounts.append(acc_id)

            if len(potential_overseas_dest_accounts) >= min_overseas_entities:
                num_to_select = random.randint(
                    min_overseas_entities,
                    min(len(potential_overseas_dest_accounts),
                        max_overseas_entities)
                )
                peripheral_overseas_account_ids = random.sample(
                    potential_overseas_dest_accounts, num_to_select)
                for i, acc_id in enumerate(peripheral_overseas_account_ids):
                    entity_roles[acc_id] = f"destination_overseas_account_{i+1}"
                break

        if not central_individual_id or not peripheral_overseas_account_ids:
            return EntitySelection(central_entities=[], peripheral_entities=[], entity_roles={})

        return EntitySelection(
            central_entities=[central_individual_account_id],
            peripheral_entities=peripheral_overseas_account_ids,
            entity_roles=entity_roles
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
    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = IndividualToOverseasStructural(
            graph_generator, params)
        temporal_component = FrequentOrPeriodicTransfersTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "RepeatedOutboundTransfersToOverseas"
