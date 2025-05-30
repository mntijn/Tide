import random
import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType


class RepeatedOverseasTransfersStructural(StructuralComponent):
    """
    Selects:
    - An individual or business entity (central).
    - One of their domestic accounts (peripheral).
    - Multiple overseas accounts (peripheral).
    """

    @property
    def num_required_entities(self) -> int:
        # Central entity (Ind/Bus) + 1 source account + min overseas accounts
        # The pattern config specifies min_overseas_entities
        # So, 1 central entity is the core requirement here.
        return 1

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        central_owner_id = None
        source_account_id = None
        peripheral_overseas_account_ids = []

        pattern_config = self.params.get("repeatedOverseas", {})
        min_overseas_entities = pattern_config.get("min_overseas_entities", 2)
        max_overseas_entities = pattern_config.get("max_overseas_entities", 5)

        # Prioritize entities likely to make overseas transfers
        potential_source_entities = []
        # Order: offshore_candidates -> super_high_risk -> (individuals/businesses in high_risk_countries)
        for cluster_name in ["offshore_candidates", "super_high_risk"]:
            entities_in_cluster = self.get_cluster(cluster_name)
            # Filter for individuals or businesses only from these clusters
            for entity_id in entities_in_cluster:
                node_type = self.graph.nodes[entity_id].get("node_type")
                if node_type in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
                    potential_source_entities.append(entity_id)
            if len(potential_source_entities) > 20:  # Gather a pool
                break

        # Add entities from high_risk_countries if pool is small
        if len(potential_source_entities) < 10:
            hr_country_entities = self.get_cluster("high_risk_countries")
            for entity_id in hr_country_entities:
                node_type = self.graph.nodes[entity_id].get("node_type")
                if node_type in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
                    potential_source_entities.append(entity_id)

        potential_source_entities = list(dict.fromkeys(
            potential_source_entities))  # Deduplicate

        # Fallback if specific clusters are empty or yield too few candidates
        if not potential_source_entities:
            for node_type_enum in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
                potential_source_entities.extend(
                    self.graph_generator.all_nodes.get(node_type_enum, []))

        random.shuffle(potential_source_entities)

        for entity_id in potential_source_entities:
            entity_node_data = self.graph.nodes[entity_id]
            entity_country = entity_node_data.get(
                "country_code")  # Using country_code

            owned_domestic_accounts = []
            # Assuming direct edges for accounts
            for acc_id in self.graph.neighbors(entity_id):
                acc_data = self.graph.nodes.get(acc_id, {})
                if acc_data.get("node_type") == NodeType.ACCOUNT and acc_data.get("country_code") == entity_country:
                    owned_domestic_accounts.append(acc_id)

            # If entity is a business, also check accounts of individuals who own this business (if relevant for pattern).
            # For this pattern, source is the entity itself, so only its direct accounts.

            if not owned_domestic_accounts:
                continue

            current_source_account_id = random.choice(owned_domestic_accounts)

            # Search all accounts in the graph for suitable overseas destinations
            all_graph_accounts = self.graph_generator.all_nodes.get(
                NodeType.ACCOUNT, [])
            high_risk_countries_list = self.params.get("high_risk_config", {}).get(
                "high_risk_countries", [])  # from graph.yaml via params

            potential_overseas_dest_accounts = []
            for acc_id in all_graph_accounts:
                # Exclude self, own domestic accounts
                if acc_id == current_source_account_id or acc_id in owned_domestic_accounts:
                    continue

                acc_node_data = self.graph.nodes[acc_id]
                acc_country = acc_node_data.get("country_code")

                if acc_country and acc_country != entity_country:
                    # Optional: Prefer accounts in high-risk countries for more suspicious patterns
                    # For now, any overseas account is a candidate, can be refined
                    # if high_risk_countries_list and acc_country not in high_risk_countries_list:
                    #     continue # Strict: only to high risk countries

                    is_owned_by_entity = False
                    for owner_id_of_dest_acc in self.graph.predecessors(acc_id):
                        owner_node_type = self.graph.nodes[owner_id_of_dest_acc].get(
                            "node_type")
                        if owner_node_type in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
                            is_owned_by_entity = True
                            break
                    if is_owned_by_entity:
                        potential_overseas_dest_accounts.append(acc_id)

            if len(potential_overseas_dest_accounts) >= min_overseas_entities:
                central_owner_id = entity_id
                source_account_id = current_source_account_id

                num_to_select = random.randint(
                    min_overseas_entities,
                    min(len(potential_overseas_dest_accounts),
                        max_overseas_entities)
                )
                peripheral_overseas_account_ids = random.sample(
                    potential_overseas_dest_accounts, num_to_select
                )
                break

        if not central_owner_id or not source_account_id or not peripheral_overseas_account_ids:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        return EntitySelection(
            # The individual or business making transfers
            central_entities=[central_owner_id],
            # Source acc + dest accs
            peripheral_entities=[source_account_id] +
            peripheral_overseas_account_ids,
        )


class FrequentOrPeriodicTransfersTemporal(TemporalComponent):
    """
    Temporal: High frequency (burst) or periodic transfers.
    Amounts are often structured below reporting thresholds.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []
        if not entity_selection.central_entities or not entity_selection.peripheral_entities or len(entity_selection.peripheral_entities) < 2:
            # Need at least one source account and one destination account in peripherals
            return sequences

        # First peripheral is source, rest are destinations
        source_account_id = entity_selection.peripheral_entities[0]
        overseas_account_ids = entity_selection.peripheral_entities[1:]

        if not overseas_account_ids:  # No destinations found
            return sequences

        # Load parameters from YAML config
        pattern_config = self.params.get("repeatedOverseas", {})
        tx_params = pattern_config.get("transaction_params", {})

        # Default from example if not in YAML
        min_tx = tx_params.get("min_transactions", 10)
        max_tx = tx_params.get("max_transactions", 30)
        amount_range = tx_params.get("transfer_amount_range", [5000, 20000])
        # transfer_interval_days: [7, 14, 30] -> used for 'periodic'
        interval_days_options = tx_params.get(
            "transfer_interval_days", [7, 14, 30])

        temporal_type = random.choice(["high_frequency", "periodic"])
        num_transactions = random.randint(min_tx, max_tx)

        # Adjust num_transactions based on temporal type if specific logic desired
        # e.g., periodic might have fewer but more spread out transactions by default.
        # For now, YAML min/max applies to both, adjusted by generate_timestamps.

        if (self.time_span["end_date"] - self.time_span["start_date"]).days < 30:
            start_day_offset_range = max(
                0, (self.time_span["end_date"] - self.time_span["start_date"]).days - 1)
        else:
            start_day_offset_range = (
                self.time_span["end_date"] - self.time_span["start_date"]).days - 30

        base_start_time = self.time_span["start_date"] + datetime.timedelta(
            days=random.randint(0, start_day_offset_range)
        )

        timestamps = []
        if temporal_type == "periodic" and interval_days_options:
            # Use interval_days for periodic transfers
            current_time = base_start_time
            for _ in range(num_transactions):
                timestamps.append(current_time)
                # Add a random interval from the configured options
                current_time += datetime.timedelta(
                    days=random.choice(interval_days_options))
                if current_time > self.time_span["end_date"]:
                    break  # Stop if we exceed simulation end time
            # Update actual number of transactions
            num_transactions = len(timestamps)
        else:  # high_frequency or periodic without specific intervals
            timestamps = self.generate_timestamps(
                base_start_time, temporal_type, num_transactions)

        # Get destination account currency for proper structuring
        # Sample a destination to determine currency (assuming all overseas accounts use similar currencies)
        sample_destination_id = random.choice(overseas_account_ids)
        destination_currency = self.graph.nodes[sample_destination_id].get(
            "currency", "EUR")

        amounts = self.generate_structured_amounts(
            count=num_transactions,
            base_amount=round(random.uniform(
                amount_range[0], amount_range[1]), 2),
            target_currency=destination_currency
        )

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
