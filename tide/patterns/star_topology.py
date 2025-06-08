import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType
from ..datastructures.attributes import TransactionAttributes
from ..utils.random_instance import random_instance


class StarTopologyStructural(StructuralComponent):
    """
    Structural: A central hub account connected to multiple outer nodes.
    The hub can be an individual (mule) or shell company account.
    """

    @property
    def num_required_entities(self) -> int:
        # Need at least 1 hub + 3 outer nodes
        return 4

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        pattern_config = self.params.get(
            "pattern_config", {}).get("starTopology", {})
        min_outer_nodes = pattern_config.get("min_outer_nodes", 4)
        max_outer_nodes = pattern_config.get("max_outer_nodes", 12)

        # Find a suitable hub entity (mule or shell company)
        hub_entity_id = None
        hub_account_id = None

        # Prioritize individuals/businesses from intermediary clusters
        potential_hubs = []

        # Look for entities marked as intermediaries or structuring candidates
        for cluster_name in ["intermediaries", "structuring_candidates", "super_high_risk"]:
            cluster_entities = self.get_cluster(cluster_name)
            for entity_id in cluster_entities:
                if self.graph.nodes[entity_id].get("node_type") in [NodeType.INDIVIDUAL, NodeType.BUSINESS]:
                    potential_hubs.append(entity_id)

        # Deduplicate and shuffle
        potential_hubs = list(set(potential_hubs))
        random_instance.shuffle(potential_hubs)

        # Find a hub with at least one account
        for entity_id in potential_hubs:
            owned_accounts = [
                n for n in self.graph.neighbors(entity_id)
                if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
            ]

            if owned_accounts:
                hub_entity_id = entity_id
                hub_account_id = random_instance.choice(owned_accounts)
                break

        if not hub_entity_id:
            # Fallback: any entity with an account
            all_entities = self.filter_entities_by_criteria(
                available_entities, {"node_type": NodeType.INDIVIDUAL}
            )
            random.shuffle(all_entities)

            for entity_id in all_entities:
                owned_accounts = [
                    n for n in self.graph.neighbors(entity_id)
                    if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
                ]
                if owned_accounts:
                    hub_entity_id = entity_id
                    hub_account_id = random.choice(owned_accounts)
                    break

        if not hub_account_id:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Find outer node accounts (various sources and destinations)
        outer_accounts = []
        hub_country = self.graph.nodes[hub_entity_id].get("country_code")

        # Get all accounts in the graph except the hub's accounts
        hub_owned_accounts = set([
            n for n in self.graph.neighbors(hub_entity_id)
            if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
        ])

        all_accounts = [
            acc_id for acc_id in self.graph_generator.all_nodes.get(NodeType.ACCOUNT, [])
            if acc_id not in hub_owned_accounts
        ]

        # Mix of domestic and international accounts
        domestic_accounts = []
        international_accounts = []

        for acc_id in all_accounts:
            acc_country = self.graph.nodes[acc_id].get("country_code")
            if acc_country == hub_country:
                domestic_accounts.append(acc_id)
            else:
                international_accounts.append(acc_id)

        # Select a mix of both
        random.shuffle(domestic_accounts)
        random.shuffle(international_accounts)

        # Take roughly 60% domestic, 40% international
        num_to_select = random.randint(min_outer_nodes,
                                       min(len(all_accounts), max_outer_nodes))
        num_domestic = int(num_to_select * 0.6)
        num_international = num_to_select - num_domestic

        outer_accounts = (domestic_accounts[:num_domestic] +
                          international_accounts[:num_international])

        if len(outer_accounts) < min_outer_nodes:
            # Just take any available accounts
            outer_accounts = all_accounts[:max_outer_nodes]

        if len(outer_accounts) < min_outer_nodes:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        return EntitySelection(
            central_entities=[hub_account_id],
            peripheral_entities=outer_accounts[:num_to_select]
        )


class StarTopologyTemporal(TemporalComponent):
    """
    Temporal: Burst of deposits followed by quick outbound transfers.
    Most received money is transferred out quickly.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []

        if not entity_selection.central_entities or not entity_selection.peripheral_entities:
            return sequences

        hub_account_id = entity_selection.central_entities[0]
        outer_accounts = entity_selection.peripheral_entities

        pattern_config = self.params.get(
            "pattern_config", {}).get("starTopology", {})
        tx_params = pattern_config.get("transaction_params", {})

        inflow_percentage = tx_params.get(
            "inflow_percentage", 0.6)  # 60% are inflows
        deposit_burst_hours = tx_params.get("deposit_burst_hours", 48)
        transfer_delay_hours = tx_params.get("transfer_delay_hours", [2, 24])
        amount_range = tx_params.get("amount_range", [1000, 15000])

        # Determine which accounts are sources (inflow) vs destinations (outflow)
        num_source_accounts = max(
            1, int(len(outer_accounts) * inflow_percentage))
        source_accounts = outer_accounts[:num_source_accounts]
        dest_accounts = outer_accounts[num_source_accounts:]

        if not dest_accounts:  # Ensure at least one destination
            dest_accounts = [outer_accounts[-1]]
            source_accounts = outer_accounts[:-1]

        # Calculate start time
        time_span_days = (
            self.time_span["end_date"] - self.time_span["start_date"]).days
        start_day_offset = random.randint(0, max(0, time_span_days - 7))
        base_start_time = self.time_span["start_date"] + \
            datetime.timedelta(days=start_day_offset)

        # Phase 1: Burst of deposits to hub
        deposit_transactions = []
        total_deposited = 0

        for source_acc in source_accounts:
            # Random time within burst window
            deposit_time = base_start_time + datetime.timedelta(
                hours=random.uniform(0, deposit_burst_hours)
            )

            # Get currency for structuring
            hub_currency = self.graph.nodes[hub_account_id].get(
                "currency", "EUR")

            amounts = self.generate_structured_amounts(
                count=1,
                base_amount=random.uniform(amount_range[0], amount_range[1]),
                target_currency=hub_currency
            )
            deposit_amount = amounts[0]

            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=source_acc,
                dest_id=hub_account_id,
                timestamp=deposit_time,
                amount=deposit_amount,
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=True
            )

            deposit_transactions.append((source_acc, hub_account_id, tx_attrs))
            total_deposited += deposit_amount

        # Create deposit sequence
        if deposit_transactions:
            deposit_times = [tx[2].timestamp for tx in deposit_transactions]
            sequences.append(TransactionSequence(
                transactions=deposit_transactions,
                sequence_name="star_inflows",
                start_time=min(deposit_times),
                duration=max(deposit_times) - min(deposit_times) if len(
                    deposit_times) > 1 else datetime.timedelta(0)
            ))

        # Phase 2: Quick outbound transfers
        last_deposit_time = max([tx[2].timestamp for tx in deposit_transactions]
                                ) if deposit_transactions else base_start_time
        transfer_start_time = last_deposit_time + datetime.timedelta(
            hours=random.uniform(
                transfer_delay_hours[0], transfer_delay_hours[1])
        )

        transfer_transactions = []
        amount_to_distribute = total_deposited * \
            random.uniform(0.85, 0.95)  # Keep some funds

        # Distribute funds to destination accounts
        for i, dest_acc in enumerate(dest_accounts):
            transfer_time = transfer_start_time + datetime.timedelta(
                minutes=random.randint(0, 120)  # Within 2 hours
            )

            # Distribute proportionally with some randomness
            if i == len(dest_accounts) - 1:
                # Last destination gets remaining
                transfer_amount = amount_to_distribute
            else:
                proportion = 1.0 / len(dest_accounts)
                transfer_amount = amount_to_distribute * \
                    proportion * random.uniform(0.8, 1.2)
                amount_to_distribute -= transfer_amount

            if transfer_amount > 0:
                tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                    src_id=hub_account_id,
                    dest_id=dest_acc,
                    timestamp=transfer_time,
                    amount=round(transfer_amount, 2),
                    transaction_type=TransactionType.TRANSFER,
                    is_fraudulent=True
                )

                transfer_transactions.append(
                    (hub_account_id, dest_acc, tx_attrs))

        # Create transfer sequence
        if transfer_transactions:
            transfer_times = [tx[2].timestamp for tx in transfer_transactions]
            sequences.append(TransactionSequence(
                transactions=transfer_transactions,
                sequence_name="star_outflows",
                start_time=min(transfer_times),
                duration=max(transfer_times) - min(transfer_times) if len(
                    transfer_times) > 1 else datetime.timedelta(0)
            ))

        return sequences


class StarTopologyPattern(CompositePattern):
    """Injects star topology pattern"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = StarTopologyStructural(graph_generator, params)
        temporal_component = StarTopologyTemporal(graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "StarTopology"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
