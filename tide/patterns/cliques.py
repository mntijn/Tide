import random
import datetime
from typing import List, Dict, Any, Tuple, Set

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType
from ..datastructures.attributes import TransactionAttributes


class CliquesStructural(StructuralComponent):
    """
    Structural: A fully connected subset of nodes (k-clique).
    Mix of individuals (who might know each other) and front companies.
    """

    @property
    def num_required_entities(self) -> int:
        # Minimum k=4 for suspicious cliques
        return 4

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        pattern_config = self.params.get(
            "pattern_config", {}).get("cliques", {})
        min_clique_size = pattern_config.get("min_clique_size", 4)
        max_clique_size = pattern_config.get("max_clique_size", 8)

        # We need a mix of individuals and businesses
        clique_entities = []
        clique_accounts = []

        # First, try to find entities that might already be connected
        # (e.g., in same clusters suggesting they might know each other)
        potential_clique_members = []

        # Look in various risk clusters
        for cluster_name in ["super_high_risk", "structuring_candidates", "intermediaries"]:
            cluster_members = self.get_cluster(cluster_name)
            # Filter for individuals and businesses
            entities_in_cluster = [
                e for e in cluster_members
                if self.graph.nodes[e].get("node_type") in [NodeType.INDIVIDUAL, NodeType.BUSINESS]
            ]
            potential_clique_members.extend(entities_in_cluster)

        # Deduplicate
        potential_clique_members = list(set(potential_clique_members))

        # If not enough, add more entities
        if len(potential_clique_members) < max_clique_size:
            all_entities = self.filter_entities_by_criteria(
                available_entities, {"node_type": NodeType.INDIVIDUAL}
            )
            all_businesses = self.filter_entities_by_criteria(
                available_entities, {"node_type": NodeType.BUSINESS}
            )
            potential_clique_members.extend(all_entities[:10])
            potential_clique_members.extend(all_businesses[:5])
            potential_clique_members = list(set(potential_clique_members))

        random.shuffle(potential_clique_members)

        # Select entities with accounts for the clique
        for entity_id in potential_clique_members:
            # Get primary account for this entity
            owned_accounts = [
                n for n in self.graph.neighbors(entity_id)
                if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
            ]

            if owned_accounts:
                clique_entities.append(entity_id)
                # Use the first account as the primary one for this entity
                clique_accounts.append(owned_accounts[0])

                if len(clique_entities) >= max_clique_size:
                    break

        if len(clique_entities) < min_clique_size:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Trim to desired size
        clique_size = random.randint(min_clique_size,
                                     min(len(clique_entities), max_clique_size))
        clique_entities = clique_entities[:clique_size]
        clique_accounts = clique_accounts[:clique_size]

        # For cliques, all entities are "central" as they're all interconnected
        return EntitySelection(
            central_entities=clique_accounts,
            peripheral_entities=[]  # No peripheral entities in a clique
        )


class CliquesTemporal(TemporalComponent):
    """
    Temporal: Frequent transactions between all members of the clique.
    Fast deposits and transfers creating a fully connected transaction graph.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []

        if not entity_selection.central_entities or len(entity_selection.central_entities) < 4:
            return sequences

        clique_accounts = entity_selection.central_entities

        pattern_config = self.params.get(
            "pattern_config", {}).get("cliques", {})
        tx_params = pattern_config.get("transaction_params", {})

        transactions_per_pair = tx_params.get("transactions_per_pair", [1, 3])
        amount_range = tx_params.get("amount_range", [5000, 50000])
        pattern_duration_days = tx_params.get("pattern_duration_days", 30)

        # Calculate start time
        time_span_days = (
            self.time_span["end_date"] - self.time_span["start_date"]).days
        max_start_offset = max(0, time_span_days - pattern_duration_days)
        start_day_offset = random.randint(0, max_start_offset)
        pattern_start_time = self.time_span["start_date"] + \
            datetime.timedelta(days=start_day_offset)
        pattern_end_time = pattern_start_time + \
            datetime.timedelta(days=pattern_duration_days)

        all_transactions = []

        # Create transactions between every pair of accounts (full connectivity)
        for i, src_account in enumerate(clique_accounts):
            for j, dest_account in enumerate(clique_accounts):
                if i != j:  # No self-transactions
                    # Determine number of transactions for this pair
                    num_transactions = random.randint(transactions_per_pair[0],
                                                      transactions_per_pair[1])

                    for _ in range(num_transactions):
                        # Random time within the pattern duration
                        tx_time = pattern_start_time + datetime.timedelta(
                            days=random.uniform(0, pattern_duration_days),
                            hours=random.randint(0, 23),
                            minutes=random.randint(0, 59)
                        )

                        # Ensure within bounds
                        if tx_time > pattern_end_time:
                            tx_time = pattern_end_time
                        if tx_time > self.time_span["end_date"]:
                            tx_time = self.time_span["end_date"]

                        # Get currency for structuring
                        src_currency = self.graph.nodes[src_account].get(
                            "currency", "EUR")

                        # Generate amount (possibly structured)
                        amounts = self.generate_structured_amounts(
                            count=1,
                            base_amount=random.uniform(
                                amount_range[0], amount_range[1]),
                            target_currency=src_currency
                        )

                        tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                            src_id=src_account,
                            dest_id=dest_account,
                            timestamp=tx_time,
                            amount=amounts[0],
                            transaction_type=TransactionType.TRANSFER,
                            is_fraudulent=True
                        )

                        all_transactions.append(
                            (src_account, dest_account, tx_attrs))

        # Sort transactions by timestamp
        all_transactions.sort(key=lambda x: x[2].timestamp)

        if all_transactions:
            # Group transactions into sequences by time windows
            # This helps organize the clique activity into logical sequences
            window_size_hours = 24  # Daily sequences

            current_sequence_txs = []
            current_window_start = all_transactions[0][2].timestamp
            sequence_counter = 1

            for tx in all_transactions:
                tx_time = tx[2].timestamp

                # Check if this transaction belongs to a new time window
                if (tx_time - current_window_start).total_seconds() > window_size_hours * 3600:
                    # Save current sequence if it has transactions
                    if current_sequence_txs:
                        seq_start = current_sequence_txs[0][2].timestamp
                        seq_end = current_sequence_txs[-1][2].timestamp
                        sequences.append(TransactionSequence(
                            transactions=current_sequence_txs,
                            sequence_name=f"clique_activity_day_{sequence_counter}",
                            start_time=seq_start,
                            duration=seq_end -
                            seq_start if len(
                                current_sequence_txs) > 1 else datetime.timedelta(0)
                        ))
                        sequence_counter += 1

                    # Start new sequence
                    current_sequence_txs = [tx]
                    current_window_start = tx_time
                else:
                    current_sequence_txs.append(tx)

            # Don't forget the last sequence
            if current_sequence_txs:
                seq_start = current_sequence_txs[0][2].timestamp
                seq_end = current_sequence_txs[-1][2].timestamp
                sequences.append(TransactionSequence(
                    transactions=current_sequence_txs,
                    sequence_name=f"clique_activity_day_{sequence_counter}",
                    start_time=seq_start,
                    duration=seq_end -
                    seq_start if len(
                        current_sequence_txs) > 1 else datetime.timedelta(0)
                ))

        return sequences


class CliquesPattern(CompositePattern):
    """Injects cliques pattern (fully connected k-clique)"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = CliquesStructural(graph_generator, params)
        temporal_component = CliquesTemporal(graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "Cliques"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
