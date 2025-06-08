import random
import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType
from ..datastructures.attributes import TransactionAttributes


class ParallelStructureStructural(StructuralComponent):
    """
    Structural: A combination of structuring (smurfing) and low-degree path hopping.
    One source, multiple intermediate paths, and one final beneficiary.
    """

    @property
    def num_required_entities(self) -> int:
        # Need at least: 1 source + 2 paths (2 intermediaries each) + 1 beneficiary = 6
        return 6

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        pattern_config = self.params.get(
            "pattern_config", {}).get("parallelStructure", {})
        min_parallel_paths = pattern_config.get("min_parallel_paths", 2)
        max_parallel_paths = pattern_config.get("max_parallel_paths", 5)
        min_hops_per_path = pattern_config.get("min_hops_per_path", 2)
        max_hops_per_path = pattern_config.get("max_hops_per_path", 4)

        # Find source entity (structuring candidate)
        source_entity_id = None
        source_account_id = None

        # Look for structuring candidates first
        structuring_candidates = self.get_cluster("structuring_candidates")
        super_high_risk = self.get_cluster("super_high_risk")

        potential_sources = list(set(structuring_candidates + super_high_risk))
        potential_sources = [
            e for e in potential_sources
            if self.graph.nodes[e].get("node_type") in [NodeType.INDIVIDUAL, NodeType.BUSINESS]
        ]

        random.shuffle(potential_sources)

        # Find source with account
        for entity_id in potential_sources:
            owned_accounts = [
                n for n in self.graph.neighbors(entity_id)
                if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
            ]

            if owned_accounts:
                source_entity_id = entity_id
                source_account_id = random.choice(owned_accounts)
                break

        if not source_entity_id:
            # Fallback to any entity with account
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
                    source_entity_id = entity_id
                    source_account_id = random.choice(owned_accounts)
                    break

        if not source_account_id:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Find beneficiary (often in high-risk jurisdiction)
        beneficiary_entity_id = None
        beneficiary_account_id = None

        # Look for beneficiaries in high-risk areas
        potential_beneficiaries = []
        for cluster_name in ["offshore_candidates", "high_risk_countries"]:
            cluster_entities = self.get_cluster(cluster_name)
            entities = [
                e for e in cluster_entities
                if self.graph.nodes[e].get("node_type") in [NodeType.INDIVIDUAL, NodeType.BUSINESS]
                and e != source_entity_id
            ]
            potential_beneficiaries.extend(entities)

        potential_beneficiaries = list(set(potential_beneficiaries))
        random.shuffle(potential_beneficiaries)

        for entity_id in potential_beneficiaries:
            owned_accounts = [
                n for n in self.graph.neighbors(entity_id)
                if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
            ]

            if owned_accounts:
                beneficiary_entity_id = entity_id
                beneficiary_account_id = random.choice(owned_accounts)
                break

        if not beneficiary_account_id:
            # Fallback: any entity different from source
            all_entities = [
                e for e in available_entities
                if self.graph.nodes[e].get("node_type") in [NodeType.INDIVIDUAL, NodeType.BUSINESS]
                and e != source_entity_id
            ]
            random.shuffle(all_entities)

            for entity_id in all_entities:
                owned_accounts = [
                    n for n in self.graph.neighbors(entity_id)
                    if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT
                ]
                if owned_accounts:
                    beneficiary_entity_id = entity_id
                    beneficiary_account_id = random.choice(owned_accounts)
                    break

        if not beneficiary_account_id:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Find intermediate accounts for parallel paths
        # Exclude source and beneficiary accounts
        excluded_accounts = {source_account_id, beneficiary_account_id}

        # Get all available accounts for intermediaries
        all_accounts = [
            acc_id for acc_id in self.graph_generator.all_nodes.get(NodeType.ACCOUNT, [])
            if acc_id not in excluded_accounts
        ]

        # Prioritize accounts from intermediary cluster
        intermediary_accounts = []
        intermediary_cluster = self.get_cluster("intermediaries")

        for acc_id in all_accounts:
            # Check if account's owner is in intermediary cluster
            for owner_id in self.graph.predecessors(acc_id):
                if owner_id in intermediary_cluster:
                    intermediary_accounts.append(acc_id)
                    break

        # If not enough, add more accounts
        if len(intermediary_accounts) < max_parallel_paths * max_hops_per_path:
            remaining_accounts = [
                acc for acc in all_accounts
                if acc not in intermediary_accounts
            ]
            random.shuffle(remaining_accounts)
            intermediary_accounts.extend(remaining_accounts)

        # Determine number of paths and hops
        num_paths = random.randint(min_parallel_paths,
                                   min(max_parallel_paths, len(intermediary_accounts) // min_hops_per_path))

        if num_paths < min_parallel_paths:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Build paths
        paths = []
        used_intermediaries = set()

        for _ in range(num_paths):
            path_length = random.randint(min_hops_per_path, max_hops_per_path)
            path = []

            # Select unique intermediaries for this path
            available_for_path = [
                acc for acc in intermediary_accounts
                if acc not in used_intermediaries
            ]

            if len(available_for_path) >= path_length:
                path = random.sample(available_for_path, path_length)
                used_intermediaries.update(path)
                paths.append(path)

        if not paths:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Flatten all intermediaries
        all_intermediaries = [acc for path in paths for acc in path]

        return EntitySelection(
            central_entities=[source_account_id, beneficiary_account_id],
            # Store as flat list, we'll reconstruct paths in temporal
            peripheral_entities=all_intermediaries
        )


class ParallelStructureTemporal(TemporalComponent):
    """
    Temporal: Funds are split from source, travel through parallel paths,
    and consolidate at beneficiary. Amounts are structured below thresholds.
    """

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences = []

        if not entity_selection.central_entities or len(entity_selection.central_entities) < 2:
            return sequences

        source_account_id = entity_selection.central_entities[0]
        beneficiary_account_id = entity_selection.central_entities[1]
        intermediary_accounts = entity_selection.peripheral_entities

        if not intermediary_accounts:
            return sequences

        pattern_config = self.params.get(
            "pattern_config", {}).get("parallelStructure", {})
        tx_params = pattern_config.get("transaction_params", {})

        total_amount_range = tx_params.get(
            "total_amount_range", [50000, 200000])
        structuring_percentage = tx_params.get(
            "structuring_percentage", 0.9)  # 90% structured
        hop_delay_hours = tx_params.get("hop_delay_hours", [1, 24])

        # Reconstruct paths from intermediaries
        # For simplicity, divide intermediaries evenly into paths
        min_hops_per_path = pattern_config.get("min_hops_per_path", 2)
        num_paths = max(1, len(intermediary_accounts) // min_hops_per_path)

        paths = []
        accounts_per_path = len(intermediary_accounts) // num_paths

        for i in range(num_paths):
            start_idx = i * accounts_per_path
            if i == num_paths - 1:
                # Last path gets remaining accounts
                path = intermediary_accounts[start_idx:]
            else:
                end_idx = start_idx + accounts_per_path
                path = intermediary_accounts[start_idx:end_idx]

            if path:
                paths.append(path)

        # Calculate timing
        time_span_days = (
            self.time_span["end_date"] - self.time_span["start_date"]).days
        # Need time for full pattern
        max_start_offset = max(0, time_span_days - 14)
        start_day_offset = random.randint(0, max_start_offset)
        pattern_start_time = self.time_span["start_date"] + \
            datetime.timedelta(days=start_day_offset)

        # Total amount to launder
        total_amount = random.uniform(
            total_amount_range[0], total_amount_range[1])

        # Get source currency for structuring
        source_currency = self.graph.nodes[source_account_id].get(
            "currency", "EUR")

        all_transactions = []
        current_time = pattern_start_time

        # Phase 1: Split funds from source to first hop of each path
        amounts_per_path = []
        remaining_amount = total_amount

        for i, path in enumerate(paths):
            if not path:
                continue

            first_hop = path[0]

            # Determine amount for this path
            if i == len(paths) - 1:
                # Last path gets remaining
                path_amount = remaining_amount
            else:
                # Random split with some variance
                proportion = 1.0 / len(paths)
                path_amount = total_amount * \
                    proportion * random.uniform(0.8, 1.2)
                remaining_amount -= path_amount

            amounts_per_path.append(path_amount)

            # Structure the amount if needed
            if random.random() < structuring_percentage:
                # Split into multiple structured transactions
                structured_amounts = self.generate_structured_amounts(
                    count=random.randint(2, 5),
                    base_amount=path_amount / 3,  # Rough split
                    target_currency=source_currency
                )

                # Adjust to match total
                scale_factor = path_amount / sum(structured_amounts)
                structured_amounts = [
                    amt * scale_factor for amt in structured_amounts]
            else:
                structured_amounts = [path_amount]

            # Create transactions
            for amount in structured_amounts:
                tx_time = current_time + datetime.timedelta(
                    minutes=random.randint(0, 60)  # Within an hour
                )

                tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                    src_id=source_account_id,
                    dest_id=first_hop,
                    timestamp=tx_time,
                    amount=round(amount, 2),
                    transaction_type=TransactionType.TRANSFER,
                    is_fraudulent=True
                )

                all_transactions.append(
                    (source_account_id, first_hop, tx_attrs))

        # Phase 2: Move funds through parallel paths
        for path_idx, path in enumerate(paths):
            if not path or path_idx >= len(amounts_per_path):
                continue

            path_amount = amounts_per_path[path_idx]
            current_amount = path_amount

            # Move through hops in the path
            for hop_idx in range(len(path) - 1):
                src = path[hop_idx]
                dest = path[hop_idx + 1]

                # Delay between hops
                hop_delay = random.uniform(
                    hop_delay_hours[0], hop_delay_hours[1])
                current_time += datetime.timedelta(hours=hop_delay)

                # Small fee at each hop
                fee = current_amount * random.uniform(0.01, 0.02)
                current_amount -= fee

                tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                    src_id=src,
                    dest_id=dest,
                    timestamp=current_time,
                    amount=round(current_amount, 2),
                    transaction_type=TransactionType.TRANSFER,
                    is_fraudulent=True
                )

                all_transactions.append((src, dest, tx_attrs))

        # Phase 3: Consolidate at beneficiary
        consolidation_time = current_time + datetime.timedelta(
            hours=random.uniform(hop_delay_hours[0], hop_delay_hours[1])
        )

        for path_idx, path in enumerate(paths):
            if not path or path_idx >= len(amounts_per_path):
                continue

            last_hop = path[-1]

            # Transfer from last hop to beneficiary
            # Amount is what's left after fees
            # Rough estimate after all fees
            final_amount = amounts_per_path[path_idx] * 0.9

            tx_attrs = PatternInjector(self.graph_generator, self.params)._create_transaction_edge(
                src_id=last_hop,
                dest_id=beneficiary_account_id,
                timestamp=consolidation_time +
                datetime.timedelta(minutes=random.randint(0, 30)),
                amount=round(final_amount, 2),
                transaction_type=TransactionType.TRANSFER,
                is_fraudulent=True
            )

            all_transactions.append(
                (last_hop, beneficiary_account_id, tx_attrs))

        # Sort by timestamp and create sequences
        all_transactions.sort(key=lambda x: x[2].timestamp)

        if all_transactions:
            # Group into logical sequences
            # Sequence 1: Initial structuring/splitting
            splitting_txs = [
                tx for tx in all_transactions if tx[0] == source_account_id]
            if splitting_txs:
                sequences.append(TransactionSequence(
                    transactions=splitting_txs,
                    sequence_name="parallel_splitting",
                    start_time=splitting_txs[0][2].timestamp,
                    duration=splitting_txs[-1][2].timestamp - splitting_txs[0][2].timestamp if len(
                        splitting_txs) > 1 else datetime.timedelta(0)
                ))

            # Sequence 2: Path hopping
            hopping_txs = [tx for tx in all_transactions
                           if tx[0] != source_account_id and tx[1] != beneficiary_account_id]
            if hopping_txs:
                sequences.append(TransactionSequence(
                    transactions=hopping_txs,
                    sequence_name="parallel_hopping",
                    start_time=hopping_txs[0][2].timestamp,
                    duration=hopping_txs[-1][2].timestamp - hopping_txs[0][2].timestamp if len(
                        hopping_txs) > 1 else datetime.timedelta(0)
                ))

            # Sequence 3: Consolidation
            consolidation_txs = [
                tx for tx in all_transactions if tx[1] == beneficiary_account_id]
            if consolidation_txs:
                sequences.append(TransactionSequence(
                    transactions=consolidation_txs,
                    sequence_name="parallel_consolidation",
                    start_time=consolidation_txs[0][2].timestamp,
                    duration=consolidation_txs[-1][2].timestamp - consolidation_txs[0][2].timestamp if len(
                        consolidation_txs) > 1 else datetime.timedelta(0)
                ))

        return sequences


class ParallelStructurePattern(CompositePattern):
    """Injects parallel structure pattern (smurfing + path hopping)"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = ParallelStructureStructural(
            graph_generator, params)
        temporal_component = ParallelStructureTemporal(graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "ParallelStructure"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
