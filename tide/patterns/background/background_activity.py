import datetime
import numpy as np
from typing import List, Dict, Any, Tuple

from ..base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ...datastructures.enums import NodeType, TransactionType
from ...datastructures.attributes import TransactionAttributes
from ...utils.random_instance import random_instance


class NonFraudulentRandomStructural(StructuralComponent):
    """Structural component: select legitimate entities for random daily transactions."""

    @property
    def num_required_entities(self) -> int:
        return 2  # Need at least 2 entities for transactions

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        # Use pre-computed account clusters for O(1) lookup instead of O(n*m) traversal
        if hasattr(self.graph_generator, 'account_clusters'):
            legit_accounts = self.graph_generator.account_clusters.get(
                "legit", [])
            if not legit_accounts:
                print(
                    "DEBUG [RandomPayments]: No legit accounts found in pre-computed clusters!")
                return EntitySelection(central_entities=[], peripheral_entities=[])
        else:
            # Fallback to original method if clusters not available
            legit_entities = self.graph_generator.entity_clusters.get(
                "legit", [])
            if not legit_entities:
                print("DEBUG [RandomPayments]: No legit entities found!")
                return EntitySelection(central_entities=[], peripheral_entities=[])

            # Get accounts belonging to legitimate entities using the helper function
            legit_accounts = []
            for entity_id in legit_entities:
                entity_accounts = self._get_owned_accounts(entity_id)
                legit_accounts.extend(entity_accounts)

            # Remove duplicates while preserving order
            from ..base import deduplicate_preserving_order
            legit_accounts = deduplicate_preserving_order(legit_accounts)

        return EntitySelection(central_entities=legit_accounts, peripheral_entities=[])


class RandomPaymentsTemporal(TemporalComponent):
    """Temporal component: generate diverse random transactions between legitimate accounts."""

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        legit_accounts = entity_selection.central_entities
        if len(legit_accounts) < 2:
            return sequences

        start_date: datetime.datetime = self.time_span["start_date"]
        end_date: datetime.datetime = self.time_span["end_date"]
        total_days = max(1, (end_date - start_date).days)

        tx_rate = self.params.get("transaction_rates", {}).get(
            "per_account_per_day", 0.2)

        # Get configuration for random payments
        random_payments_config = self.params.get(
            "backgroundPatterns", {}).get("randomPayments", {})

        total_expected_txs = int(
            tx_rate * total_days * len(legit_accounts))

        if total_expected_txs == 0:
            return sequences

        print(
            f"RandomPayments: Generating {total_expected_txs:,} transactions for {len(legit_accounts):,} accounts")

        # Pre-compute transaction type selection for efficiency
        tx_type_probs = random_payments_config.get("transaction_type_probabilities", {
            "transfer": 0.4, "payment": 0.3, "deposit": 0.15, "withdrawal": 0.15
        })
        transaction_types = [
            (TransactionType.TRANSFER, tx_type_probs.get("transfer", 0.4)),
            (TransactionType.PAYMENT, tx_type_probs.get("payment", 0.3)),
            (TransactionType.DEPOSIT, tx_type_probs.get("deposit", 0.15)),
            (TransactionType.WITHDRAWAL, tx_type_probs.get("withdrawal", 0.15))
        ]

        # OPTIMIZATION: Use vectorized operations for bulk generation
        total_minutes = total_days * 24 * 60  # days to minutes

        # Generate all random values at once using numpy vectorized operations
        random_timestamps = np.random.randint(
            0, total_minutes, size=total_expected_txs)

        # Vectorized transaction type selection using numpy choice with probabilities
        tx_types = [tx_type for tx_type, _ in transaction_types]
        tx_probs = [prob for _, prob in transaction_types]
        selected_types = np.random.choice(
            tx_types, size=total_expected_txs, p=tx_probs)

        # Vectorized account selection
        legit_accounts_array = np.array(legit_accounts)
        src_accounts = np.random.choice(
            legit_accounts_array, size=total_expected_txs)
        dest_accounts = np.random.choice(
            legit_accounts_array, size=total_expected_txs)

        # Handle case where src == dest (vectorized)
        same_account_mask = src_accounts == dest_accounts
        if np.any(same_account_mask) and len(legit_accounts) > 1:
            # For accounts that are the same, pick different destinations
            num_to_replace = np.sum(same_account_mask)
            dest_accounts[same_account_mask] = np.random.choice(
                legit_accounts_array, size=num_to_replace)

        # Vectorized amount generation based on transaction types
        amount_ranges = random_payments_config.get("amount_ranges", {})
        amounts = np.zeros(total_expected_txs)

        # Generate amounts for each transaction type
        for tx_type in tx_types:
            mask = selected_types == tx_type
            count = np.sum(mask)
            if count == 0:
                continue

            if tx_type == TransactionType.PAYMENT:
                payment_range = amount_ranges.get("payment", [10.0, 2000.0])
                amounts[mask] = np.round(np.random.uniform(
                    payment_range[0], payment_range[1], size=count), 2)
            elif tx_type == TransactionType.TRANSFER:
                transfer_range = amount_ranges.get("transfer", [5.0, 800.0])
                amounts[mask] = np.round(np.random.uniform(
                    transfer_range[0], transfer_range[1], size=count), 2)
            elif tx_type in [TransactionType.DEPOSIT, TransactionType.WITHDRAWAL]:
                cash_range = amount_ranges.get(
                    "cash_operations", [20.0, 500.0])
                base_amounts = np.random.uniform(
                    cash_range[0], cash_range[1], size=count)
                amounts[mask] = np.round(
                    base_amounts / 10) * 10  # Round to nearest 10
            else:
                # Default fallback using transfer range
                transfer_range = amount_ranges.get("transfer", [5.0, 800.0])
                amounts[mask] = np.round(np.random.uniform(
                    transfer_range[0], transfer_range[1], size=count), 2)

        # Create all transaction objects (only remaining loop - unavoidable for object creation)
        transactions: List[Tuple[str, str, TransactionAttributes]] = []
        pattern_injector = PatternInjector(self.graph_generator, self.params)

        for i in range(total_expected_txs):
            timestamp = start_date + \
                datetime.timedelta(minutes=int(random_timestamps[i]))

            tx_attrs = pattern_injector._create_transaction_edge(
                src_id=src_accounts[i],
                dest_id=dest_accounts[i],
                timestamp=timestamp,
                amount=float(amounts[i]),
                transaction_type=selected_types[i],
                is_fraudulent=False,
            )
            transactions.append((src_accounts[i], dest_accounts[i], tx_attrs))

        # Create transaction sequences
        if transactions:
            # Sort by timestamp for sequence creation
            transactions.sort(key=lambda x: x[2].timestamp)

            sequence_name = "legitimate_random_activity"
            timestamps = [tx[2].timestamp for tx in transactions]
            sequences.append(
                TransactionSequence(
                    transactions=transactions,
                    sequence_name=sequence_name,
                    start_time=min(timestamps),
                    duration=max(timestamps) - min(timestamps),
                )
            )
        return sequences


class RandomPaymentsPattern(CompositePattern):
    """Random payments pattern for legitimate baseline activity between non-fraudulent entities."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = NonFraudulentRandomStructural(
            graph_generator, params)
        temporal_component = RandomPaymentsTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "RandomPayments"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
