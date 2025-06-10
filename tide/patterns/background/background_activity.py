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
        # Filter to get only legitimate entities and their accounts
        legit_entities = self.graph_generator.entity_clusters.get("legit", [])

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
        total_seconds = total_days * 86400  # days to seconds

        tx_rate = self.params.get("transaction_rates", {}).get(
            "per_account_per_day", 0.2)

        # Get configuration for random payments
        random_payments_config = self.params.get(
            "backgroundPatterns", {}).get("randomPayments", {})

        # Calculate total expected transactions (using configurable multiplier)
        rate_multiplier_range = random_payments_config.get(
            "legit_rate_multiplier", [0.3, 0.8])
        rate_multiplier = random_instance.uniform(
            rate_multiplier_range[0], rate_multiplier_range[1])
        total_expected_txs = int(
            tx_rate * total_days * len(legit_accounts) * rate_multiplier)
        if total_expected_txs == 0:
            return sequences

        # Get transaction type probabilities from config
        tx_type_probs = random_payments_config.get("transaction_type_probabilities", {
            "transfer": 0.4, "payment": 0.3, "deposit": 0.15, "withdrawal": 0.15
        })
        transaction_types = [
            (TransactionType.TRANSFER, tx_type_probs.get("transfer", 0.4)),
            (TransactionType.PAYMENT, tx_type_probs.get("payment", 0.3)),
            (TransactionType.DEPOSIT, tx_type_probs.get("deposit", 0.15)),
            (TransactionType.WITHDRAWAL, tx_type_probs.get("withdrawal", 0.15))
        ]

        # Generate transaction data
        transactions: List[Tuple[str, str, TransactionAttributes]] = []
        pattern_injector = PatternInjector(self.graph_generator, self.params)

        for i in range(total_expected_txs):
            # Random timestamp
            random_seconds = random_instance.randint(0, total_seconds - 1)
            timestamp = start_date + datetime.timedelta(seconds=random_seconds)

            # Select transaction type based on probabilities
            rand_val = random_instance.random()
            cumulative_prob = 0.0
            selected_tx_type = TransactionType.TRANSFER
            for tx_type, prob in transaction_types:
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    selected_tx_type = tx_type
                    break

            # Get amount ranges from config
            amount_ranges = random_payments_config.get("amount_ranges", {})

            # Generate amount based on transaction type
            if selected_tx_type == TransactionType.PAYMENT:
                payment_range = amount_ranges.get("payment", [10.0, 2000.0])
                amount = round(random_instance.uniform(
                    payment_range[0], payment_range[1]), 2)
            elif selected_tx_type == TransactionType.TRANSFER:
                transfer_range = amount_ranges.get("transfer", [5.0, 800.0])
                amount = round(random_instance.uniform(
                    transfer_range[0], transfer_range[1]), 2)
            elif selected_tx_type in [TransactionType.DEPOSIT, TransactionType.WITHDRAWAL]:
                cash_range = amount_ranges.get(
                    "cash_operations", [20.0, 500.0])
                base_amount = random_instance.uniform(
                    cash_range[0], cash_range[1])
                amount = round(base_amount / 10) * 10  # Round to nearest 10
            else:
                # Default fallback using transfer range
                transfer_range = amount_ranges.get("transfer", [5.0, 800.0])
                amount = round(random_instance.uniform(
                    transfer_range[0], transfer_range[1]), 2)

            # Select source and destination accounts (ensure different accounts)
            src_account = random_instance.choice(legit_accounts)
            dest_account = random_instance.choice(legit_accounts)
            if src_account == dest_account and len(legit_accounts) > 1:
                available = [
                    acc for acc in legit_accounts if acc != src_account]
                dest_account = random_instance.choice(available)

            # Create transaction
            tx_attrs = pattern_injector._create_transaction_edge(
                src_id=src_account,
                dest_id=dest_account,
                timestamp=timestamp,
                amount=float(amount),
                transaction_type=selected_tx_type,
                is_fraudulent=False,
            )
            transactions.append((src_account, dest_account, tx_attrs))

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
