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

            print(
                "DEBUG [RandomPayments]: No legit accounts found in pre-computed clusters!")
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

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        """
        A generator that yields individual transactions in chronological order.
        This approach avoids loading all transactions into memory at once.
        """
        legit_accounts = entity_selection.central_entities
        if len(legit_accounts) < 2:
            return

        start_date: datetime.datetime = self.time_span["start_date"]
        end_date: datetime.datetime = self.time_span["end_date"]
        total_days = max(1, (end_date - start_date).days)

        tx_rate = self.params.get("transaction_rates", {}).get(
            "per_account_per_day", 0.2)
        random_payments_config = self.params.get(
            "backgroundPatterns", {}).get("randomPayments", {})
        total_expected_txs = int(tx_rate * total_days * len(legit_accounts))

        if total_expected_txs == 0:
            return

        print(
            f"RandomPayments: Generating {total_expected_txs:,} transactions for {len(legit_accounts):,} accounts")

        tx_type_probs = random_payments_config.get("transaction_type_probabilities", {
            "transfer": 0.4, "payment": 0.3, "deposit": 0.15, "withdrawal": 0.15
        })

        # Use string keys for np.choice and map to enums later for efficiency
        tx_type_keys = list(tx_type_probs.keys())
        tx_probs_values = list(tx_type_probs.values())
        type_to_enum = {
            "transfer": TransactionType.TRANSFER,
            "payment": TransactionType.PAYMENT,
            "deposit": TransactionType.DEPOSIT,
            "withdrawal": TransactionType.WITHDRAWAL
        }

        # OPTIMIZATION: Generate timestamps and sort them first to ensure chronological order.
        total_minutes = total_days * 24 * 60
        random_minutes_offsets = np.random.randint(
            0, total_minutes, size=total_expected_txs)
        random_minutes_offsets.sort()  # Sort timestamps to generate transactions in order

        # Vectorized generation of other attributes
        selected_type_keys = np.random.choice(
            tx_type_keys, size=total_expected_txs, p=tx_probs_values)
        legit_accounts_array = np.array(legit_accounts)
        src_accounts = np.random.choice(
            legit_accounts_array, size=total_expected_txs)
        dest_accounts = np.random.choice(
            legit_accounts_array, size=total_expected_txs)

        # Vectorized amount generation
        amount_ranges = random_payments_config.get("amount_ranges", {})
        amounts = np.zeros(total_expected_txs)

        for tx_type_key in tx_type_keys:
            mask = selected_type_keys == tx_type_key
            count = np.sum(mask)
            if count == 0:
                continue

            tx_type_enum = type_to_enum[tx_type_key]

            if tx_type_enum == TransactionType.PAYMENT:
                payment_range = amount_ranges.get("payment", [10.0, 2000.0])
                amounts[mask] = np.round(np.random.uniform(
                    payment_range[0], payment_range[1], size=count), 2)
            elif tx_type_enum == TransactionType.TRANSFER:
                transfer_range = amount_ranges.get("transfer", [5.0, 800.0])
                amounts[mask] = np.round(np.random.uniform(
                    transfer_range[0], transfer_range[1], size=count), 2)
            elif tx_type_enum in [TransactionType.DEPOSIT, TransactionType.WITHDRAWAL]:
                cash_range = amount_ranges.get(
                    "cash_operations", [20.0, 500.0])
                base_amounts = np.random.uniform(
                    cash_range[0], cash_range[1], size=count)
                amounts[mask] = np.round(base_amounts / 10) * 10
            else:
                default_range = amount_ranges.get("transfer", [5.0, 800.0])
                amounts[mask] = np.round(np.random.uniform(
                    default_range[0], default_range[1], size=count), 2)

        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Yield transactions one by one instead of storing them in a list
        for i in range(total_expected_txs):
            # Resolve sender/receiver accounts ensuring they are not the same
            src, dest = src_accounts[i], dest_accounts[i]
            if src == dest:
                # Simple and fast retry logic for the rare cases of collision
                while dest == src:
                    dest = np.random.choice(legit_accounts_array)

            timestamp = start_date + \
                datetime.timedelta(minutes=int(random_minutes_offsets[i]))

            tx_attrs = pattern_injector._create_transaction_edge(
                src_id=src,
                dest_id=dest,
                timestamp=timestamp,
                amount=float(amounts[i]),
                transaction_type=type_to_enum[selected_type_keys[i]],
                is_fraudulent=False,
            )
            yield (src, dest, tx_attrs)

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        """
        Returns a list containing a single TransactionSequence.
        The sequence's 'transactions' attribute is a generator, not a list,
        to ensure memory-efficient processing.
        """
        sequences: List[TransactionSequence] = []
        legit_accounts = entity_selection.central_entities
        if len(legit_accounts) < 2:
            return sequences

        # The transaction data is provided by a generator for memory efficiency
        transaction_generator = self._generate_transactions_stream(
            entity_selection)

        # The framework expects a TransactionSequence, so we wrap the generator.
        # The consumer (CompositePattern.inject_pattern) can iterate over the generator.
        sequences.append(
            TransactionSequence(
                transactions=transaction_generator,
                sequence_name="legitimate_random_activity",
                # Timing metadata is less critical here as transactions are streamed
                start_time=self.time_span.get("start_date"),
                duration=self.time_span.get(
                    "end_date") - self.time_span.get("start_date"),
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

    def generate(self) -> None:
        """
        Generates and injects the random payment pattern into the graph.
        This method now handles the generator from the temporal component.
        """
        print(f"Generating pattern: {self.pattern_name}")
        entity_selection = self.structural.select_entities(
            self.graph_generator.get_all_entities()
        )

        if not entity_selection.central_entities:
            print(
                f"Skipping pattern {self.pattern_name} due to lack of suitable entities.")
            return

        transaction_generator = self.temporal.generate_transaction_sequences(
            entity_selection)

        # The pattern injector now consumes the generator, adding tx to the graph one by one
        self.pattern_injector.inject_transactions_from_generator(
            generator=transaction_generator,
            pattern_name=self.pattern_name,
            tag_assets=True
        )

    @property
    def pattern_name(self) -> str:
        return "RandomPayments"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
