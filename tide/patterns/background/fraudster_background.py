import datetime
from typing import List, Dict, Any, Tuple

from ..base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ...datastructures.enums import NodeType, TransactionType
from ...datastructures.attributes import TransactionAttributes
from ...utils.random_instance import random_instance


class FraudsterBackgroundStructural(StructuralComponent):
    """Structural component: select fraudulent entities for background camouflage activity."""

    @property
    def num_required_entities(self) -> int:
        return 1  # Need at least 1 fraudulent entity

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        # Get fraudulent entities only
        fraudulent_entities = self.graph_generator.entity_clusters.get(
            "fraudulent", [])
        if not fraudulent_entities:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Get accounts for fraudulent entities
        fraudster_accounts = []

        for fraudster_id in fraudulent_entities:
            fraudster_account_ids = self._get_owned_accounts(fraudster_id)
            for account_id in fraudster_account_ids:
                fraudster_accounts.append((account_id, fraudster_id))

        if not fraudster_accounts:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Also get legitimate accounts as transaction counterparts
        legit_entities = self.graph_generator.entity_clusters.get("legit", [])
        legit_accounts = []

        for legit_id in legit_entities:
            legit_account_ids = self._get_owned_accounts(legit_id)
            for account_id in legit_account_ids:
                legit_accounts.append((account_id, legit_id))

        # Central = fraudster accounts, Peripheral = legitimate accounts (counterparts)
        return EntitySelection(
            # [(account_id, fraudster_id), ...]
            central_entities=fraudster_accounts,
            # [(account_id, legit_id), ...]
            peripheral_entities=legit_accounts
        )


class FraudsterBackgroundTemporal(TemporalComponent):
    """Temporal component: generate low-frequency, small-amount camouflage transactions."""

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        fraudster_accounts = entity_selection.central_entities
        legit_accounts = entity_selection.peripheral_entities

        if not fraudster_accounts or not legit_accounts:
            return

        # Get fraudster background configuration
        fraudster_config = self.params.get(
            "backgroundPatterns", {}).get("fraudsterBackground", {})

        # Rate multiplier (much lower than legitimate activity)
        rate_multiplier_range = fraudster_config.get(
            "fraudster_rate_multiplier", [0.1, 0.5])

        # Amount ranges (smaller amounts to stay under radar)
        amount_ranges = fraudster_config.get("amount_ranges", {
            "small_transactions": [5.0, 200.0],
            "medium_transactions": [200.0, 1000.0]
        })

        # Transaction size probabilities (mostly small)
        size_probs = fraudster_config.get("transaction_size_probabilities", {
            "small": 0.8,
            "medium": 0.2
        })

        start_date: datetime.datetime = self.time_span["start_date"]
        end_date: datetime.datetime = self.time_span["end_date"]
        total_days = (end_date - start_date).days

        if total_days <= 0:
            total_days = 1

        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Generate camouflage transactions for each fraudster account
        for fraudster_account_id, fraudster_entity_id in fraudster_accounts:
            # Determine this fraudster's activity rate (lower than legitimate)
            rate_multiplier = random_instance.uniform(
                rate_multiplier_range[0], rate_multiplier_range[1])

            # Base rate per day for this fraudster (much lower than legitimate background)
            base_daily_rate = 0.1  # Very low base rate
            daily_rate = base_daily_rate * rate_multiplier

            # Total expected transactions over time span
            expected_total_txs = int(daily_rate * total_days)
            if expected_total_txs == 0:
                expected_total_txs = random_instance.randint(
                    1, 3)  # Minimum camouflage activity

            # Generate random transaction dates
            tx_dates = []
            for _ in range(expected_total_txs):
                random_day = random_instance.randint(0, total_days - 1)
                random_hour = random_instance.randint(8, 18)  # Business hours
                random_minute = random_instance.randint(0, 59)

                tx_date = start_date + datetime.timedelta(
                    days=random_day, hours=random_hour, minutes=random_minute
                )
                tx_dates.append(tx_date)

            tx_dates.sort()  # Chronological order

            # Generate transactions
            available_counterparts = legit_accounts.copy()
            random_instance.shuffle(available_counterparts)

            for i, tx_date in enumerate(tx_dates):

                if not available_counterparts:
                    available_counterparts = legit_accounts.copy()
                    random_instance.shuffle(available_counterparts)

                counterpart_account_id, counterpart_entity_id = available_counterparts.pop()

                # Determine transaction amount (mostly small)
                if random_instance.random() < size_probs.get("small", 0.8):
                    amount_range = amount_ranges["small_transactions"]
                else:
                    amount_range = amount_ranges["medium_transactions"]

                amount = round(random_instance.uniform(
                    amount_range[0], amount_range[1]), 2)

                # Randomly choose direction (fraudster pays or receives)
                if random_instance.random() < 0.5:
                    # Fraudster pays (outgoing)
                    src_id = fraudster_account_id
                    dest_id = counterpart_account_id
                    tx_type = TransactionType.PAYMENT
                else:
                    # Fraudster receives (incoming)
                    src_id = counterpart_account_id
                    dest_id = fraudster_account_id
                    tx_type = TransactionType.PAYMENT

                tx_attrs = pattern_injector._create_transaction_edge(
                    src_id=src_id,
                    dest_id=dest_id,
                    timestamp=tx_date,
                    amount=float(amount),
                    transaction_type=tx_type,
                    is_fraudulent=False,  # These are camouflage, not fraudulent
                )

                yield (src_id, dest_id, tx_attrs)

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        fraudster_accounts = entity_selection.central_entities
        legit_accounts = entity_selection.peripheral_entities

        if not fraudster_accounts or not legit_accounts:
            return sequences

        # The transaction data is provided by a generator for memory efficiency
        transaction_generator = self._generate_transactions_stream(
            entity_selection)

        # The framework expects a TransactionSequence, so we wrap the generator.
        sequences.append(
            TransactionSequence(
                transactions=transaction_generator,
                sequence_name="fraudster_background_camouflage",
                start_time=self.time_span.get("start_date"),
                duration=self.time_span.get(
                    "end_date") - self.time_span.get("start_date"),
            )
        )

        return sequences


class FraudsterBackgroundPattern(CompositePattern):
    """Fraudster background camouflage pattern for low-frequency legitimate-looking transactions."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = FraudsterBackgroundStructural(
            graph_generator, params)
        temporal_component = FraudsterBackgroundTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "FraudsterBackground"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
