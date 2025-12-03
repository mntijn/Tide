import datetime
from typing import List, Dict, Any, Tuple

from ..base import (
    StructuralComponent,
    TemporalComponent,
    EntitySelection,
    TransactionSequence,
    CompositePattern,
    PatternInjector,
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
    """
    Temporal component: generate camouflage transactions that mimic legitimate
    background activity statistics (rate and amount ranges).

    Fraudster grooming behaviour should look statistically identical to normal
    users in edge attributes so that models must rely on graph structure.
    """

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        fraudster_accounts = entity_selection.central_entities
        legit_accounts = entity_selection.peripheral_entities

        if not fraudster_accounts or not legit_accounts:
            return

        # Copy configuration from legitimate random background activity so that
        # fraudster "grooming" behaviour is statistically indistinguishable
        # from normal users in edge attributes (rate, amounts, timing).
        background_cfg = self.params.get("backgroundPatterns", {})
        random_payments_cfg = background_cfg.get("randomPayments", {})

        # Use the same per-account daily transaction rate as legitimate random payments.
        tx_rate = self.params.get("transaction_rates", {}).get(
            "per_account_per_day", 0.2
        )

        # Amount ranges: reuse legitimate ranges, especially for payments.
        legit_amount_ranges = random_payments_cfg.get(
            "amount_ranges",
            {
                "payment": [10.0, 2000.0],
                "transfer": [5.0, 800.0],
            },
        )
        payment_range = legit_amount_ranges.get("payment", [10.0, 2000.0])

        start_date: datetime.datetime = self.time_span["start_date"]
        end_date: datetime.datetime = self.time_span["end_date"]
        total_days = (end_date - start_date).days

        if total_days <= 0:
            total_days = 1

        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Respect tx_budget if set
        budget = getattr(self, "tx_budget", None)
        tx_count = 0

        # Generate camouflage transactions for each fraudster account
        for fraudster_account_id, fraudster_entity_id in fraudster_accounts:
            # Match legitimate background rate as closely as possible.
            daily_rate = tx_rate

            # Total expected transactions over time span
            expected_total_txs = int(daily_rate * total_days)
            if expected_total_txs == 0:
                # Ensure at least minimal background activity to avoid trivial patterns
                expected_total_txs = random_instance.randint(1, 3)

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
                if budget is not None and tx_count >= budget:
                    return

                if not available_counterparts:
                    available_counterparts = legit_accounts.copy()
                    random_instance.shuffle(available_counterparts)

                counterpart_account_id, counterpart_entity_id = available_counterparts.pop()

                # Determine transaction amount using *legitimate* payment range.
                # This deliberately removes the previous "mostly small" tell.
                amount = round(
                    random_instance.uniform(
                        payment_range[0], payment_range[1]), 2
                )

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

                tx_count += 1
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
