"""
Legitimate periodic payments pattern.

This pattern injects regular, repeating payments that occur at fixed intervals.
Without this, only fraud patterns have periodic timing signatures, making
regular intervals a fraud signal (data leakage).

Examples of legitimate periodic payments:
- Monthly bills (utilities, rent, insurance)
- Weekly subscriptions (meal delivery, cleaning services)
- Bi-weekly/monthly salary-like payments to self (savings transfers)
"""
import datetime
from typing import List, Dict, Any

from ..base import (
    StructuralComponent,
    TemporalComponent,
    EntitySelection,
    TransactionSequence,
    CompositePattern,
    PatternInjector,
    deduplicate_preserving_order,
)
from ...datastructures.enums import NodeType, TransactionType
from ...utils.random_instance import random_instance
from ...utils.amount_distributions import sample_lognormal_scalar


class LegitimatePeriodicStructural(StructuralComponent):
    """Selects legitimate accounts that make periodic payments."""

    @property
    def num_required_entities(self) -> int:
        return 2

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        # Get legitimate accounts + fraud accounts for mixing
        if hasattr(self.graph_generator, "account_clusters"):
            legit_accounts = list(
                self.graph_generator.account_clusters.get("legit", [])
            )
            fraud_accounts = list(
                self.graph_generator.account_clusters.get("fraudulent", [])
            )
        else:
            legit_entities = self.graph_generator.entity_clusters.get(
                "legit", [])
            fraud_entities = self.graph_generator.entity_clusters.get(
                "fraudulent", [])

            legit_accounts = []
            for entity_id in legit_entities:
                legit_accounts.extend(self._get_owned_accounts(entity_id))

            fraud_accounts = []
            for entity_id in fraud_entities:
                fraud_accounts.extend(self._get_owned_accounts(entity_id))

        # Mix in 50% of fraud accounts (fraudsters also have recurring bills)
        if fraud_accounts:
            selected_fraud = random_instance.sample(
                fraud_accounts, min(len(fraud_accounts), int(len(fraud_accounts) * 0.9)))
            legit_accounts.extend(selected_fraud)

        legit_accounts = deduplicate_preserving_order(legit_accounts)

        if len(legit_accounts) < 2:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Config-driven fraction of users who make periodic payments
        periodic_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimatePeriodicPayments", {}
        )
        periodic_user_fraction = periodic_cfg.get(
            "periodic_user_fraction", 0.4)

        num_periodic_users = max(
            1, int(len(legit_accounts) * periodic_user_fraction))
        selected_users = random_instance.sample(
            legit_accounts, min(num_periodic_users, len(legit_accounts))
        )

        # Remaining accounts can be recipients (utilities, services, etc.)
        remaining = [
            acc for acc in legit_accounts if acc not in selected_users]

        return EntitySelection(
            central_entities=selected_users,
            peripheral_entities=remaining if remaining else legit_accounts
        )


class LegitimatePeriodicTemporal(TemporalComponent):
    """
    Generates legitimate periodic payments at regular intervals.

    This creates the same timing signatures (weekly, bi-weekly, monthly) that
    fraud patterns use, so that periodic timing is no longer a unique fraud signal.
    """

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        payers = entity_selection.central_entities
        payees = entity_selection.peripheral_entities

        if not payers or not payees:
            return

        start_date = self.time_span["start_date"]
        end_date = self.time_span["end_date"]
        total_days = max(1, (end_date - start_date).days)

        # Config
        periodic_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimatePeriodicPayments", {}
        )
        intervals = periodic_cfg.get("payment_intervals", [7, 14, 30])
        subscriptions_per_user = periodic_cfg.get(
            "subscriptions_per_user_range", [1, 4])
        use_lognormal = periodic_cfg.get("use_lognormal", True)
        dist_config = self.params.get(
            "backgroundPatterns", {}
        ).get("amount_distributions", {}).get("payment", {})

        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Respect tx_budget if set
        budget = getattr(self, "tx_budget", None)
        tx_count = 0

        for payer_account in payers:
            # Each user has 1-4 recurring payments
            num_subscriptions = random_instance.randint(
                subscriptions_per_user[0], subscriptions_per_user[1]
            )

            for _ in range(num_subscriptions):
                # Pick a random recipient (utility company, service, etc.)
                payee_account = random_instance.choice(payees)
                if payee_account == payer_account:
                    continue

                # Pick a random interval
                interval_days = random_instance.choice(intervals)

                # Sample base amount from log-normal or uniform
                if use_lognormal:
                    base_amount = sample_lognormal_scalar(
                        "payment", config=dist_config)
                else:
                    base_amount = random_instance.uniform(20.0, 5000.0)

                # Random start day within first interval
                start_offset = random_instance.randint(
                    0, min(interval_days - 1, total_days - 1))
                current_date = start_date + \
                    datetime.timedelta(days=start_offset)

                # Generate all payments for this subscription
                while current_date <= end_date:
                    if budget is not None and tx_count >= budget:
                        return

                    # Small variation in amount (±5%)
                    amount = round(
                        base_amount * random_instance.uniform(0.95, 1.05), 2
                    )

                    # Payment typically at consistent time of day
                    tx_time = current_date.replace(
                        # Early morning auto-payments
                        hour=random_instance.randint(6, 10),
                        minute=random_instance.randint(0, 59)
                    )

                    tx_attrs = pattern_injector._create_transaction_edge(
                        src_id=payer_account,
                        dest_id=payee_account,
                        timestamp=tx_time,
                        amount=amount,
                        transaction_type=TransactionType.PAYMENT,
                        is_fraudulent=False,
                    )
                    tx_count += 1
                    yield (payer_account, payee_account, tx_attrs)

                    # Next payment
                    current_date += datetime.timedelta(days=interval_days)

    def generate_transaction_sequences(
        self, entity_selection: EntitySelection
    ) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        payers = entity_selection.central_entities

        if not payers:
            return sequences

        transaction_generator = self._generate_transactions_stream(
            entity_selection)

        sequences.append(
            TransactionSequence(
                transactions=transaction_generator,
                sequence_name="legitimate_periodic_payments",
                start_time=self.time_span.get("start_date"),
                duration=self.time_span.get("end_date")
                - self.time_span.get("start_date"),
            )
        )
        return sequences


class LegitimatePeriodicPaymentsPattern(CompositePattern):
    """Pattern for legitimate periodic/recurring payments."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = LegitimatePeriodicStructural(
            graph_generator, params)
        temporal_component = LegitimatePeriodicTemporal(
            graph_generator, params)
        super().__init__(
            structural_component, temporal_component, graph_generator, params
        )

    @property
    def pattern_name(self) -> str:
        return "LegitimatePeriodicPayments"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
