"""
Legitimate cash operations pattern.

This pattern injects realistic cash deposits and withdrawals for legitimate users.
Without this, cash operations only appear in fraud patterns, making any cash
transaction an instant fraud signal (data leakage).

Examples of legitimate cash operations:
- ATM withdrawals for daily spending
- Depositing cash income (tips, freelance work, sales)
- Small business cash handling
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


class LegitimateCashStructural(StructuralComponent):
    """Selects legitimate accounts that will have cash operations."""

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

        # Mix in 50% of fraud accounts (fraudsters also use ATMs for normal cash needs)
        if fraud_accounts:
            selected_fraud = random_instance.sample(
                fraud_accounts, min(len(fraud_accounts), int(len(fraud_accounts) * 0.9)))
            legit_accounts.extend(selected_fraud)

        legit_accounts = deduplicate_preserving_order(legit_accounts)

        if len(legit_accounts) < 2:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Config-driven fraction of users who do cash operations
        cash_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateCashOperations", {}
        )
        cash_user_fraction = cash_cfg.get("cash_user_fraction", 0.3)

        num_cash_users = max(1, int(len(legit_accounts) * cash_user_fraction))
        selected_cash_users = random_instance.sample(
            legit_accounts, min(num_cash_users, len(legit_accounts))
        )

        return EntitySelection(
            central_entities=selected_cash_users,
            peripheral_entities=[]
        )


class LegitimateCashTemporal(TemporalComponent):
    """
    Generates legitimate cash deposits and withdrawals.

    This creates the same transaction types (DEPOSIT, WITHDRAWAL) that fraud
    patterns use, so that these types are no longer unique fraud signals.
    """

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        cash_users = entity_selection.central_entities

        if not cash_users:
            return

        cash_account_id = self.graph_generator.cash_account_id
        if not cash_account_id:
            return

        start_date = self.time_span["start_date"]
        end_date = self.time_span["end_date"]
        total_days = max(1, (end_date - start_date).days)

        # Config
        cash_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateCashOperations", {}
        )
        ops_per_month = cash_cfg.get("operations_per_month_range", [2, 8])
        withdrawal_range = cash_cfg.get(
            "withdrawal_amount_range", [20.0, 500.0])
        deposit_prob = cash_cfg.get("deposit_probability", 0.3)

        # Tiered deposit amounts (small tips to large car sale deposits)
        deposit_ranges = cash_cfg.get("deposit_amount_ranges", {
            "small": [50.0, 500.0],
            "medium": [500.0, 2000.0],
            "large": [2000.0, 5000.0],
            "very_large": [5000.0, 9500.0]
        })
        deposit_probs = cash_cfg.get("deposit_probabilities", {
            "small": 0.5,
            "medium": 0.3,
            "large": 0.15,
            "very_large": 0.05
        })
        tier_names = list(deposit_probs.keys())
        tier_weights = [deposit_probs[t] for t in tier_names]

        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Respect tx_budget if set
        budget = getattr(self, "tx_budget", None)
        tx_count = 0

        total_months = max(1, total_days // 30)

        for account_id in cash_users:
            # Random number of cash operations for this account
            num_ops = random_instance.randint(
                ops_per_month[0] * total_months,
                ops_per_month[1] * total_months
            )
            num_ops = max(1, num_ops)

            for _ in range(num_ops):
                if budget is not None and tx_count >= budget:
                    return

                # Random timestamp
                random_day = random_instance.randint(0, total_days - 1)
                # ATM hours: 7am - 10pm
                random_hour = random_instance.randint(7, 22)
                random_minute = random_instance.randint(0, 59)

                tx_time = start_date + datetime.timedelta(
                    days=random_day, hours=random_hour, minutes=random_minute
                )

                # Decide deposit or withdrawal
                is_deposit = random_instance.random() < deposit_prob

                if is_deposit:
                    # Cash deposit - select tier based on probabilities
                    tier = random_instance.choices(
                        tier_names, weights=tier_weights)[0]
                    tier_range = deposit_ranges.get(tier, [50.0, 500.0])
                    amount = round(
                        random_instance.uniform(
                            tier_range[0], tier_range[1]), 2
                    )
                    # Round to realistic denominations
                    amount = round(amount / 10) * 10

                    tx_attrs = pattern_injector._create_transaction_edge(
                        src_id=cash_account_id,
                        dest_id=account_id,
                        timestamp=tx_time,
                        amount=amount,
                        transaction_type=TransactionType.DEPOSIT,
                        is_fraudulent=False,
                    )
                    tx_count += 1
                    yield (cash_account_id, account_id, tx_attrs)
                else:
                    # Cash withdrawal (ATM, spending money)
                    amount = round(
                        random_instance.uniform(
                            withdrawal_range[0], withdrawal_range[1]
                        ), 2
                    )
                    # ATM withdrawals are often in $20 increments
                    amount = round(amount / 20) * 20
                    amount = max(20.0, amount)

                    tx_attrs = pattern_injector._create_transaction_edge(
                        src_id=account_id,
                        dest_id=cash_account_id,
                        timestamp=tx_time,
                        amount=amount,
                        transaction_type=TransactionType.WITHDRAWAL,
                        is_fraudulent=False,
                    )
                    tx_count += 1
                    yield (account_id, cash_account_id, tx_attrs)

    def generate_transaction_sequences(
        self, entity_selection: EntitySelection
    ) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        cash_users = entity_selection.central_entities

        if not cash_users:
            return sequences

        transaction_generator = self._generate_transactions_stream(
            entity_selection)

        sequences.append(
            TransactionSequence(
                transactions=transaction_generator,
                sequence_name="legitimate_cash_operations",
                start_time=self.time_span.get("start_date"),
                duration=self.time_span.get("end_date")
                - self.time_span.get("start_date"),
            )
        )
        return sequences


class LegitimateCashOperationsPattern(CompositePattern):
    """Pattern for legitimate cash deposits and withdrawals."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = LegitimateCashStructural(
            graph_generator, params)
        temporal_component = LegitimateCashTemporal(graph_generator, params)
        super().__init__(
            structural_component, temporal_component, graph_generator, params
        )

    @property
    def pattern_name(self) -> str:
        return "LegitimateCashOperations"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
