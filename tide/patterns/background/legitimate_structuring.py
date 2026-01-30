"""
Legitimate structuring-like deposits pattern.

This pattern injects deposits that mimic structuring behavior but for legitimate reasons.
Without this, deposits in the $7,500-$9,999 range are almost exclusively fraudulent,
making amount thresholds a trivial fraud signal (data leakage).

Realistic scenarios modeled:
1. House down payment savings: Regular large deposits over months
2. Small business cash handling: Daily/weekly deposits below reporting threshold
3. Tax-conscious transfers: Staying below reporting requirements legally
4. Self-employed income: Irregular large cash deposits from freelance work
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


# Scenario configurations for realistic structuring-like behavior
STRUCTURING_SCENARIOS = {
    "house_savings": {
        "weight": 0.30,
        "amount_range": [8000.0, 9999.0],  # Saving large amounts
        "deposits_per_month": [1, 2],
        "description": "Saving for house down payment",
    },
    "small_business": {
        "weight": 0.35,
        "amount_range": [7500.0, 9500.0],  # Business cash handling
        "deposits_per_month": [4, 8],  # More frequent
        "description": "Small business cash deposits",
    },
    "freelance_income": {
        "weight": 0.25,
        "amount_range": [7500.0, 9900.0],  # Project payments
        "deposits_per_month": [2, 4],
        "description": "Freelance/contractor income",
    },
    "tax_conscious": {
        "weight": 0.10,
        "amount_range": [9000.0, 9999.0],  # Just below threshold
        "deposits_per_month": [1, 3],
        "description": "Tax-conscious large transfers",
    },
}


class LegitimateStructuringStructural(StructuralComponent):
    """Selects legitimate accounts that will have structuring-like deposits."""

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

        # Mix in fraud accounts so they also have legitimate structuring-like behavior
        if fraud_accounts:
            selected_fraud = random_instance.sample(
                fraud_accounts, min(len(fraud_accounts),
                                    int(len(fraud_accounts) * 0.5))
            )
            legit_accounts.extend(selected_fraud)

        legit_accounts = deduplicate_preserving_order(legit_accounts)

        if len(legit_accounts) < 2:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Config-driven fraction of users who do structuring-like deposits
        structuring_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateStructuring", {}
        )
        user_fraction = structuring_cfg.get("user_fraction", 0.08)

        num_users = max(1, int(len(legit_accounts) * user_fraction))
        selected_users = random_instance.sample(
            legit_accounts, min(num_users, len(legit_accounts))
        )

        return EntitySelection(
            central_entities=selected_users,
            peripheral_entities=[]
        )


class LegitimateStructuringTemporal(TemporalComponent):
    """
    Generates legitimate deposits in the structuring amount range ($7,500-$9,999).

    This creates the same amount signatures that fraud structuring uses,
    so that amount thresholds are no longer unique fraud signals.
    """

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        structuring_users = entity_selection.central_entities

        if not structuring_users:
            return

        cash_account_id = self.graph_generator.cash_account_id
        if not cash_account_id:
            return

        start_date = self.time_span["start_date"]
        end_date = self.time_span["end_date"]
        total_days = max(1, (end_date - start_date).days)
        total_months = max(1, total_days // 30)

        # Config
        structuring_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateStructuring", {}
        )

        # Override ranges from config if provided
        config_amount_range = structuring_cfg.get("deposit_amount_range", None)
        config_deposits_per_month = structuring_cfg.get(
            "deposits_per_month_range", None)

        # Build scenario weights for random selection
        scenario_names = sorted(STRUCTURING_SCENARIOS.keys())
        scenario_weights = [STRUCTURING_SCENARIOS[s]["weight"]
                            for s in scenario_names]

        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Respect tx_budget if set
        budget = getattr(self, "tx_budget", None)
        tx_count = 0

        for account_id in structuring_users:
            if budget is not None and tx_count >= budget:
                return

            # Select a random scenario for this user
            scenario_name = random_instance.choices(
                scenario_names, weights=scenario_weights
            )[0]
            scenario = STRUCTURING_SCENARIOS[scenario_name]

            # Get scenario parameters (config overrides if provided)
            if config_amount_range:
                amount_range = config_amount_range
            else:
                amount_range = scenario["amount_range"]

            if config_deposits_per_month:
                deposits_per_month = config_deposits_per_month
            else:
                deposits_per_month = scenario["deposits_per_month"]

            # Calculate total deposits for this user over the time span
            monthly_deposits = random_instance.randint(
                deposits_per_month[0], deposits_per_month[1]
            )
            total_deposits = monthly_deposits * total_months

            # INTERLEAVING: Some users have burst deposits (like fraud "high_frequency")
            # This matches the fraud pattern's temporal clustering
            burst_probability = structuring_cfg.get("burst_probability", 0.35)
            use_burst_timing = random_instance.random() < burst_probability

            if use_burst_timing:
                # Generate deposits in burst clusters (fraud-like timing)
                # ~4 deposits per burst
                num_bursts = max(1, total_deposits // 4)
                deposits_remaining = total_deposits

                for burst_idx in range(num_bursts):
                    if budget is not None and tx_count >= budget:
                        return
                    if deposits_remaining <= 0:
                        break

                    # Burst start time
                    burst_day = random_instance.randint(0, total_days - 1)
                    burst_start = start_date + datetime.timedelta(
                        days=burst_day, hours=random_instance.randint(9, 14)
                    )

                    # 2-5 deposits per burst within 24 hours (like fraud "high_frequency")
                    burst_size = min(deposits_remaining,
                                     random_instance.randint(2, 5))
                    deposits_remaining -= burst_size

                    for i in range(burst_size):
                        if budget is not None and tx_count >= budget:
                            return

                        # Deposits within burst are hours apart (fraud-like)
                        offset_hours = random_instance.uniform(0, 8) * i
                        tx_time = burst_start + \
                            datetime.timedelta(hours=offset_hours)

                        # Generate amount in structuring range
                        amount = random_instance.uniform(
                            amount_range[0], amount_range[1])
                        amount = round(amount / 100) * 100
                        amount = max(amount_range[0], min(
                            amount, amount_range[1]))

                        tx_attrs = pattern_injector._create_transaction_edge(
                            src_id=cash_account_id,
                            dest_id=account_id,
                            timestamp=tx_time,
                            amount=round(amount, 2),
                            transaction_type=TransactionType.DEPOSIT,
                            is_fraudulent=False,
                        )
                        tx_count += 1
                        yield (cash_account_id, account_id, tx_attrs)
            else:
                # Original spread-out timing
                for _ in range(total_deposits):
                    if budget is not None and tx_count >= budget:
                        return

                    # Random day within time span
                    random_day = random_instance.randint(0, total_days - 1)
                    # Business hours for deposits
                    random_hour = random_instance.randint(9, 17)
                    random_minute = random_instance.randint(0, 59)

                    tx_time = start_date + datetime.timedelta(
                        days=random_day, hours=random_hour, minutes=random_minute
                    )

                    # Generate amount in structuring range
                    amount = random_instance.uniform(
                        amount_range[0], amount_range[1])
                    # Round to realistic denominations (nearest $100 or $50)
                    if random_instance.random() < 0.7:
                        amount = round(amount / 100) * 100
                    else:
                        amount = round(amount / 50) * 50
                    amount = max(amount_range[0], min(amount, amount_range[1]))

                    tx_attrs = pattern_injector._create_transaction_edge(
                        src_id=cash_account_id,
                        dest_id=account_id,
                        timestamp=tx_time,
                        amount=round(amount, 2),
                        transaction_type=TransactionType.DEPOSIT,
                        is_fraudulent=False,
                    )
                    tx_count += 1
                    yield (cash_account_id, account_id, tx_attrs)

    def generate_transaction_sequences(
        self, entity_selection: EntitySelection
    ) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        structuring_users = entity_selection.central_entities

        if not structuring_users:
            return sequences

        transaction_generator = self._generate_transactions_stream(
            entity_selection)

        sequences.append(
            TransactionSequence(
                transactions=transaction_generator,
                sequence_name="legitimate_structuring_deposits",
                start_time=self.time_span.get("start_date"),
                duration=self.time_span.get(
                    "end_date") - self.time_span.get("start_date"),
            )
        )
        return sequences


class LegitimateStructuringPattern(CompositePattern):
    """Pattern for legitimate deposits in the structuring amount range."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = LegitimateStructuringStructural(
            graph_generator, params)
        temporal_component = LegitimateStructuringTemporal(
            graph_generator, params)
        super().__init__(
            structural_component, temporal_component, graph_generator, params
        )

    @property
    def pattern_name(self) -> str:
        return "LegitimateStructuring"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
