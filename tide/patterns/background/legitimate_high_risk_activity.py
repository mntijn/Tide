"""
Legitimate high-risk entity activity pattern.

This pattern ensures that high-risk entities (those with high risk_score,
from high-risk countries, or in high-risk business categories) also engage
in normal, legitimate transaction activity.

Without this, the correlation between high-risk attributes and fraud is
too strong, making risk_score alone a near-perfect predictor (data leakage).

The pattern creates:
1. Normal business operations for high-risk businesses
2. Regular transfers for individuals in high-risk countries
3. Standard payments from high risk_score entities
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


class LegitimateHighRiskStructural(StructuralComponent):
    """Selects high-risk entities for legitimate activity generation."""

    @property
    def num_required_entities(self) -> int:
        return 2

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        # Get high-risk entities from multiple clusters
        high_risk_clusters = [
            "super_high_risk",
            "high_risk_score",
            "high_risk_countries",
            "high_risk_business_categories",
            "offshore_candidates",
        ]

        high_risk_entities = self.get_combined_clusters(high_risk_clusters)

        if not high_risk_entities:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Get accounts for these high-risk entities
        high_risk_accounts = []
        for entity_id in high_risk_entities:
            entity_accounts = self._get_owned_accounts(entity_id)
            high_risk_accounts.extend(entity_accounts)

        high_risk_accounts = deduplicate_preserving_order(high_risk_accounts)

        if len(high_risk_accounts) < 2:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Config-driven activity level
        hr_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateHighRiskActivity", {}
        )
        activity_fraction = hr_cfg.get("activity_fraction", 0.8)

        # Select most high-risk accounts (80% by default)
        num_active = max(1, int(len(high_risk_accounts) * activity_fraction))
        selected_accounts = random_instance.sample(
            high_risk_accounts, min(num_active, len(high_risk_accounts))
        )

        # Get all legitimate accounts as potential counterparties
        if hasattr(self.graph_generator, "account_clusters"):
            legit_accounts = list(
                self.graph_generator.account_clusters.get("legit", [])
            )
        else:
            legit_entities = self.graph_generator.entity_clusters.get(
                "legit", [])
            legit_accounts = []
            for entity_id in legit_entities:
                legit_accounts.extend(self._get_owned_accounts(entity_id))

        legit_accounts = deduplicate_preserving_order(legit_accounts)

        return EntitySelection(
            central_entities=selected_accounts,
            peripheral_entities=legit_accounts if legit_accounts else selected_accounts
        )


class LegitimateHighRiskTemporal(TemporalComponent):
    """
    Generates normal transaction activity for high-risk entities.

    This ensures that high-risk attributes don't correlate exclusively
    with fraudulent behavior - high-risk entities also do normal things.
    """

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        high_risk_accounts = entity_selection.central_entities
        counterparties = entity_selection.peripheral_entities

        if not high_risk_accounts or not counterparties:
            return

        start_date = self.time_span["start_date"]
        end_date = self.time_span["end_date"]
        total_days = max(1, (end_date - start_date).days)

        # Config
        hr_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateHighRiskActivity", {}
        )
        tx_per_account_range = hr_cfg.get(
            "transactions_per_account_range", [5, 20])
        amount_ranges = hr_cfg.get("amount_ranges", {
            "small": [20.0, 200.0],
            "medium": [200.0, 1000.0],
            "large": [1000.0, 5000.0],
        })
        amount_probs = hr_cfg.get("amount_probabilities", {
            "small": 0.5,
            "medium": 0.35,
            "large": 0.15,
        })

        tier_names = sorted(amount_probs.keys())
        tier_weights = [amount_probs[t] for t in tier_names]

        use_lognormal = hr_cfg.get("use_lognormal", True)
        dist_config = self.params.get(
            "backgroundPatterns", {}
        ).get("amount_distributions", {})

        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Respect tx_budget if set
        budget = getattr(self, "tx_budget", None)
        tx_count = 0

        for hr_account in high_risk_accounts:
            if budget is not None and tx_count >= budget:
                return

            # Generate multiple normal transactions for this high-risk account
            num_txs = random_instance.randint(
                tx_per_account_range[0], tx_per_account_range[1]
            )

            for _ in range(num_txs):
                if budget is not None and tx_count >= budget:
                    return

                # Random timestamp across the time span
                random_day = random_instance.randint(0, total_days - 1)
                random_hour = random_instance.randint(8, 20)
                random_minute = random_instance.randint(0, 59)

                tx_time = start_date + datetime.timedelta(
                    days=random_day, hours=random_hour, minutes=random_minute
                )

                # Select counterparty
                counterparty = random_instance.choice(counterparties)
                if counterparty == hr_account:
                    continue

                # Determine amount
                if use_lognormal:
                    amount = sample_lognormal_scalar(
                        "payment",
                        config=dist_config.get("payment", {}))
                else:
                    tier = random_instance.choices(
                        tier_names, weights=tier_weights)[0]
                    tier_range = amount_ranges.get(tier, [20.0, 200.0])
                    amount = round(
                        random_instance.uniform(
                            tier_range[0], tier_range[1]), 2
                    )

                # Randomly decide direction (high-risk sends or receives)
                if random_instance.random() < 0.5:
                    src_id = hr_account
                    dest_id = counterparty
                else:
                    src_id = counterparty
                    dest_id = hr_account

                # Mix of transaction types
                tx_type_rand = random_instance.random()
                if tx_type_rand < 0.5:
                    tx_type = TransactionType.PAYMENT
                elif tx_type_rand < 0.8:
                    tx_type = TransactionType.TRANSFER
                else:
                    tx_type = TransactionType.PAYMENT  # More payments than transfers

                tx_attrs = pattern_injector._create_transaction_edge(
                    src_id=src_id,
                    dest_id=dest_id,
                    timestamp=tx_time,
                    amount=amount,
                    transaction_type=tx_type,
                    is_fraudulent=False,
                )
                tx_count += 1
                yield (src_id, dest_id, tx_attrs)

    def generate_transaction_sequences(
        self, entity_selection: EntitySelection
    ) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        high_risk_accounts = entity_selection.central_entities

        if not high_risk_accounts:
            return sequences

        transaction_generator = self._generate_transactions_stream(
            entity_selection)

        sequences.append(
            TransactionSequence(
                transactions=transaction_generator,
                sequence_name="legitimate_high_risk_activity",
                start_time=self.time_span.get("start_date"),
                duration=self.time_span.get(
                    "end_date") - self.time_span.get("start_date"),
            )
        )
        return sequences


class LegitimateHighRiskActivityPattern(CompositePattern):
    """Pattern for legitimate activity by high-risk entities."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = LegitimateHighRiskStructural(
            graph_generator, params)
        temporal_component = LegitimateHighRiskTemporal(
            graph_generator, params)
        super().__init__(
            structural_component, temporal_component, graph_generator, params
        )

    @property
    def pattern_name(self) -> str:
        return "LegitimateHighRiskActivity"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
