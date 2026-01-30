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


class LegitimateBurstStructural(StructuralComponent):
    """Selects legitimate users who will have a busy day (shopping spree / bill run)."""

    @property
    def num_required_entities(self) -> int:
        # At least one user and one potential counterparty set
        return 2

    def _get_legit_accounts_flat(self) -> List[str]:
        """
        Helper: return a flat list of legitimate account ids, PLUS some fraud accounts.

        We include fraud accounts to ensure they participate in normal "bursty" behavior
        (shopping sprees etc.), preventing topological isolation.
        """
        if hasattr(self.graph_generator, "account_clusters"):
            legit_accounts = list(
                self.graph_generator.account_clusters.get("legit", []))
            fraud_accounts = list(
                self.graph_generator.account_clusters.get("fraudulent", []))
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

        # Mix in 90% of fraud accounts for broad coverage
        if fraud_accounts:
            selected_fraud = random_instance.sample(
                fraud_accounts, min(len(fraud_accounts), int(len(fraud_accounts) * 0.9)))
            legit_accounts.extend(selected_fraud)

        legit_accounts = deduplicate_preserving_order(legit_accounts)
        return legit_accounts

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        legit_accounts = self._get_legit_accounts_flat()

        if len(legit_accounts) < 2:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Config-driven burst fraction (increased default from 0.05 to 0.08)
        burst_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateBurst", {}
        )
        burst_fraction = burst_cfg.get("burst_user_fraction", 0.08)
        num_burst_users = max(1, int(len(legit_accounts) * burst_fraction))
        selected_burst_users = random_instance.sample(
            legit_accounts, min(num_burst_users, len(legit_accounts))
        )

        # Targets can be any legitimate account (friends/merchants/etc.).
        targets = legit_accounts

        return EntitySelection(
            central_entities=selected_burst_users, peripheral_entities=targets
        )


class LegitimateBurstTemporal(TemporalComponent):
    """
    Generates tight temporal clusters of transactions (legitimate bursts).

    This introduces non-fraudulent bursts so that temporal frequency alone is
    not a strong fraud signal; models must instead consider *who* is involved
    and the surrounding topology.

    Enhanced to support:
    - Higher burst intensity (5-15 transactions) to match fraud patterns
    - Configurable time between transactions (1-30 minutes)
    - Cash operations (deposits/withdrawals) in addition to payments
    - Multiple bursts per user for more realistic behavior
    """

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        burst_users = entity_selection.central_entities
        targets = entity_selection.peripheral_entities

        if not burst_users or not targets:
            return

        start_date = self.time_span["start_date"]
        end_date = self.time_span["end_date"]
        total_days = max(1, (end_date - start_date).days)

        # Config-driven burst size and amount (enhanced defaults)
        burst_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateBurst", {}
        )
        burst_size_range = burst_cfg.get(
            "burst_size_range", [5, 15])  # Increased from [3, 6]
        amount_range = burst_cfg.get(
            "amount_range", [50.0, 2000.0])  # Expanded upper bound
        time_between_txs_minutes = burst_cfg.get(
            "time_between_txs_minutes", [1, 30])
        # NEW: support deposits/withdrawals
        include_cash_ops = burst_cfg.get("include_cash_ops", True)
        bursts_per_user_range = burst_cfg.get(
            "bursts_per_user_range", [1, 3])  # Multiple bursts

        # Cash operations settings
        cash_ops_probability = burst_cfg.get(
            "cash_ops_probability", 0.15)  # 15% of bursts include cash
        cash_amount_range = burst_cfg.get("cash_amount_range", [100.0, 2000.0])

        # Log-normal distribution settings
        use_lognormal = burst_cfg.get("use_lognormal", True)
        dist_config = self.params.get(
            "backgroundPatterns", {}
        ).get("amount_distributions", {})

        # NEW: Fraud-like amount interleaving - some bursts use structuring amounts
        structuring_burst_probability = burst_cfg.get(
            "structuring_burst_probability", 0.12)  # 12% of bursts use fraud-like amounts
        structuring_amount_range = burst_cfg.get(
            "structuring_amount_range", [7000.0, 9900.0])

        pattern_injector = PatternInjector(self.graph_generator, self.params)
        cash_account_id = self.graph_generator.cash_account_id

        # Respect tx_budget if set
        budget = getattr(self, "tx_budget", None)
        tx_count = 0

        for user in burst_users:
            # Each user can have multiple burst events
            num_bursts = random_instance.randint(
                bursts_per_user_range[0], bursts_per_user_range[1]
            )

            for _ in range(num_bursts):
                if budget is not None and tx_count >= budget:
                    return

                # Pick a random day for the burst
                burst_day = random_instance.randint(0, total_days - 1)
                burst_start_time = start_date + datetime.timedelta(
                    days=burst_day, hours=random_instance.randint(8, 21)
                )

                # Generate burst transactions
                num_txs = random_instance.randint(
                    burst_size_range[0], burst_size_range[1]
                )

                # Determine if this burst includes cash operations
                is_cash_burst = include_cash_ops and random_instance.random() < cash_ops_probability

                # Track cumulative time offset for realistic sequencing
                cumulative_minutes = 0

                for i in range(num_txs):
                    if budget is not None and tx_count >= budget:
                        return

                    # Progressive time offsets for realistic burst timing
                    time_offset = random_instance.randint(
                        time_between_txs_minutes[0], time_between_txs_minutes[1]
                    )
                    cumulative_minutes += time_offset
                    tx_time = burst_start_time + \
                        datetime.timedelta(minutes=cumulative_minutes)

                    # Determine transaction type
                    if is_cash_burst and i < 2:  # First 1-2 transactions can be cash ops
                        # Cash operation (deposit or withdrawal)
                        is_deposit = random_instance.random() < 0.4
                        if use_lognormal:
                            dist_key = "deposit" if is_deposit else "withdrawal"
                            amount = sample_lognormal_scalar(
                                dist_key, config=dist_config.get(dist_key, {}))
                        else:
                            amount = round(
                                random_instance.uniform(
                                    cash_amount_range[0], cash_amount_range[1]), 2
                            )
                        # Round to realistic cash denominations
                        amount = round(amount / 20) * 20
                        amount = max(20.0, amount)

                        if is_deposit and cash_account_id:
                            tx_attrs = pattern_injector._create_transaction_edge(
                                src_id=cash_account_id,
                                dest_id=user,
                                timestamp=tx_time,
                                amount=amount,
                                transaction_type=TransactionType.DEPOSIT,
                                is_fraudulent=False,
                            )
                            tx_count += 1
                            yield (cash_account_id, user, tx_attrs)
                        elif cash_account_id:
                            tx_attrs = pattern_injector._create_transaction_edge(
                                src_id=user,
                                dest_id=cash_account_id,
                                timestamp=tx_time,
                                amount=amount,
                                transaction_type=TransactionType.WITHDRAWAL,
                                is_fraudulent=False,
                            )
                            tx_count += 1
                            yield (user, cash_account_id, tx_attrs)
                    else:
                        # Regular payment/transfer
                        target = random_instance.choice(targets)
                        if target == user:
                            continue

                        # INTERLEAVING: Some transactions use fraud-like structuring amounts
                        if random_instance.random() < structuring_burst_probability:
                            amount = round(
                                random_instance.uniform(
                                    structuring_amount_range[0],
                                    structuring_amount_range[1]), 2
                            )
                            # Round to realistic denominations like fraud patterns
                            amount = round(amount / 100) * 100
                        else:
                            if use_lognormal:
                                amount = sample_lognormal_scalar(
                                    "payment", config=dist_config.get("payment", {}))
                            else:
                                amount = round(
                                    random_instance.uniform(
                                        amount_range[0], amount_range[1]), 2
                                )

                        # Mix of payments and transfers
                        tx_type = TransactionType.PAYMENT if random_instance.random(
                        ) < 0.7 else TransactionType.TRANSFER

                        tx_attrs = pattern_injector._create_transaction_edge(
                            src_id=user,
                            dest_id=target,
                            timestamp=tx_time,
                            amount=amount,
                            transaction_type=tx_type,
                            is_fraudulent=False,
                        )
                        tx_count += 1
                        yield (user, target, tx_attrs)

    def generate_transaction_sequences(
        self, entity_selection: EntitySelection
    ) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        burst_users = entity_selection.central_entities
        targets = entity_selection.peripheral_entities

        if not burst_users or not targets:
            return sequences

        transaction_generator = self._generate_transactions_stream(
            entity_selection)

        sequences.append(
            TransactionSequence(
                transactions=transaction_generator,
                sequence_name="legitimate_burst_activity",
                start_time=self.time_span.get("start_date"),
                duration=self.time_span.get("end_date")
                - self.time_span.get("start_date"),
            )
        )
        return sequences


class LegitimateBurstPattern(CompositePattern):
    """Pattern for legitimate bursty activity (shopping sprees, bill runs, etc.)."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = LegitimateBurstStructural(
            graph_generator, params)
        temporal_component = LegitimateBurstTemporal(graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "LegitimateBurst"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
