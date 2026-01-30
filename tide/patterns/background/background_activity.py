import datetime
import numpy as np
from typing import List, Dict, Any, Tuple

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
from ...datastructures.attributes import TransactionAttributes
from ...utils.random_instance import random_instance, get_numpy_rng
from ...utils.amount_distributions import sample_lognormal, DEFAULT_DISTRIBUTIONS


class NonFraudulentRandomStructural(StructuralComponent):
    """
    Structural component: select entities for random daily transactions.

    This implementation introduces a hub-and-spoke structure for *legitimate* accounts
    to better mimic real-world payment networks (many accounts paying a few hubs).
    Fraudulent accounts can still participate as spenders, but hubs are always legit.
    """

    @property
    def num_required_entities(self) -> int:
        # Need at least 2 entities for transactions
        return 2

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        """
        Select spenders and hubs.

        - Spenders: all legitimate accounts + a large subset of fraud accounts
        - Hubs: top 5% of legitimate accounts (simulating merchants/utilities)
        """
        # Use pre-computed account clusters for O(1) lookup instead of O(n*m) traversal
        if hasattr(self.graph_generator, "account_clusters"):
            legit_accounts = list(
                self.graph_generator.account_clusters.get("legit", []))
            fraud_accounts = list(
                self.graph_generator.account_clusters.get("fraudulent", []))
        else:
            # Fallback to original method if clusters not available
            legit_entities = self.graph_generator.entity_clusters.get(
                "legit", [])
            fraud_entities = self.graph_generator.entity_clusters.get(
                "fraudulent", [])

            # Get accounts belonging to legitimate entities using the helper function
            legit_accounts: List[str] = []
            for entity_id in legit_entities:
                entity_accounts = self._get_owned_accounts(entity_id)
                legit_accounts.extend(entity_accounts)

            # Get accounts belonging to fraudulent entities
            fraud_accounts: List[str] = []
            for entity_id in fraud_entities:
                entity_accounts = self._get_owned_accounts(entity_id)
                fraud_accounts.extend(entity_accounts)

            # Remove duplicates while preserving order
            legit_accounts = deduplicate_preserving_order(legit_accounts)
            fraud_accounts = deduplicate_preserving_order(fraud_accounts)

        # Select most fraud accounts (80-90%), but not all, to keep broad coverage
        if fraud_accounts:
            selection_rate = random_instance.uniform(0.8, 0.9)
            num_fraud_to_select = max(
                1, int(len(fraud_accounts) * selection_rate))
            selected_fraud_accounts = random_instance.sample(
                fraud_accounts, min(num_fraud_to_select, len(fraud_accounts))
            )
        else:
            selected_fraud_accounts = []

        # Spenders: everyone (legit + selected fraud)
        spenders = list(legit_accounts) + list(selected_fraud_accounts)

        if not spenders:
            print("DEBUG [RandomPayments]: No accounts found (legit or fraud)!")
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Hubs: top 5% of legitimate accounts (if account_clusters already
        # encode degree, these tend to be higher-degree nodes; otherwise this
        # still creates a smaller, popular subset).
        if legit_accounts:
            num_hubs = max(1, int(len(legit_accounts) * 0.05))
            hubs = legit_accounts[:num_hubs]
        else:
            hubs = []

        return EntitySelection(central_entities=spenders, peripheral_entities=hubs)


class RandomPaymentsTemporal(TemporalComponent):
    """Temporal component: generate diverse random transactions between legitimate accounts."""

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        """
        A generator that yields individual transactions in chronological order.
        This approach avoids loading all transactions into memory at once.
        Respects self.tx_budget if set by the pattern manager.
        """
        spenders = entity_selection.central_entities
        hubs = entity_selection.peripheral_entities

        if len(spenders) < 2:
            return

        start_date: datetime.datetime = self.time_span["start_date"]
        end_date: datetime.datetime = self.time_span["end_date"]
        total_days = max(1, (end_date - start_date).days)

        tx_rate = self.params.get("transaction_rates", {}).get(
            "per_account_per_day", 0.2)
        random_payments_config = self.params.get("backgroundPatterns", {}).get(
            "randomPayments", {}
        )
        rate_based_count = int(tx_rate * total_days * len(spenders))

        # Respect tx_budget if set (capped by fraud ratio)
        budget = getattr(self, "tx_budget", None)
        if budget is not None and budget < rate_based_count:
            total_expected_txs = budget
        else:
            total_expected_txs = rate_based_count

        if total_expected_txs == 0:
            return

        print(
            f"RandomPayments: Generating {total_expected_txs:,} transactions for {len(spenders):,} accounts"
        )

        tx_type_probs = random_payments_config.get("transaction_type_probabilities", {
            "transfer": 0.4, "payment": 0.3, "deposit": 0.15, "withdrawal": 0.15
        })

        # Use string keys for np.choice and map to enums later for efficiency
        # IMPORTANT: Sort keys to ensure deterministic order across runs
        tx_type_keys = sorted(tx_type_probs.keys())
        tx_probs_values = [tx_type_probs[k] for k in tx_type_keys]
        type_to_enum = {
            "transfer": TransactionType.TRANSFER,
            "payment": TransactionType.PAYMENT,
            "deposit": TransactionType.DEPOSIT,
            "withdrawal": TransactionType.WITHDRAWAL
        }

        # OPTIMIZATION: Generate timestamps and sort them first to ensure chronological order.
        # NEW: Add temporal INTERLEAVING - some transactions clustered in bursts
        # This matches fraud patterns' "high_frequency" timestamp generation
        total_minutes = total_days * 24 * 60

        # Probability of a transaction being part of a burst cluster
        burst_probability = random_payments_config.get(
            "burst_probability", 0.15)
        burst_duration_minutes = random_payments_config.get(
            "burst_duration_minutes", 120)  # 2-hour burst windows

        # Generate base timestamps using dedicated numpy generator
        random_minutes_offsets = get_numpy_rng().integers(
            0, total_minutes, size=total_expected_txs)

        # Create burst clusters for some transactions
        is_burst = get_numpy_rng().random(total_expected_txs) < burst_probability
        num_bursts = max(1, int(np.sum(is_burst) / 10))  # ~10 txs per burst

        if num_bursts > 0 and np.sum(is_burst) > 0:
            # Select random burst start times
            burst_starts = get_numpy_rng().integers(0, total_minutes, size=num_bursts)
            # Assign burst transactions to clusters
            burst_indices = np.where(is_burst)[0]
            for i, idx in enumerate(burst_indices):
                burst_id = i % num_bursts
                # Cluster transactions within burst_duration_minutes of burst start
                offset_within_burst = get_numpy_rng().integers(
                    0, burst_duration_minutes)
                random_minutes_offsets[idx] = (
                    burst_starts[burst_id] + offset_within_burst) % total_minutes

        random_minutes_offsets.sort()  # Sort timestamps to generate transactions in order

        # Vectorized generation of other attributes
        selected_type_keys = get_numpy_rng().choice(
            tx_type_keys, size=total_expected_txs, p=tx_probs_values
        )

        spenders_array = np.array(spenders)
        src_accounts = get_numpy_rng().choice(
            spenders_array, size=total_expected_txs)

        # Preferential attachment logic: 80% of txs go to hubs (many-to-one),
        # 20% go to random peers (one-to-one). If no hubs exist, fall back to peers only.
        prob_hub = 0.8 if len(hubs) > 0 else 0.0
        is_hub_tx = get_numpy_rng().random(total_expected_txs) < prob_hub

        dest_accounts = np.empty(total_expected_txs, dtype=object)

        if len(hubs) > 0:
            hubs_array = np.array(hubs)
            dest_accounts[is_hub_tx] = get_numpy_rng().choice(
                hubs_array, size=np.sum(is_hub_tx)
            )

        # Peers: random spenders. This includes both legit and fraud accounts,
        # but the graph topology for legit activity is dominated by hub traffic.
        dest_accounts[~is_hub_tx] = get_numpy_rng().choice(
            spenders_array, size=np.sum(~is_hub_tx)
        )

        # Amount generation: log-normal distributions calibrated to
        # real-world data (Fed Payments Study, Nacha, Nilson Report)
        amounts = np.zeros(total_expected_txs)

        # Structuring interleaving (small fraction uses near-threshold amounts)
        structured_amount_probability = random_payments_config.get(
            "structured_amount_probability", 0.03)
        structuring_range = random_payments_config.get(
            "structuring_range", [7000.0, 9900.0])

        # Get per-type distribution overrides from config
        dist_config = self.params.get(
            "backgroundPatterns", {}
        ).get("amount_distributions", {})
        use_lognormal = random_payments_config.get(
            "use_lognormal", True)

        for tx_type_key in tx_type_keys:
            mask = selected_type_keys == tx_type_key
            count = np.sum(mask)
            if count == 0:
                continue

            tx_type_enum = type_to_enum[tx_type_key]

            if use_lognormal:
                # Map tx type to distribution key
                dist_key = tx_type_key
                if tx_type_enum in (
                    TransactionType.DEPOSIT,
                    TransactionType.WITHDRAWAL
                ):
                    dist_key = tx_type_key  # "deposit" or "withdrawal"
                cfg = dist_config.get(dist_key, {})
                base_amounts = sample_lognormal(
                    dist_key, size=count, config=cfg)
            else:
                # Legacy: uniform from ranges
                amount_ranges = random_payments_config.get(
                    "amount_ranges", {})
                if tx_type_enum == TransactionType.PAYMENT:
                    r = amount_ranges.get("payment", [10.0, 2000.0])
                elif tx_type_enum == TransactionType.TRANSFER:
                    r = amount_ranges.get("transfer", [5.0, 800.0])
                else:
                    r = amount_ranges.get(
                        "cash_operations", [20.0, 500.0])
                base_amounts = get_numpy_rng().uniform(
                    r[0], r[1], size=count)

            # Interleave: some amounts use structuring range
            mult = 1.5 if tx_type_enum in (
                TransactionType.DEPOSIT,
                TransactionType.WITHDRAWAL
            ) else 1.0
            structured_mask = get_numpy_rng().random(
                count) < (structured_amount_probability * mult)
            if np.sum(structured_mask) > 0:
                base_amounts[structured_mask] = get_numpy_rng().uniform(
                    structuring_range[0], structuring_range[1],
                    size=np.sum(structured_mask))

            # Cash ops round to nearest 10
            if tx_type_enum in (
                TransactionType.DEPOSIT,
                TransactionType.WITHDRAWAL
            ):
                amounts[mask] = np.round(base_amounts / 10) * 10
            else:
                amounts[mask] = np.round(base_amounts, 2)

        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Pre-compute timedeltas for batch efficiency
        # Converting int minutes to timedelta is slow in a loop; batch it
        timestamps = [
            start_date + datetime.timedelta(minutes=int(m))
            for m in random_minutes_offsets
        ]

        # Yield transactions one by one instead of storing them in a list
        for i in range(total_expected_txs):
            # Resolve sender/receiver accounts ensuring they are not the same
            src, dest = src_accounts[i], dest_accounts[i]
            if src == dest:
                # Simple and fast retry logic for the rare cases of collision
                while dest == src:
                    dest = get_numpy_rng().choice(spenders_array)

            tx_attrs = pattern_injector._create_transaction_edge(
                src_id=src,
                dest_id=dest,
                timestamp=timestamps[i],
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
