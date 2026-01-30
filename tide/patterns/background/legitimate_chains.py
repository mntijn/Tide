"""
Legitimate multi-hop transaction chains.

This pattern injects legitimate transaction chains where money flows through
multiple accounts. Without this, chain/path structures only appear in fraud
patterns (layering, U-turns), making chains a fraud signal (data leakage).

Realistic chain scenarios modeled:
1. Group expenses: Friends split dinner/trip costs → organizer → vendor
   - Amount: $50-$500, Delay: minutes-hours, Length: 2-3
2. Family support: Parent → Child → Grandchild (gifts, education)
   - Amount: $500-$5,000, Delay: same day, Length: 2-3
3. Event collection: Multiple contributors → organizer → vendor
   - Amount: $100-$1,000 per person, Delay: days, Length: 3-4
4. Business supply: Buyer → Distributor → Manufacturer
   - Amount: $1K-$15K, Delay: 1-7 days, Length: 3-5
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


# Chain scenario configurations for realism
CHAIN_SCENARIOS = {
    "group_expense": {
        "weight": 0.35,
        "amount_range": [50.0, 500.0],
        "chain_length": [2, 3],
        "hop_delay_hours": [0.1, 4],  # Minutes to hours
        "tx_type": TransactionType.PAYMENT,
    },
    "family_support": {
        "weight": 0.30,
        "amount_range": [500.0, 5000.0],
        "chain_length": [2, 3],
        "hop_delay_hours": [0.5, 24],  # Hours to same day
        "tx_type": TransactionType.TRANSFER,
    },
    "event_collection": {
        "weight": 0.20,
        "amount_range": [200.0, 2000.0],
        "chain_length": [3, 4],
        "hop_delay_hours": [12, 72],  # Half day to 3 days
        "tx_type": TransactionType.TRANSFER,
    },
    "business_supply": {
        "weight": 0.15,
        "amount_range": [1000.0, 15000.0],
        "chain_length": [3, 5],
        "hop_delay_hours": [24, 168],  # 1-7 days
        "tx_type": TransactionType.PAYMENT,
    },
}


class LegitimateChainStructural(StructuralComponent):
    """Selects legitimate accounts to form transaction chains."""

    @property
    def num_required_entities(self) -> int:
        return 3  # Minimum chain length

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        # Get legitimate accounts + some fraud accounts (for topological mixing)
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

        # Mix in 50% of fraud accounts so they participate in normal chains
        if fraud_accounts:
            selected_fraud = random_instance.sample(
                fraud_accounts, min(len(fraud_accounts), int(len(fraud_accounts) * 0.9)))
            legit_accounts.extend(selected_fraud)

        legit_accounts = deduplicate_preserving_order(legit_accounts)

        if len(legit_accounts) < 3:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Scale number of chains with account count (roughly 1 chain per 100 accounts)
        chain_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateChains", {}
        )
        base_chains = chain_cfg.get("num_chains_per_generation", 50)
        # Scale: more accounts = more chains, but with diminishing returns
        num_chains = max(base_chains, len(legit_accounts) // 100)

        return EntitySelection(
            central_entities=legit_accounts,
            peripheral_entities=legit_accounts
        )


class LegitimateChainTemporal(TemporalComponent):
    """
    Generates legitimate multi-hop transaction chains using realistic scenarios.

    Creates path structures (A -> B -> C -> D) that are topologically similar
    to fraud layering, but are actually legitimate fund flows.
    """

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        all_accounts = entity_selection.peripheral_entities

        if len(all_accounts) < 3:
            return

        start_date = self.time_span["start_date"]
        end_date = self.time_span["end_date"]
        total_days = max(1, (end_date - start_date).days)

        # Config
        chain_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateChains", {}
        )
        base_chains = chain_cfg.get("num_chains_per_generation", 50)
        num_chains = max(base_chains, len(all_accounts) // 100)
        amount_decay = chain_cfg.get("amount_decay_range", [0.95, 1.0])
        use_lognormal = chain_cfg.get("use_lognormal", True)
        dist_config = self.params.get(
            "backgroundPatterns", {}
        ).get("amount_distributions", {})

        # Build scenario weights for random selection
        scenario_names = sorted(CHAIN_SCENARIOS.keys())
        scenario_weights = [CHAIN_SCENARIOS[s]["weight"]
                            for s in scenario_names]

        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Respect tx_budget if set
        budget = getattr(self, "tx_budget", None)
        tx_count = 0

        for _ in range(num_chains):
            if budget is not None and tx_count >= budget:
                return

            # Select a random scenario
            scenario_name = random_instance.choices(
                scenario_names, weights=scenario_weights
            )[0]
            scenario = CHAIN_SCENARIOS[scenario_name]

            # Get scenario parameters
            amount_range = scenario["amount_range"]
            chain_length_range = scenario["chain_length"]
            hop_delay_range = scenario["hop_delay_hours"]
            tx_type = scenario["tx_type"]

            # Create a chain of scenario-appropriate length
            chain_length = random_instance.randint(
                chain_length_range[0], chain_length_range[1]
            )

            # Select accounts for this chain
            chain = random_instance.sample(
                all_accounts, min(chain_length, len(all_accounts))
            )

            if len(chain) < 2:
                continue

            # Initial amount from scenario range or log-normal
            if use_lognormal:
                # Map scenario to distribution key
                dist_key = "transfer" if tx_type == TransactionType.TRANSFER else "payment"
                if scenario_name == "business_supply":
                    dist_key = "high_value"
                initial_amount = sample_lognormal_scalar(
                    dist_key, config=dist_config.get(dist_key, {}))
            else:
                initial_amount = random_instance.uniform(
                    amount_range[0], amount_range[1]
                )
            current_amount = initial_amount

            # Random start time
            chain_start_day = random_instance.randint(
                0, max(0, total_days - 10))
            chain_start_time = start_date + datetime.timedelta(
                days=chain_start_day,
                hours=random_instance.randint(8, 21),
                minutes=random_instance.randint(0, 59)
            )
            current_time = chain_start_time

            # Generate transactions along the chain
            for i in range(len(chain) - 1):
                if budget is not None and tx_count >= budget:
                    return

                src = chain[i]
                dest = chain[i + 1]

                # Amount may decay slightly at each hop (fees, splits)
                # But for group expenses, amount might stay same or round
                decay_factor = random_instance.uniform(
                    amount_decay[0], amount_decay[1]
                )
                current_amount *= decay_factor
                tx_amount = round(current_amount, 2)

                tx_attrs = pattern_injector._create_transaction_edge(
                    src_id=src,
                    dest_id=dest,
                    timestamp=current_time,
                    amount=tx_amount,
                    transaction_type=tx_type,
                    is_fraudulent=False,
                )
                tx_count += 1
                yield (src, dest, tx_attrs)

                # Scenario-appropriate delay before next hop
                delay_hours = random_instance.uniform(
                    hop_delay_range[0], hop_delay_range[1]
                )
                current_time += datetime.timedelta(hours=delay_hours)

    def generate_transaction_sequences(
        self, entity_selection: EntitySelection
    ) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        chain_accounts = entity_selection.central_entities

        if len(chain_accounts) < 3:
            return sequences

        transaction_generator = self._generate_transactions_stream(
            entity_selection)

        sequences.append(
            TransactionSequence(
                transactions=transaction_generator,
                sequence_name="legitimate_chains",
                start_time=self.time_span.get("start_date"),
                duration=self.time_span.get("end_date")
                - self.time_span.get("start_date"),
            )
        )
        return sequences


class LegitimateChainsPattern(CompositePattern):
    """Pattern for legitimate multi-hop transaction chains."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = LegitimateChainStructural(
            graph_generator, params)
        temporal_component = LegitimateChainTemporal(graph_generator, params)
        super().__init__(
            structural_component, temporal_component, graph_generator, params
        )

    @property
    def pattern_name(self) -> str:
        return "LegitimateChains"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
