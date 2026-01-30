"""
Legitimate rapid fund flow pattern.

This pattern injects legitimate rapid fund movements that mimic fraud patterns
but represent real-world scenarios. Without this, rapid inflow-outflow sequences
are almost exclusively fraudulent, making temporal patterns a trivial fraud signal.

Realistic scenarios modeled:
1. Account consolidation: Closing old accounts, moving funds to new bank
2. Emergency expenses: Medical bills, urgent repairs, travel emergencies
3. Large purchase funding: Car, appliances, home improvements
4. Event funding: Wedding, graduation, family events with collection and disbursement
5. Business cash flow: Receiving payment and immediately paying suppliers
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


# Scenario configurations for realistic rapid fund flows
RAPID_FLOW_SCENARIOS = {
    "account_consolidation": {
        "weight": 0.25,
        "inflow_amount_range": [2000.0, 15000.0],
        "outflow_amount_range": [1000.0, 10000.0],
        "inflows_per_event": [3, 8],  # Multiple account closures
        "outflows_per_event": [1, 3],  # Consolidate to fewer accounts
        "inflow_to_outflow_delay_hours": [1, 24],  # Quick consolidation
        "tx_types": {"inflow": TransactionType.TRANSFER, "outflow": TransactionType.TRANSFER},
    },
    "emergency_expense": {
        "weight": 0.20,
        "inflow_amount_range": [1000.0, 8000.0],
        "outflow_amount_range": [500.0, 5000.0],
        "inflows_per_event": [2, 5],  # Gathering funds from savings/family
        "outflows_per_event": [3, 8],  # Multiple payments for emergency
        "inflow_to_outflow_delay_hours": [2, 48],
        "tx_types": {"inflow": TransactionType.TRANSFER, "outflow": TransactionType.PAYMENT},
    },
    "large_purchase": {
        "weight": 0.25,
        "inflow_amount_range": [3000.0, 12000.0],
        "outflow_amount_range": [5000.0, 15000.0],
        "inflows_per_event": [2, 4],  # Gathering funds
        "outflows_per_event": [1, 2],  # One big purchase
        "inflow_to_outflow_delay_hours": [4, 72],
        "tx_types": {"inflow": TransactionType.TRANSFER, "outflow": TransactionType.PAYMENT},
    },
    "event_funding": {
        "weight": 0.15,
        # Contributions from many people
        "inflow_amount_range": [500.0, 3000.0],
        "outflow_amount_range": [1000.0, 8000.0],
        "inflows_per_event": [5, 15],  # Many contributors
        "outflows_per_event": [3, 6],  # Vendor payments
        "inflow_to_outflow_delay_hours": [24, 168],  # Days to weeks
        "tx_types": {"inflow": TransactionType.TRANSFER, "outflow": TransactionType.PAYMENT},
    },
    "business_cash_flow": {
        "weight": 0.15,
        "inflow_amount_range": [2000.0, 10000.0],
        "outflow_amount_range": [1500.0, 8000.0],
        "inflows_per_event": [1, 3],  # Client payments
        "outflows_per_event": [2, 5],  # Supplier payments
        "inflow_to_outflow_delay_hours": [1, 48],  # Quick turnaround
        "tx_types": {"inflow": TransactionType.PAYMENT, "outflow": TransactionType.PAYMENT},
    },
}


class LegitimateRapidFlowStructural(StructuralComponent):
    """Selects accounts that will have rapid fund flow events."""

    @property
    def num_required_entities(self) -> int:
        return 3  # Central account + inflow sources + outflow destinations

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

        # Mix in fraud accounts so they also have legitimate rapid flows
        if fraud_accounts:
            selected_fraud = random_instance.sample(
                fraud_accounts, min(len(fraud_accounts),
                                    int(len(fraud_accounts) * 0.7))
            )
            legit_accounts.extend(selected_fraud)

        legit_accounts = deduplicate_preserving_order(legit_accounts)

        if len(legit_accounts) < 5:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Config-driven event rate
        rapid_flow_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateRapidFlow", {}
        )
        events_per_1000_users = rapid_flow_cfg.get("events_per_1000_users", 20)

        # Calculate number of users who will have rapid flow events
        num_events = max(1, int(len(legit_accounts) *
                         events_per_1000_users / 1000))

        # Select central accounts (accounts that will be the hub of rapid flow)
        central_accounts = random_instance.sample(
            legit_accounts, min(num_events, len(legit_accounts))
        )

        # Remaining accounts serve as counterparties
        counterparties = [
            acc for acc in legit_accounts if acc not in central_accounts]

        return EntitySelection(
            central_entities=central_accounts,
            peripheral_entities=counterparties
        )


class LegitimateRapidFlowTemporal(TemporalComponent):
    """
    Generates legitimate rapid inflow-outflow sequences.

    This creates the same temporal patterns (rapid succession of transfers)
    that fraud patterns use, so that temporal clustering is no longer
    a unique fraud signal.
    """

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        central_accounts = entity_selection.central_entities
        counterparties = entity_selection.peripheral_entities

        if not central_accounts or not counterparties:
            return

        start_date = self.time_span["start_date"]
        end_date = self.time_span["end_date"]
        total_days = max(1, (end_date - start_date).days)

        # Config
        rapid_flow_cfg = self.params.get("backgroundPatterns", {}).get(
            "legitimateRapidFlow", {}
        )

        use_lognormal = rapid_flow_cfg.get("use_lognormal", True)
        dist_config = self.params.get(
            "backgroundPatterns", {}
        ).get("amount_distributions", {})

        # Override ranges from config if provided
        config_amount_range = rapid_flow_cfg.get("amount_range", None)
        config_inflows = rapid_flow_cfg.get("inflows_per_event", None)
        config_outflows = rapid_flow_cfg.get("outflows_per_event", None)
        config_delay = rapid_flow_cfg.get(
            "inflow_to_outflow_delay_hours", None)

        # Build scenario weights for random selection
        scenario_names = sorted(RAPID_FLOW_SCENARIOS.keys())
        scenario_weights = [RAPID_FLOW_SCENARIOS[s]["weight"]
                            for s in scenario_names]

        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Respect tx_budget if set
        budget = getattr(self, "tx_budget", None)
        tx_count = 0

        for central_account in central_accounts:
            if budget is not None and tx_count >= budget:
                return

            # Select a random scenario for this event
            scenario_name = random_instance.choices(
                scenario_names, weights=scenario_weights
            )[0]
            scenario = RAPID_FLOW_SCENARIOS[scenario_name]

            # Get scenario parameters (config overrides if provided)
            if config_amount_range:
                inflow_amount_range = config_amount_range
                outflow_amount_range = config_amount_range
            else:
                inflow_amount_range = scenario["inflow_amount_range"]
                outflow_amount_range = scenario["outflow_amount_range"]

            if config_inflows:
                inflows_range = config_inflows
            else:
                inflows_range = scenario["inflows_per_event"]

            if config_outflows:
                outflows_range = config_outflows
            else:
                outflows_range = scenario["outflows_per_event"]

            if config_delay:
                delay_range = config_delay
            else:
                delay_range = scenario["inflow_to_outflow_delay_hours"]

            tx_types = scenario["tx_types"]

            # Determine event timing
            # Leave enough room for the full event sequence
            max_event_duration_days = 10
            event_start_day = random_instance.randint(
                0, max(0, total_days - max_event_duration_days - 1)
            )
            event_start_time = start_date + datetime.timedelta(
                days=event_start_day,
                hours=random_instance.randint(8, 18),
                minutes=random_instance.randint(0, 59)
            )

            # Generate inflows
            num_inflows = random_instance.randint(
                inflows_range[0], inflows_range[1])
            inflow_times = []
            current_time = event_start_time

            for i in range(num_inflows):
                if budget is not None and tx_count >= budget:
                    return

                # Select random source account
                source_account = random_instance.choice(counterparties)
                if source_account == central_account:
                    continue

                # Inflows arrive over a short period (hours to a day)
                time_offset_hours = random_instance.uniform(0, 24)
                tx_time = current_time + \
                    datetime.timedelta(hours=time_offset_hours)
                inflow_times.append(tx_time)

                # Generate amount with INTERLEAVING
                # Some amounts use fraud-like structuring range
                structuring_probability = rapid_flow_cfg.get(
                    "structuring_amount_probability", 0.15)
                structuring_range = rapid_flow_cfg.get(
                    "structuring_amount_range", [7000.0, 9900.0])

                if random_instance.random() < structuring_probability:
                    amount = random_instance.uniform(
                        structuring_range[0], structuring_range[1])
                    amount = round(amount / 100) * 100
                elif use_lognormal:
                    amount = sample_lognormal_scalar(
                        "transfer", config=dist_config.get("transfer", {}))
                else:
                    amount = random_instance.uniform(
                        inflow_amount_range[0], inflow_amount_range[1]
                    )
                amount = round(amount, 2)

                tx_attrs = pattern_injector._create_transaction_edge(
                    src_id=source_account,
                    dest_id=central_account,
                    timestamp=tx_time,
                    amount=amount,
                    transaction_type=tx_types["inflow"],
                    is_fraudulent=False,
                )
                tx_count += 1
                yield (source_account, central_account, tx_attrs)

            # Generate outflows after delay
            if not inflow_times:
                continue

            last_inflow_time = max(inflow_times)
            delay_hours = random_instance.uniform(
                delay_range[0], delay_range[1])
            outflow_start_time = last_inflow_time + \
                datetime.timedelta(hours=delay_hours)

            num_outflows = random_instance.randint(
                outflows_range[0], outflows_range[1])

            for i in range(num_outflows):
                if budget is not None and tx_count >= budget:
                    return

                # Select random destination account
                dest_account = random_instance.choice(counterparties)
                if dest_account == central_account:
                    continue

                # Outflows happen quickly (within hours)
                time_offset_hours = random_instance.uniform(0, 12)
                tx_time = outflow_start_time + \
                    datetime.timedelta(hours=time_offset_hours)

                # Generate amount
                if use_lognormal:
                    amount = sample_lognormal_scalar(
                        "transfer", config=dist_config.get("transfer", {}))
                else:
                    amount = random_instance.uniform(
                        outflow_amount_range[0], outflow_amount_range[1]
                    )
                amount = round(amount, 2)

                tx_attrs = pattern_injector._create_transaction_edge(
                    src_id=central_account,
                    dest_id=dest_account,
                    timestamp=tx_time,
                    amount=amount,
                    transaction_type=tx_types["outflow"],
                    is_fraudulent=False,
                )
                tx_count += 1
                yield (central_account, dest_account, tx_attrs)

    def generate_transaction_sequences(
        self, entity_selection: EntitySelection
    ) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        central_accounts = entity_selection.central_entities

        if not central_accounts:
            return sequences

        transaction_generator = self._generate_transactions_stream(
            entity_selection)

        sequences.append(
            TransactionSequence(
                transactions=transaction_generator,
                sequence_name="legitimate_rapid_flow",
                start_time=self.time_span.get("start_date"),
                duration=self.time_span.get(
                    "end_date") - self.time_span.get("start_date"),
            )
        )
        return sequences


class LegitimateRapidFlowPattern(CompositePattern):
    """Pattern for legitimate rapid fund movements."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = LegitimateRapidFlowStructural(
            graph_generator, params)
        temporal_component = LegitimateRapidFlowTemporal(
            graph_generator, params)
        super().__init__(
            structural_component, temporal_component, graph_generator, params
        )

    @property
    def pattern_name(self) -> str:
        return "LegitimateRapidFlow"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
