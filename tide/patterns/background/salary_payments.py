import datetime
import calendar
from typing import List, Dict, Any, Tuple, Optional

from ..base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector, deduplicate_preserving_order
)
from ...datastructures.enums import NodeType, TransactionType
from ...datastructures.attributes import TransactionAttributes
from ...utils.random_instance import random_instance
from ...utils.constants import HIGH_PAID_OCCUPATIONS
from ...utils.amount_distributions import sample_lognormal_scalar


class SalaryPaymentsStructural(StructuralComponent):
    """Structural component: select business entities as payers and individuals as recipients."""

    @property
    def num_required_entities(self) -> int:
        return 2  # Need at least 1 business and 1 individual

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        # Get legitimate entities
        legit_entities = list(
            self.graph_generator.entity_clusters.get("legit", []))

        # Also include a significant portion of fraudulent entities
        # Fraudsters (front businesses) pay salaries, and fraudsters (individuals) receive them
        # This topological mixing is crucial to prevent easy detection
        fraud_entities = list(
            self.graph_generator.entity_clusters.get("fraudulent", []))
        if fraud_entities:
            # Include 50% of fraud entities in the salary economy
            selected_fraud = random_instance.sample(
                fraud_entities, min(len(fraud_entities), int(len(fraud_entities) * 0.9)))
            legit_entities.extend(selected_fraud)

        # Dedup while preserving order for determinism
        all_candidate_entities = deduplicate_preserving_order(legit_entities)

        if not all_candidate_entities:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Separate businesses and individuals
        business_entities = []
        individual_entities = []

        for entity_id in all_candidate_entities:
            node_data = self.graph_generator.graph.nodes[entity_id]
            if node_data.get("node_type") == NodeType.BUSINESS:
                business_entities.append(entity_id)
            elif node_data.get("node_type") == NodeType.INDIVIDUAL:
                individual_entities.append(entity_id)

        # Need at least 1 business and 1 individual
        if not business_entities or not individual_entities:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Get accounts for selected entities using the helper function
        business_accounts = []
        individual_accounts = []

        for business_id in business_entities:
            business_account_ids = self._get_owned_accounts(business_id)
            for account_id in business_account_ids:
                business_accounts.append((account_id, business_id))

        for individual_id in individual_entities:
            individual_account_ids = self._get_owned_accounts(individual_id)
            for account_id in individual_account_ids:
                individual_accounts.append((account_id, individual_id))

        # Store both account and entity mappings for the temporal component
        # Central entities = business accounts, Peripheral = individual accounts
        return EntitySelection(
            # [(account_id, business_id), ...]
            central_entities=business_accounts,
            # [(account_id, individual_id), ...]
            peripheral_entities=individual_accounts
        )


class SalaryPaymentsTemporal(TemporalComponent):
    """Temporal component: generate regular salary payments on scheduled dates."""

    def _generate_transactions_stream(self, entity_selection: EntitySelection):
        business_accounts = entity_selection.central_entities
        individual_accounts = entity_selection.peripheral_entities

        if not business_accounts or not individual_accounts:
            return

        # Get salary payment configuration
        salary_config = self.params.get(
            "backgroundPatterns", {}).get("salaryPayments", {})

        payment_intervals = salary_config.get(
            "payment_intervals", [14, 30])  # bi-weekly or monthly
        salary_variation = salary_config.get("salary_variation", 0.05)  # ±5%

        # Distribution config
        use_lognormal = salary_config.get("use_lognormal", True)
        dist_config = self.params.get(
            "backgroundPatterns", {}
        ).get("amount_distributions", {}).get("salary", {})
        high_earner_dist = {
            "mu": salary_config.get("high_earner_mu", 8.5),
            "sigma": salary_config.get("high_earner_sigma", 0.5),
            "min": dist_config.get("min", 200.0),
            "max": dist_config.get("max", 30000.0),
        }
        # Legacy fallback ranges
        salary_range = salary_config.get("salary_range", [2500.0, 7500.0])
        high_earner_salary_range = salary_config.get(
            "high_earner_salary_range", [8000.0, 20000.0]
        )
        preferred_days = salary_config.get(
            "preferred_payment_days", [1, 15, 30])

        start_date: datetime.datetime = self.time_span["start_date"]
        end_date: datetime.datetime = self.time_span["end_date"]

        # Generate salary payments - businesses pay individuals regularly
        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Respect tx_budget if set
        budget = getattr(self, "tx_budget", None)
        tx_count = 0

        # Each business pays some random individuals
        available_individuals = individual_accounts.copy()
        random_instance.shuffle(available_individuals)

        for business_account_id, business_entity_id in business_accounts:
            if not available_individuals:
                break

            # Each business pays 1-3 individuals
            num_recipients = min(random_instance.randint(
                1, 3), len(available_individuals))

            for _ in range(num_recipients):
                if not available_individuals:
                    break

                individual_account_id, individual_entity_id = available_individuals.pop()

                # Check if the individual has a high-paying occupation
                individual_node = self.graph_generator.graph.nodes[individual_entity_id]
                occupation = individual_node.get("occupation")

                is_high_earner = (
                    occupation and occupation in HIGH_PAID_OCCUPATIONS
                )

                # Sample base salary from log-normal or uniform
                if use_lognormal:
                    cfg = high_earner_dist if is_high_earner else dist_config
                    base_salary = sample_lognormal_scalar("salary", config=cfg)
                else:
                    current_salary_range = (
                        high_earner_salary_range if is_high_earner
                        else salary_range
                    )
                    base_salary = random_instance.uniform(
                        current_salary_range[0], current_salary_range[1]
                    )
                payment_interval = random_instance.choice(payment_intervals)

                # Generate payment dates
                payment_dates = self._generate_payment_dates(
                    start_date, end_date, payment_interval, preferred_days)

                # Create salary payment transactions
                for payment_date in payment_dates:
                    if budget is not None and tx_count >= budget:
                        return

                    # Add small variation to salary amount
                    variation_factor = random_instance.uniform(
                        1 - salary_variation, 1 + salary_variation)
                    payment_amount = round(base_salary * variation_factor, 2)

                    tx_attrs = pattern_injector._create_transaction_edge(
                        src_id=business_account_id,
                        dest_id=individual_account_id,
                        timestamp=payment_date,
                        amount=float(payment_amount),
                        transaction_type=TransactionType.PAYMENT,
                        is_fraudulent=False,
                    )
                    tx_count += 1
                    yield (business_account_id, individual_account_id, tx_attrs)

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        """
        Returns a list containing a single TransactionSequence.
        The sequence's 'transactions' attribute is a generator, not a list.
        """
        sequences: List[TransactionSequence] = []
        business_accounts = entity_selection.central_entities
        individual_accounts = entity_selection.peripheral_entities

        if not business_accounts or not individual_accounts:
            return sequences

        # The transaction data is provided by a generator for memory efficiency
        transaction_generator = self._generate_transactions_stream(
            entity_selection)

        sequences.append(
            TransactionSequence(
                transactions=transaction_generator,
                sequence_name="salary_payments",
                start_time=self.time_span.get("start_date"),
                duration=self.time_span.get(
                    "end_date") - self.time_span.get("start_date"),
            )
        )
        return sequences

    def _generate_payment_dates(self, start_date: datetime.datetime,
                                end_date: datetime.datetime,
                                payment_interval: int,
                                preferred_days: List[int]) -> List[datetime.datetime]:
        """Generate regular payment dates based on interval and preferred days."""
        payment_dates = []
        current_date = start_date

        while current_date <= end_date:
            if payment_interval == 30:  # Monthly payments
                # Use preferred days of month
                for day in preferred_days:
                    try:
                        # Get the last day of the current month
                        last_day = calendar.monthrange(
                            current_date.year, current_date.month)[1]
                        # If preferred day is beyond month end, use last day
                        actual_day = min(day, last_day)

                        payment_date = datetime.datetime(
                            current_date.year, current_date.month, actual_day,
                            random_instance.randint(8, 17),  # Business hours
                            random_instance.randint(0, 59)   # Random minute
                        )

                        if start_date <= payment_date <= end_date:
                            payment_dates.append(payment_date)
                    except ValueError:
                        # Skip invalid dates
                        continue

                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(
                        year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(
                        month=current_date.month + 1)

            elif payment_interval == 14:  # Bi-weekly payments
                # Add business hours timing
                payment_date = current_date.replace(
                    hour=random_instance.randint(8, 17),
                    minute=random_instance.randint(0, 59),
                    second=0,
                    microsecond=0
                )

                if start_date <= payment_date <= end_date:
                    payment_dates.append(payment_date)

                # Move forward by 2 weeks
                current_date += datetime.timedelta(days=14)

            elif payment_interval == 7:  # Weekly payments
                payment_date = current_date.replace(
                    hour=random_instance.randint(8, 17),
                    minute=random_instance.randint(0, 59),
                    second=0,
                    microsecond=0
                )

                if start_date <= payment_date <= end_date:
                    payment_dates.append(payment_date)

                # Move forward by 1 week
                current_date += datetime.timedelta(days=7)
            else:
                # Custom interval in days
                payment_date = current_date.replace(
                    hour=random_instance.randint(8, 17),
                    minute=random_instance.randint(0, 59),
                    second=0,
                    microsecond=0
                )

                if start_date <= payment_date <= end_date:
                    payment_dates.append(payment_date)

                current_date += datetime.timedelta(days=payment_interval)

        return payment_dates


class SalaryPaymentsPattern(CompositePattern):
    """Salary payments pattern for regular business-to-individual payments."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = SalaryPaymentsStructural(
            graph_generator, params)
        temporal_component = SalaryPaymentsTemporal(graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "SalaryPayments"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
