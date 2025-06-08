import datetime
from typing import List, Dict, Any, Tuple

from .base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ..datastructures.enums import NodeType, TransactionType, EdgeType
from ..datastructures.attributes import TransactionAttributes
from ..utils.random_instance import random_instance
from ..utils.constants import HIGH_PAID_OCCUPATIONS


class AllBusinessesAndIndividualsStructural(StructuralComponent):
    """Structural component: selects all businesses and individuals."""

    @property
    def num_required_entities(self) -> int:
        return 0

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        return EntitySelection(central_entities=[], peripheral_entities=[])


class SalaryPaymentsTemporal(TemporalComponent):
    """Temporal component: generates salary payments on specific days of the month."""

    def _get_business_owner(self, business_id: str) -> str | None:
        """Find the owner of a business."""
        for pred_id in self.graph.predecessors(business_id):
            edge_data = self.graph.get_edge_data(pred_id, business_id)
            if edge_data and edge_data.get("edge_type") == EdgeType.OWNERSHIP:
                return pred_id
        return None

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        transactions: List[Tuple[str, str, TransactionAttributes]] = []

        bg_activity_config = self.params.get("background_activity", {})
        salaries_config = bg_activity_config.get(
            "patterns", {}).get("salaries", {})

        if not salaries_config.get("enabled", False):
            return sequences

        payment_days = salaries_config.get("payment_days", [1, 28])
        salary_range = self.params.get("salary_amount_range", [2500.0, 7500.0])
        high_earner_multiplier = 2.0

        all_businesses = self.graph_generator.all_nodes.get(
            NodeType.BUSINESS, [])
        all_individuals = self.graph_generator.all_nodes.get(
            NodeType.INDIVIDUAL, [])

        if not all_businesses or not all_individuals:
            return sequences

        potential_employees = all_individuals.copy()
        pattern_injector = PatternInjector(self.graph_generator, self.params)

        current_date = self.time_span["start_date"]
        while current_date <= self.time_span["end_date"]:
            if current_date.day in payment_days:
                for business_id in all_businesses:
                    business_node = self.graph.nodes[business_id]
                    num_employees = business_node.get("number_of_employees", 0)

                    if num_employees == 0:
                        continue

                    owner_id = self._get_business_owner(business_id)
                    employee_pool = [
                        p for p in potential_employees if p != owner_id]

                    if not employee_pool:
                        continue

                    num_employees = min(num_employees, len(employee_pool))
                    selected_employees = random_instance.sample(
                        employee_pool, num_employees)
                    business_accounts = self._get_owned_accounts(business_id)

                    if not business_accounts:
                        continue

                    business_account = random_instance.choice(
                        business_accounts)

                    for employee_id in selected_employees:
                        employee_accounts = self._get_owned_accounts(
                            employee_id)
                        if not employee_accounts:
                            continue

                        employee_account = random_instance.choice(
                            employee_accounts)
                        employee_node = self.graph.nodes[employee_id]

                        base_salary = random_instance.uniform(
                            salary_range[0], salary_range[1])

                        occupation = employee_node.get("occupation", "")
                        if occupation in HIGH_PAID_OCCUPATIONS:
                            base_salary *= high_earner_multiplier

                        payment_timestamp = current_date + datetime.timedelta(
                            hours=random_instance.randint(9, 17),
                            minutes=random_instance.randint(0, 59)
                        )

                        tx_attrs = pattern_injector._create_transaction_edge(
                            src_id=business_account,
                            dest_id=employee_account,
                            timestamp=payment_timestamp,
                            amount=round(base_salary, 2),
                            transaction_type=TransactionType.SALARY,
                            is_fraudulent=False,
                        )
                        transactions.append(
                            (business_account, employee_account, tx_attrs))

            current_date += datetime.timedelta(days=1)

        if transactions:
            sequences.append(TransactionSequence(
                transactions=transactions,
                sequence_name="salary_payments",
                start_time=self.time_span["start_date"],
                duration=self.time_span["end_date"] -
                self.time_span["start_date"],
            ))

        return sequences


class SalaryPaymentsPattern(CompositePattern):
    """Composite pattern for salary payments from businesses to individuals."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = AllBusinessesAndIndividualsStructural(
            graph_generator, params)
        temporal_component = SalaryPaymentsTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "SalaryPayments"

    @property
    def num_required_entities(self) -> int:
        return 0
