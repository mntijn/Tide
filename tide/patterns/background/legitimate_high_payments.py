import datetime
from typing import List, Dict, Any, Tuple

from ..base import PatternInjector
from ...datastructures.enums import NodeType, TransactionType
from ...datastructures.attributes import TransactionAttributes
from ...utils.random_instance import random_instance


class LegitimateHighPaymentsPattern:
    """Simple pattern to generate legitimate high-value transactions for obfuscation."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        self.graph_generator = graph_generator
        self.params = params
        self.time_span = params.get("time_span", {})

    @property
    def pattern_name(self) -> str:
        return "LegitimateHighPayments"

    def inject_pattern_generator(self, available_entities: List[str]):
        """Generate legitimate high-value transactions between wealthy individuals and large businesses."""

        # Get high-value transactors from existing clusters
        high_paid_individuals = self.graph_generator.entity_clusters.get(
            "high_paid_occupations", [])
        legit_entities = self.graph_generator.entity_clusters.get("legit", [])

        # Find large businesses
        large_businesses = []
        for entity_id in legit_entities:
            node_data = self.graph_generator.graph.nodes[entity_id]
            if (node_data.get("node_type") == NodeType.BUSINESS and
                    node_data.get("number_of_employees", 0) > 10):
                large_businesses.append(entity_id)

        all_high_value_accounts = []
        for entity_id in high_paid_individuals + large_businesses:
            for neighbor in self.graph_generator.graph.neighbors(entity_id):
                neighbor_data = self.graph_generator.graph.nodes[neighbor]
                if neighbor_data.get("node_type") == NodeType.ACCOUNT:
                    all_high_value_accounts.append((neighbor, entity_id))

        if len(all_high_value_accounts) < 2:
            return

        # Get configuration
        high_payments_config = self.params.get(
            "backgroundPatterns", {}).get("legitimateHighPayments", {})

        amount_ranges = high_payments_config.get("high_payment_ranges", {
            "property_transactions": [50000.0, 500000.0],
            "business_deals": [10000.0, 100000.0],
            "luxury_purchases": [5000.0, 50000.0]
        })

        type_probs = high_payments_config.get("high_payment_type_probabilities", {
            "property": 0.2, "business": 0.6, "luxury": 0.2
        })

        monthly_rate = high_payments_config.get(
            "high_payment_rate_per_month", 0.1)

        # Calculate total transactions
        start_date = self.time_span["start_date"]
        end_date = self.time_span["end_date"]
        total_months = max(1, ((end_date.year - start_date.year)
                           * 12 + (end_date.month - start_date.month)))
        total_expected_txs = int(
            monthly_rate * total_months * len(all_high_value_accounts))

        if total_expected_txs == 0:
            return

        # Generate transactions efficiently
        pattern_injector = PatternInjector(self.graph_generator, self.params)
        total_seconds = max(1, (end_date - start_date).total_seconds())

        for i in range(total_expected_txs):
            # Random business hours timestamp
            random_seconds = random_instance.randint(0, int(total_seconds - 1))
            base_timestamp = start_date + \
                datetime.timedelta(seconds=random_seconds)
            business_hour = random_instance.randint(9, 17)
            business_minute = random_instance.randint(0, 59)
            timestamp = base_timestamp.replace(
                hour=business_hour, minute=business_minute)

            # Select different entities for src/dest
            src_account_id, src_entity_id = random_instance.choice(
                all_high_value_accounts)
            valid_dest_accounts = [(acc_id, ent_id) for acc_id, ent_id in all_high_value_accounts
                                   if ent_id != src_entity_id]

            if not valid_dest_accounts:
                continue

            dest_account_id, dest_entity_id = random_instance.choice(
                valid_dest_accounts)

            # Determine transaction type and amount
            rand_val = random_instance.random()
            if rand_val <= type_probs["property"]:
                tx_type = TransactionType.TRANSFER
                amount_range = amount_ranges["property_transactions"]
            elif rand_val <= type_probs["property"] + type_probs["business"]:
                tx_type = TransactionType.PAYMENT
                amount_range = amount_ranges["business_deals"]
            else:
                tx_type = TransactionType.PAYMENT
                amount_range = amount_ranges["luxury_purchases"]

            # Generate realistic amount (rounded to nearest 100)
            amount = random_instance.uniform(amount_range[0], amount_range[1])
            amount = round(amount / 100) * 100

            tx_attrs = pattern_injector._create_transaction_edge(
                src_id=src_account_id,
                dest_id=dest_account_id,
                timestamp=timestamp,
                amount=float(amount),
                transaction_type=tx_type,
                is_fraudulent=False,
            )

            yield (src_account_id, dest_account_id, tx_attrs)
