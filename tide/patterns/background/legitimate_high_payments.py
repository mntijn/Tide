import datetime
import calendar
from typing import List, Dict, Any, Tuple

from ..base import (
    StructuralComponent, TemporalComponent, EntitySelection, TransactionSequence,
    CompositePattern, PatternInjector
)
from ...datastructures.enums import NodeType, TransactionType
from ...datastructures.attributes import TransactionAttributes
from ...utils.random_instance import random_instance


class LegitimateHighPaymentsStructural(StructuralComponent):
    """Structural component: select legitimate entities for high-value transactions."""

    @property
    def num_required_entities(self) -> int:
        return 2  # Need at least 2 entities for transactions

    def select_entities(self, available_entities: List[str]) -> EntitySelection:
        # Get legitimate entities only
        legit_entities = self.graph_generator.entity_clusters.get("legit", [])
        if not legit_entities:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Separate individuals and businesses for different types of high-value transactions
        individual_entities = []
        business_entities = []

        for entity_id in legit_entities:
            if self.graph_generator.graph.has_node(entity_id):
                node_data = self.graph_generator.graph.nodes[entity_id]
                if node_data.get("node_type") == NodeType.INDIVIDUAL:
                    individual_entities.append(entity_id)
                elif node_data.get("node_type") == NodeType.BUSINESS:
                    business_entities.append(entity_id)

        # Get accounts for selected entities
        individual_accounts = []
        business_accounts = []

        for individual_id in individual_entities:
            individual_account_ids = self._get_owned_accounts(individual_id)
            for account_id in individual_account_ids:
                individual_accounts.append((account_id, individual_id))

        for business_id in business_entities:
            business_account_ids = self._get_owned_accounts(business_id)
            for account_id in business_account_ids:
                business_accounts.append((account_id, business_id))

        # All legitimate accounts can participate in high-value transactions
        all_legit_accounts = individual_accounts + business_accounts

        if len(all_legit_accounts) < 2:
            return EntitySelection(central_entities=[], peripheral_entities=[])

        # Split into two groups for transactions (buyers and sellers)
        random_instance.shuffle(all_legit_accounts)
        midpoint = len(all_legit_accounts) // 2

        return EntitySelection(
            central_entities=all_legit_accounts[:midpoint],     # Buyers/payers
            # Sellers/recipients
            peripheral_entities=all_legit_accounts[midpoint:]
        )


class LegitimateHighPaymentsTemporal(TemporalComponent):
    """Temporal component: generate occasional high-value legitimate transactions."""

    def generate_transaction_sequences(self, entity_selection: EntitySelection) -> List[TransactionSequence]:
        sequences: List[TransactionSequence] = []
        buyer_accounts = entity_selection.central_entities
        seller_accounts = entity_selection.peripheral_entities

        if not buyer_accounts or not seller_accounts:
            return sequences

        # Get legitimate high payments configuration
        high_payments_config = self.params.get(
            "backgroundPatterns", {}).get("legitimateHighPayments", {})

        # High payment amount ranges
        amount_ranges = high_payments_config.get("high_payment_ranges", {
            "property_transactions": [50000.0, 500000.0],
            "business_deals": [10000.0, 100000.0],
            "luxury_purchases": [5000.0, 50000.0]
        })

        # Payment type probabilities
        type_probs = high_payments_config.get("high_payment_type_probabilities", {
            "property": 0.2,
            "business": 0.6,
            "luxury": 0.2
        })

        # Rate of high payments (per account per month)
        monthly_rate = high_payments_config.get(
            "high_payment_rate_per_month", 0.1)

        start_date: datetime.datetime = self.time_span["start_date"]
        end_date: datetime.datetime = self.time_span["end_date"]

        # Calculate total months in time span
        total_months = ((end_date.year - start_date.year) * 12 +
                        (end_date.month - start_date.month))
        if total_months == 0:
            total_months = 1

        all_transactions = []
        pattern_injector = PatternInjector(self.graph_generator, self.params)

        # Generate high-value transactions for each buyer account
        for buyer_account_id, buyer_entity_id in buyer_accounts:
            # Determine how many high-value transactions this account will make
            expected_transactions = total_months * monthly_rate
            num_transactions = max(
                0, int(random_instance.poisson(expected_transactions)))

            if num_transactions == 0:
                continue

            # Generate transaction dates spread across the time span
            tx_dates = []
            for _ in range(num_transactions):
                # Random date within time span
                total_days = (end_date - start_date).days
                random_day = random_instance.randint(0, total_days - 1)

                # Business hours for high-value transactions (more formal timing)
                random_hour = random_instance.randint(9, 17)  # 9 AM - 5 PM
                random_minute = random_instance.randint(0, 59)

                tx_date = start_date + datetime.timedelta(
                    days=random_day, hours=random_hour, minutes=random_minute
                )
                tx_dates.append(tx_date)

            tx_dates.sort()  # Chronological order

            # Generate high-value transactions
            available_sellers = seller_accounts.copy()
            random_instance.shuffle(available_sellers)

            for tx_date in tx_dates:
                if not available_sellers:
                    available_sellers = seller_accounts.copy()
                    random_instance.shuffle(available_sellers)

                # Select seller
                seller_account_id, seller_entity_id = None, None
                for seller_acc, seller_ent in available_sellers:
                    if seller_ent != buyer_entity_id:  # Different entities
                        seller_account_id, seller_entity_id = seller_acc, seller_ent
                        available_sellers.remove((seller_acc, seller_ent))
                        break

                if not seller_account_id:
                    continue  # Skip if no suitable seller found

                # Determine transaction type and amount
                rand_val = random_instance.random()
                cumulative_prob = 0.0

                payment_type = None
                for pay_type, prob in type_probs.items():
                    cumulative_prob += prob
                    if rand_val <= cumulative_prob:
                        payment_type = pay_type
                        break

                if not payment_type:
                    payment_type = "business"  # Default

                # Get amount range based on payment type
                if payment_type == "property":
                    amount_range = amount_ranges["property_transactions"]
                    tx_type = TransactionType.TRANSFER  # Property transfers
                elif payment_type == "business":
                    amount_range = amount_ranges["business_deals"]
                    tx_type = TransactionType.PAYMENT   # Business payments
                else:  # luxury
                    amount_range = amount_ranges["luxury_purchases"]
                    tx_type = TransactionType.PAYMENT   # Luxury purchases

                # Generate amount (round to nearest 100 for realism)
                amount = random_instance.uniform(
                    amount_range[0], amount_range[1])
                amount = round(amount / 100) * 100  # Round to nearest 100

                tx_attrs = pattern_injector._create_transaction_edge(
                    src_id=buyer_account_id,
                    dest_id=seller_account_id,
                    timestamp=tx_date,
                    amount=float(amount),
                    transaction_type=tx_type,
                    is_fraudulent=False,
                )

                all_transactions.append(
                    (buyer_account_id, seller_account_id, tx_attrs))

        # Create sequences grouped by transaction type for better organization
        if all_transactions:
            # Sort by timestamp
            all_transactions.sort(key=lambda x: x[2].timestamp)

            # Group by quarter for manageable sequences
            quarterly_transactions = self._group_transactions_by_quarter(
                all_transactions)

            for quarter, quarter_transactions in quarterly_transactions.items():
                if quarter_transactions:
                    sequences.append(TransactionSequence(
                        transactions=quarter_transactions,
                        sequence_name=f"legitimate_high_payments_{quarter}",
                        start_time=quarter_transactions[0][2].timestamp,
                        duration=quarter_transactions[-1][2].timestamp -
                        quarter_transactions[0][2].timestamp
                    ))

        return sequences

    def _group_transactions_by_quarter(self, transactions: List[Tuple]) -> Dict[str, List[Tuple]]:
        """Group transactions by calendar quarter for better organization."""
        quarterly_groups = {}

        for tx in transactions:
            timestamp = tx[2].timestamp
            quarter = f"{timestamp.year}_Q{(timestamp.month - 1) // 3 + 1}"

            if quarter not in quarterly_groups:
                quarterly_groups[quarter] = []
            quarterly_groups[quarter].append(tx)

        return quarterly_groups


class LegitimateHighPaymentsPattern(CompositePattern):
    """Legitimate high payments pattern for property, business deals, and luxury purchases."""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        structural_component = LegitimateHighPaymentsStructural(
            graph_generator, params)
        temporal_component = LegitimateHighPaymentsTemporal(
            graph_generator, params)
        super().__init__(structural_component, temporal_component, graph_generator, params)

    @property
    def pattern_name(self) -> str:
        return "LegitimateHighPayments"

    @property
    def num_required_entities(self) -> int:
        return self.structural.num_required_entities
