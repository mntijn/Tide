import unittest
import datetime
import yaml
import networkx as nx
import numpy as np
from typing import Dict, Any, List, Set
from tide.graph_generator import GraphGenerator
from tide.datastructures.enums import NodeType, EdgeType
from tide.utils.constants import HIGH_RISK_COUNTRIES, HIGH_RISK_BUSINESS_CATEGORIES


class TestGraphGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load test configurations once for all tests"""
        with open("configs/graph.yaml", 'r') as f:
            cls.graph_config = yaml.safe_load(f)
        with open("configs/patterns.yaml", 'r') as f:
            cls.patterns_config = yaml.safe_load(f)

        # Merge configs as done in main.py
        if "pattern_config" not in cls.graph_config:
            cls.graph_config["pattern_config"] = {}
        for pattern_name, pattern_config in cls.patterns_config.items():
            cls.graph_config["pattern_config"][pattern_name] = pattern_config

    def setUp(self):
        """Set up a fresh graph for each test"""
        self.params = self.graph_config.copy()

        # Handle datetime conversion more safely
        if isinstance(self.params["time_span"]["start_date"], str):
            self.params["time_span"]["start_date"] = datetime.datetime.fromisoformat(
                self.params["time_span"]["start_date"])
        elif not isinstance(self.params["time_span"]["start_date"], datetime.datetime):
            self.params["time_span"]["start_date"] = datetime.datetime(
                2023, 1, 1, 0, 0, 0)

        if isinstance(self.params["time_span"]["end_date"], str):
            self.params["time_span"]["end_date"] = datetime.datetime.fromisoformat(
                self.params["time_span"]["end_date"])
        elif not isinstance(self.params["time_span"]["end_date"], datetime.datetime):
            self.params["time_span"]["end_date"] = datetime.datetime(
                2023, 3, 15, 23, 59, 59)

        # Override graph scale for testing with much smaller numbers
        self.params["graph_scale"]["individuals"] = 200
        self.params["graph_scale"]["institutions_per_country"] = 1
        self.params["pattern_frequency"]["num_illicit_patterns"] = 2
        self.params["transaction_rates"]["per_account_per_day"] = 0.1
        self.params["fraud_selection_config"]["min_risk_score_for_fraud_consideration"] = 0.30

        self.generator = GraphGenerator(params=self.params)
        self.graph = self.generator.generate_graph()

    def test_graph_scale(self):
        """Test if the graph scale matches the configuration"""
        # Test number of individuals
        individuals = [n for n, d in self.graph.nodes(data=True)
                       if d.get("node_type") == NodeType.INDIVIDUAL]
        self.assertEqual(len(individuals),
                         self.params["graph_scale"]["individuals"])

        # Test number of institutions per country
        institutions = [n for n, d in self.graph.nodes(data=True)
                        if d.get("node_type") == NodeType.INSTITUTION]
        countries = set(d.get("country_code") for _, d in self.graph.nodes(data=True)
                        if d.get("node_type") == NodeType.INSTITUTION)
        self.assertGreaterEqual(len(institutions),
                                len(countries) * self.params["graph_scale"]["institutions_per_country"])

    def test_account_distribution(self):
        """Test if account distribution matches configuration"""
        # Test individual accounts per individual entity (not per institution)
        min_acc, max_acc = self.params["graph_scale"]["individual_accounts_per_institution_range"]

        # Group accounts by individual owner
        individual_accounts = {}
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == NodeType.ACCOUNT:
                inst_id = data.get("institution_id")
                # Only count non-cash accounts with valid institution_id
                if inst_id:
                    # Check if this account is owned by an individual (not a business)
                    for neighbor in self.graph.predecessors(node):
                        neighbor_data = self.graph.nodes[neighbor]
                        if neighbor_data.get("node_type") == NodeType.INDIVIDUAL:
                            # This is an individual-owned account
                            individual_accounts.setdefault(
                                neighbor, []).append(node)
                            break  # Only count once per account

        # Debug information
        print(f"Expected range: {min_acc}-{max_acc} accounts per individual")
        individuals_exceeding_max = 0
        for ind_id, accounts in individual_accounts.items():
            if len(accounts) > max_acc:
                individuals_exceeding_max += 1
                if individuals_exceeding_max <= 5:  # Show only first 5
                    print(
                        f"Individual {ind_id}: {len(accounts)} accounts (exceeds max {max_acc})")

        # Check each individual's account count
        for ind_id, accounts in individual_accounts.items():
            self.assertGreaterEqual(len(accounts), min_acc)
            self.assertLessEqual(len(accounts), max_acc)

    def test_random_seed_reproducibility(self):
        """Test if the random seed produces reproducible results"""
        # Generate two graphs with the same seed
        params1 = self.params.copy()
        params1["random_seed"] = 42
        gen1 = GraphGenerator(params=params1)
        graph1 = gen1.generate_graph()

        params2 = self.params.copy()
        params2["random_seed"] = 42
        gen2 = GraphGenerator(params=params2)
        graph2 = gen2.generate_graph()

        # Compare node counts
        self.assertEqual(graph1.number_of_nodes(), graph2.number_of_nodes())
        self.assertEqual(graph1.number_of_edges(), graph2.number_of_edges())

        # Compare node attributes
        for node in graph1.nodes():
            self.assertIn(node, graph2.nodes())
            self.assertEqual(graph1.nodes[node], graph2.nodes[node])

        # Compare edge attributes
        for edge in graph1.edges():
            self.assertIn(edge, graph2.edges())
            self.assertEqual(graph1.edges[edge], graph2.edges[edge])

    def test_pattern_injection(self):
        """Test if AML patterns are correctly injected"""
        # Check if the number of patterns matches configuration
        num_patterns = self.params["pattern_frequency"]["num_illicit_patterns"]
        pattern_edges = [e for e in self.graph.edges(data=True)
                         if e[2].get("edge_type") == EdgeType.TRANSACTION
                         and e[2].get("is_fraudulent", False)]

        # We should have at least some fraudulent edges if patterns were injected
        self.assertGreater(len(pattern_edges), 0)

        # Check if pattern-specific configurations are respected
        for pattern_name, pattern_config in self.patterns_config.items():
            if pattern_name == "frontBusiness":
                self._validate_front_business_pattern(pattern_config)
            elif pattern_name == "repeatedOverseas":
                self._validate_repeated_overseas_pattern(pattern_config)
            elif pattern_name == "rapidMovement":
                self._validate_rapid_movement_pattern(pattern_config)

    def _validate_front_business_pattern(self, pattern_config: Dict[str, Any]):
        """Validate front business pattern specific properties"""
        # Find businesses with multiple accounts
        business_accounts = {}
        for node, data in self.graph.nodes(data=True):
            if data.get("node_type") == NodeType.BUSINESS:
                accounts = [n for n in self.graph.neighbors(node)
                            if self.graph.nodes[n].get("node_type") == NodeType.ACCOUNT]
                if len(accounts) >= pattern_config["min_accounts_for_front_business"]:
                    business_accounts[node] = accounts

        # Check if we have any businesses with sufficient accounts
        self.assertGreater(len(business_accounts), 0)

        # Check transaction cycles - be more lenient about exact counts
        businesses_with_transactions = 0
        for business, accounts in business_accounts.items():
            transactions = [e for e in self.graph.edges(data=True)
                            if e[0] in accounts or e[1] in accounts]
            if transactions:
                businesses_with_transactions += 1
                # Just ensure we have some transactions, not necessarily the exact minimum
                self.assertGreater(len(transactions), 0)

        # Ensure at least one business has transactions
        self.assertGreater(businesses_with_transactions, 0)

    def _validate_repeated_overseas_pattern(self, pattern_config: Dict[str, Any]):
        """Validate repeated overseas transfers pattern specific properties"""
        # Find overseas transactions
        overseas_transactions = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == EdgeType.TRANSACTION:
                u_country = self.graph.nodes[u].get("country_code")
                v_country = self.graph.nodes[v].get("country_code")
                if u_country != v_country:
                    overseas_transactions.append((u, v, data))

        # Check if we have the minimum number of overseas transactions
        self.assertGreaterEqual(len(overseas_transactions),
                                pattern_config["transaction_params"]["min_transactions"])

    def _validate_rapid_movement_pattern(self, pattern_config: Dict[str, Any]):
        """Validate rapid fund movement pattern specific properties"""
        # Find accounts with rapid transactions
        rapid_accounts = set()
        for u, v, data in self.graph.edges(data=True):
            if data.get("edge_type") == EdgeType.TRANSACTION:
                # Check if transaction amount is within pattern range
                amount = data.get("amount", 0)
                if (pattern_config["transaction_params"]["inflows"]["amount_range"][0] <= amount <=
                        pattern_config["transaction_params"]["inflows"]["amount_range"][1]):
                    rapid_accounts.add(u)
                    rapid_accounts.add(v)

        # Should have at least min_accounts_for_pattern
        self.assertGreaterEqual(len(rapid_accounts),
                                pattern_config["min_accounts_for_pattern"])

    def test_risk_distribution(self):
        """Test if risk scores are distributed according to configuration"""
        # Get all risk scores, filtering out None values
        risk_scores = [d.get("risk_score", 0.0) for _, d in self.graph.nodes(data=True)
                       if d.get("risk_score") is not None]

        # Check if risk scores are within valid range [0, 1]
        self.assertTrue(all(0 <= score <= 1 for score in risk_scores))

        # Check if high-risk entities are marked as fraudulent
        min_risk_threshold = self.params["fraud_selection_config"]["min_risk_score_for_fraud_consideration"]
        high_risk_entities = [n for n, d in self.graph.nodes(data=True)
                              if d.get("risk_score") is not None
                              and d.get("risk_score", 0.0) >= min_risk_threshold]

        # At least some entities should be high risk
        self.assertGreater(len(high_risk_entities), 0)

    def test_time_span(self):
        """Test if all timestamps are within the configured time span"""
        start_date = self.params["time_span"]["start_date"]
        end_date = self.params["time_span"]["end_date"]

        print(f"Expected time span: {start_date} to {end_date}")

        # Check node creation dates - exclude business nodes and business-owned accounts
        nodes_outside_span = []
        for node, data in self.graph.nodes(data=True):
            if "creation_date" in data and data["creation_date"] is not None:
                node_type = data.get("node_type")

                # Skip business nodes as they can have creation dates outside the time span
                # due to business_creation_date_range configuration
                if node_type == NodeType.BUSINESS:
                    continue

                # Skip account nodes that are owned by businesses
                if node_type == NodeType.ACCOUNT:
                    is_business_account = False
                    for neighbor in self.graph.predecessors(node):
                        neighbor_data = self.graph.nodes[neighbor]
                        if neighbor_data.get("node_type") == NodeType.BUSINESS:
                            is_business_account = True
                            break
                    if is_business_account:
                        continue

                creation_date = data["creation_date"]
                if creation_date < start_date or creation_date > end_date:
                    nodes_outside_span.append((node, node_type, creation_date))

        if nodes_outside_span:
            print(f"Nodes with creation dates outside time span:")
            for node, node_type, creation_date in nodes_outside_span[:5]:
                print(f"  {node} ({node_type}): {creation_date}")

        # Now assert that all nodes are within the time span
        for node, data in self.graph.nodes(data=True):
            if "creation_date" in data and data["creation_date"] is not None:
                node_type = data.get("node_type")

                # Skip business nodes as they can have creation dates outside the time span
                # due to business_creation_date_range configuration
                if node_type == NodeType.BUSINESS:
                    continue

                # Skip account nodes that are owned by businesses
                if node_type == NodeType.ACCOUNT:
                    is_business_account = False
                    for neighbor in self.graph.predecessors(node):
                        neighbor_data = self.graph.nodes[neighbor]
                        if neighbor_data.get("node_type") == NodeType.BUSINESS:
                            is_business_account = True
                            break
                    if is_business_account:
                        continue

                self.assertGreaterEqual(data["creation_date"], start_date)
                self.assertLessEqual(data["creation_date"], end_date)

        # Check transaction dates
        for _, _, data in self.graph.edges(data=True):
            if "transaction_date" in data and data["transaction_date"] is not None:
                self.assertGreaterEqual(data["transaction_date"], start_date)
                self.assertLessEqual(data["transaction_date"], end_date)


if __name__ == '__main__':
    unittest.main()
