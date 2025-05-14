import random
import datetime
from typing import Dict, Any

from graph_generator import GraphGenerator

if __name__ == "__main__":
    generator_parameters = {
        "graph_scale": {
            "individuals": 50,
            "businesses": 10,
            "institutions": 3,
            "individual_accounts_per_institution_range": (1, 3),
            "business_accounts_per_institution_range": (1, 6)
        },
        "transaction_rates": {
            # Transaction rates per account per day
            "per_account_per_day": 1
        },
        "time_span": {
            # Time span of the simulation, entities can be created before this date.
            # Format is (year, month, day, hour, minute, second)
            "start_date": datetime.datetime(2023, 1, 1, 0, 0, 0),
            "end_date": datetime.datetime(2023, 3, 15, 23, 59, 59)
        },
        "account_balance_range_normal": (1000.0, 50000.0),
        "background_amount_range": (10.0, 500.0),
        "salary_payment_day": 28,
        "salary_amount_range": (2500.0, 7500.0),
        "pattern_frequency": {
            # Number of illicit patterns to inject into the graph
            "num_illicit_patterns": 5
        },
        "amount_distribution_params": {"mean": 100, "variance": 50},
        "high_transaction_amount_ratio": 0.05,
        "low_transaction_amount_ratio": 0.1,
    }

    aml_graph_gen = GraphGenerator(params=generator_parameters)
    aml_graph_gen.generate_graph()

    print(f"\n--- Graph Summary ---")
    print(f"Number of nodes: {aml_graph_gen.num_of_nodes()}")
    print(f"Number of edges: {aml_graph_gen.num_of_edges()}")

    aml_graph_gen.export_to_csv(
        nodes_filepath="generated_nodes.csv", edges_filepath="generated_edges.csv")
