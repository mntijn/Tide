import random
import datetime
import yaml
from typing import Dict, Any

from tide.graph_generator import GraphGenerator


def load_configurations() -> Dict[str, Any]:
    """Load and merge main config and patterns config"""
    with open("configs/graph.yaml", 'r') as f:
        main_config = yaml.safe_load(f)

    with open("configs/patterns.yaml", 'r') as f:
        patterns_config = yaml.safe_load(f)

    if "pattern_config" not in main_config:
        main_config["pattern_config"] = {}

    for pattern_name, pattern_config in patterns_config.items():
        main_config["pattern_config"][pattern_name] = pattern_config

    return main_config


if __name__ == "__main__":
    # Load and merge configurations
    generator_parameters = load_configurations()

    # Convert date strings to datetime objects
    generator_parameters["time_span"]["start_date"] = datetime.datetime.fromisoformat(
        generator_parameters["time_span"]["start_date"])
    generator_parameters["time_span"]["end_date"] = datetime.datetime.fromisoformat(
        generator_parameters["time_span"]["end_date"])

    aml_graph_gen = GraphGenerator(params=generator_parameters)
    aml_graph_gen.generate_graph()

    print(f"\n--- Graph Summary ---")
    print(f"Number of nodes: {aml_graph_gen.num_of_nodes()}")
    print(f"Number of edges: {aml_graph_gen.num_of_edges()}")

    aml_graph_gen.export_to_csv(
        nodes_filepath="generated_nodes.csv", edges_filepath="generated_edges.csv")
