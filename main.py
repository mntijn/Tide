import random
import datetime
import yaml
from typing import Dict, Any

from graph_generator import GraphGenerator

if __name__ == "__main__":
    with open("config.yaml", 'r') as f:
        generator_parameters = yaml.safe_load(f)

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
