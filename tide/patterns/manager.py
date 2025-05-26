# TODO: maybe give the pattern manager awareness of previously
# selected fraudulent entities so it can bias its selection logic.

# TODO: add a pattern selector that can be used to select patterns based on the graph


import random
from typing import Dict, Any, List

from ..datastructures.enums import NodeType
from .repeated_overseas_transfers import RepeatedOverseasTransfersPattern
from .rapid_fund_movement import RapidFundMovementPattern
from .front_business_activity import FrontBusinessPattern


class PatternManager:
    """Manages injection of all patterns"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        self.graph_generator = graph_generator
        self.params = params

        self.patterns = [
            RepeatedOverseasTransfersPattern(graph_generator, params),
            RapidFundMovementPattern(graph_generator, params),
            FrontBusinessPattern(graph_generator, params),
            # add new patterns here
        ]

        # Initialize entity selector (set by graph generator)
        self.entity_selector = None

    def inject_patterns(self):
        """Inject specified number of patterns"""
        num_patterns = self.params.get(
            "pattern_frequency", {}).get("num_illicit_patterns", 5)

        # Check if we have enough entities
        all_entities = []
        for node_type in NodeType:
            all_entities.extend(
                self.graph_generator.all_nodes.get(node_type, []))

        if len(all_entities) < 10:
            print("Warning: Not enough entities to inject patterns")
            return

        fraudulent_edges = []

        for i in range(num_patterns):
            pattern_injector = random.choice(self.patterns)
            pattern_name = pattern_injector.pattern_name
            required_entities_for_pattern = pattern_injector.num_required_entities

            pattern_entities = self.entity_selector.select_entities_for_pattern(
                pattern_name, required_entities_for_pattern
            )

            try:
                edges = pattern_injector.inject_pattern(pattern_entities)
                fraudulent_edges.extend(edges)
                print(
                    f"Injected {pattern_injector.pattern_name} with {len(edges)} edges")
            except Exception as e:
                print(
                    f"Failed to inject {pattern_injector.pattern_name}: {e}")

        print(f"Total fraudulent edges created: {len(fraudulent_edges)}")
        return fraudulent_edges

    def get_available_patterns(self) -> List[str]:
        """Return list of available pattern names"""
        available = [
            pattern.pattern_name for pattern in self.patterns]
        return available
