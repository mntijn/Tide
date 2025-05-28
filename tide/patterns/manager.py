# TODO: maybe give the pattern manager awareness of previously
# selected fraudulent entities so it can bias its selection logic.

# TODO: add a pattern selector that can be used to select patterns based on the graph


import random
from typing import Dict, Any, List, Tuple

from ..datastructures.enums import NodeType
from ..datastructures.attributes import TransactionAttributes
from .repeated_overseas_transfers import RepeatedOverseasTransfersPattern
from .rapid_fund_movement import RapidFundMovementPattern
from .front_business_activity import FrontBusinessPattern
from .base import EntitySelection


class PatternManager:
    """Manages injection of all patterns"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        self.graph_generator = graph_generator
        self.params = params

        self.patterns: Dict[str, Any] = {
            p.pattern_name: p for p in [
                RepeatedOverseasTransfersPattern(graph_generator, params),
                RapidFundMovementPattern(graph_generator, params),
                FrontBusinessPattern(graph_generator, params),
                # add new patterns here
            ]}

        # Initialize entity selector (set by graph generator)
        # With Option 2, GraphGenerator *is* the selector. PatternManager just holds patterns.
        # self.entity_selector = graph_generator # Deprecated: will be removed

    def inject_patterns(self):
        """
        Selects patterns, gets their transaction data, and then injects these transactions into the graph.
        """
        num_patterns_to_inject = self.params.get(
            "pattern_frequency", {}).get("num_illicit_patterns", 5)

        if not self.entity_selector:
            print("Error: Entity selector not initialized in PatternManager.")
            return []

        all_entities_count = 0
        if hasattr(self.graph_generator, 'all_nodes'):
            for node_type in NodeType:
                all_entities_count += len(
                    self.graph_generator.all_nodes.get(node_type, []))

        if all_entities_count < 10:
            print(
                f"Warning: Low number of total entities ({all_entities_count}) available for pattern selection.")

        actual_injected_edges_count = 0

        for _i in range(num_patterns_to_inject):
            if not self.patterns:
                print("Warning: No patterns available to inject.")
                break
            selected_pattern = random.choice(self.patterns)

            entity_selection_for_pattern: Optional[EntitySelection] = \
                self.entity_selector.select_entities_for_pattern(
                    pattern_name=selected_pattern.pattern_name,
                    num_entities_required=selected_pattern.num_required_entities
            )

            if not entity_selection_for_pattern:
                print(
                    f"Could not select entities for pattern: {selected_pattern.pattern_name}. Skipping.")
                continue

            try:
                transaction_data_list: List[Tuple[str, str, TransactionAttributes]] = \
                    selected_pattern.inject_pattern_with_selection(
                        entity_selection_for_pattern)

                if transaction_data_list:
                    print(
                        f"Pattern {selected_pattern.pattern_name} generated {len(transaction_data_list)} transaction descriptions.")
                    for src_id, dest_id, tx_attrs in transaction_data_list:
                        self.graph_generator._add_edge(
                            src_id, dest_id, tx_attrs)
                    actual_injected_edges_count += 1
                else:
                    print(
                        f"Pattern {selected_pattern.pattern_name} generated no transaction data with the selected entities.")

            except Exception as e:
                print(
                    f"Failed during transaction data generation or injection for {selected_pattern.pattern_name}: {e}")

        print(
            f"Total actual transaction edges injected into the graph: {actual_injected_edges_count}")
        return actual_injected_edges_count

    def get_available_patterns(self) -> List[str]:
        """Return list of available pattern names"""
        available = list(self.patterns.keys())
        return available
