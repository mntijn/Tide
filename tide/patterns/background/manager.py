from typing import Dict, Any, List

from .background_activity import RandomPaymentsPattern
from .salary_payments import SalaryPaymentsPattern
from .fraudster_background import FraudsterBackgroundPattern


class BackgroundPatternManager:
    """Manages injection of background patterns"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        self.graph_generator = graph_generator
        self.params = params

        self.patterns: Dict[str, Any] = {
            p.pattern_name: p for p in [
                RandomPaymentsPattern(graph_generator, params),
                SalaryPaymentsPattern(graph_generator, params),
                FraudsterBackgroundPattern(graph_generator, params),
                # add new background patterns here
            ]}

    def get_available_patterns(self) -> List[str]:
        """Return list of available background pattern names"""
        available = list(self.patterns.keys())
        return available
