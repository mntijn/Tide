from typing import Dict, Any, List

from .salary_payments import SalaryPaymentsPattern
from .cover_transactions import CoverTransactionsPattern
from .background_activity import RandomTransfersPattern


class BackgroundPatternManager:
    """Manages injection of all background activity patterns"""

    def __init__(self, graph_generator, params: Dict[str, Any]):
        self.graph_generator = graph_generator
        self.params = params

        self.patterns: Dict[str, Any] = {}

        bg_config = self.params.get("background_activity", {})
        if bg_config.get("enabled", False):
            self.patterns = {
                p.pattern_name: p for p in [
                    SalaryPaymentsPattern(graph_generator, params),
                    CoverTransactionsPattern(graph_generator, params),
                    RandomTransfersPattern(graph_generator, params),
                ]}

    def get_available_patterns(self) -> List[str]:
        """Return list of available pattern names"""
        available = list(self.patterns.keys())
        return available
