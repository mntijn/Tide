import logging
from typing import Dict, List, Set, Callable

from ..datastructures.enums import NodeType
from ..utils.constants import HIGH_RISK_COUNTRIES, HIGH_RISK_AGE_GROUPS, HIGH_RISK_OCCUPATIONS

logger = logging.getLogger(__name__)


# Configuration for single-factor clusters
SINGLE_FACTOR_CLUSTERS = {
    "high_risk_countries": {
        "condition": lambda data: data.get("country_code") in HIGH_RISK_COUNTRIES,
        "risk_factor": "high_risk_country"
    },
    "high_risk_business_categories": {
        "condition": lambda data: (data.get("node_type") == NodeType.BUSINESS and
                                   data.get("is_high_risk_category", False)),
        "risk_factor": "high_risk_business"
    },
    "high_risk_age_groups": {
        "condition": lambda data: (data.get("age_group") and
                                   str(data.get("age_group")).upper() in HIGH_RISK_AGE_GROUPS),
        "risk_factor": "high_risk_age"
    },
    "high_risk_occupations": {
        "condition": lambda data: data.get("occupation") in HIGH_RISK_OCCUPATIONS,
        "risk_factor": "high_risk_occupation"
    },
    "high_risk_score": {
        "condition": lambda data, min_score: (data.get("risk_score", 0.0) or 0.0) >= min_score,
        "risk_factor": "high_risk_score"
    }
}

# Configuration for composite clusters
COMPOSITE_CLUSTERS = {
    "intermediaries": {
        "conditions": [
            lambda data: (data.get("age_group") and
                          str(data.get("age_group")).upper() == "EIGHTEEN_TO_TWENTY_FOUR"),
            lambda data: (data.get("age_group") and
                          str(data.get("age_group")).upper() == "SIXTY_FIVE_PLUS"),
            lambda data: (data.get("occupation") and
                          data.get("occupation") in HIGH_RISK_OCCUPATIONS),
            lambda data: (data.get("node_type") == NodeType.INDIVIDUAL and
                          data.get("country_code") in HIGH_RISK_COUNTRIES),
            lambda data: (data.get("node_type") == NodeType.BUSINESS and
                          data.get("is_high_risk_category", False) and
                          data.get("number_of_employees", 0) <= 10)
        ]
    },
    "offshore_candidates": {
        "conditions": [
            lambda data: data.get("occupation") in {
                "Banker", "Investment banker", "Financial trader",
                "Lawyer", "Chartered accountant", "IT consultant"
            },
            lambda data: (data.get("node_type") == NodeType.BUSINESS and
                          data.get("business_category") in {
                "Private Banking", "Investment Banking",
                "Trust Services", "Currency Exchange"
            }),
            lambda data: (data.get("risk_score", 0.0) or 0.0) >= 0.7
        ]
    },
    "structuring_candidates": {
        "conditions": [
            lambda data: (data.get("node_type") == NodeType.BUSINESS and
                          data.get("business_category") in {
                "Casinos", "Currency Exchange", "Check Cashing Services",
                "Convenience Stores", "Gas Stations", "Bars and Nightclubs",
                "Pawn Shops", "Laundromats"
            }),
            lambda data: (data.get("node_type") == NodeType.INDIVIDUAL and
                          data.get("occupation") and
                          any(keyword in data.get("occupation", "").lower()
                              for keyword in ["bank", "financial", "accountant", "trader"]))
        ]
    }
}


def evaluate_entity_clustering(data: Dict, min_risk_score: float) -> tuple[Set[str], Set[str]]:
    """Evaluate which clusters an entity belongs to and its risk factors.

    Returns:
        tuple: (clusters, risk_factors) sets
    """
    clusters = set()
    risk_factors = set()

    # Evaluate single-factor clusters
    for cluster_name, config in SINGLE_FACTOR_CLUSTERS.items():
        condition = config["condition"]

        # Handle special case for risk score which needs additional parameter
        if cluster_name == "high_risk_score":
            if condition(data, min_risk_score):
                clusters.add(cluster_name)
                risk_factors.add(config["risk_factor"])
        else:
            if condition(data):
                clusters.add(cluster_name)
                risk_factors.add(config["risk_factor"])

    # Add super_high_risk for entities with 3+ risk factors
    if len(risk_factors) >= 3:
        clusters.add("super_high_risk")

    # Evaluate composite clusters
    for cluster_name, config in COMPOSITE_CLUSTERS.items():
        if any(condition(data) for condition in config["conditions"]):
            clusters.add(cluster_name)

        # Special case: structuring_candidates also includes entities with 2+ risk factors
        if cluster_name == "structuring_candidates" and len(risk_factors) >= 2:
            clusters.add(cluster_name)

    return clusters, risk_factors


def build_entity_clusters(graph_generator) -> Dict[str, List[str]]:
    """Build entity clusters for pattern targeting.

    Pattern Source Selection:
    1. super_high_risk/high_risk_score: Entities with 3+ risk factors
    2. offshore_candidates: Financial sophistication for international operations
    3. structuring_candidates: Usually executors, not sources

    Other Roles:
    - intermediaries: Money mules/fronts (often unwitting participants)
    - Basic single-factor clusters: Country, business category, age, occupation
    - legit: Non-fraudulent entities (populated by default, updated during fraud injection)

    Note: Clusters contain only entities (individuals and businesses).
    Patterns find accounts by traversing the graph from these entities.

    Args:
        graph_generator: The GraphGenerator instance containing graph and configuration

    Returns:
        Dict[str, List[str]]: Mapping of cluster names to lists of entity IDs
    """
    logger.info("Building entity clusters...")
    min_risk_score = graph_generator.fraud_selection_config.get(
        "min_risk_score_for_fraud_consideration", 0.30)

    # Initialize all clusters (including legit which will be populated later)
    all_cluster_names = (
        list(SINGLE_FACTOR_CLUSTERS.keys()) +
        ["super_high_risk"] +
        list(COMPOSITE_CLUSTERS.keys()) +
        ["fraudulent", "legit"]
    )
    clusters: Dict[str, Set[str]] = {name: set() for name in all_cluster_names}

    # Single pass through all nodes to build all clusters
    # Sort nodes for deterministic iteration order
    for node_id, data in sorted(graph_generator.graph.nodes(data=True)):
        # Skip accounts and institutions for clustering (focus on individuals/businesses)
        if data.get("node_type") in [NodeType.ACCOUNT, NodeType.INSTITUTION]:
            continue

        entity_clusters, _ = evaluate_entity_clustering(data, min_risk_score)

        # Add entity to all applicable clusters
        for cluster_name in entity_clusters:
            clusters[cluster_name].add(node_id)

    # Convert sets to sorted lists for deterministic order
    result_clusters = {name: sorted(list(cluster_set))
                       for name, cluster_set in clusters.items()}

    # Log cluster sizes for debugging
    cluster_summary = [(k, len(v)) for k, v in result_clusters.items() if v]
    logger.info(f"Built entity clusters: {cluster_summary}")

    # Log entity counts in clusters (all cluster items are now entities)
    for cluster_name, entities in result_clusters.items():
        if entities:
            logger.info(f"  {cluster_name}: {len(entities)} entities")

    # Log overlap statistics for insight
    total_entities = len([n for n, d in graph_generator.graph.nodes(data=True)
                          if d.get("node_type") in [NodeType.INDIVIDUAL, NodeType.BUSINESS]])
    super_high_risk_count = len(result_clusters["super_high_risk"])
    if total_entities > 0:
        logger.info(f"Super high-risk entities: {super_high_risk_count}/{total_entities} "
                    f"({super_high_risk_count/total_entities*100:.1f}%)")

    return result_clusters
