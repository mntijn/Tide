import logging
from typing import Dict, List

from ..datastructures.enums import NodeType
from ..utils.constants import HIGH_RISK_COUNTRIES, HIGH_RISK_AGE_GROUPS, HIGH_RISK_OCCUPATIONS

logger = logging.getLogger(__name__)


def build_entity_clusters(graph_generator) -> Dict[str, List[str]]:
    """Build entity clusters for pattern targeting.

    Pattern Source Selection:
    1. super_high_risk/high_risk_score: Entities with 3+ risk factors
    2. offshore_candidates: Financial sophistication for international operations
    3. structuring_candidates: Usually executors, not sources

    Other Roles:
    - intermediaries: Money mules/fronts (often unwitting participants)
    - Basic single-factor clusters: Country, business category, age, occupation

    Note: Clusters now include both entities AND their accounts for direct pattern usage.

    Args:
        graph_generator: The GraphGenerator instance containing graph and configuration

    Returns:
        Dict[str, List[str]]: Mapping of cluster names to lists of node IDs
    """
    logger.info("Building entity clusters...")
    min_risk_score = graph_generator.fraud_selection_config.get(
        "min_risk_score_for_fraud_consideration", 0.30)

    # Build clusters using a more sophisticated approach that recognizes overlapping risk factors
    clusters: Dict[str, List[str]] = {
        # Basic single-factor clusters
        "high_risk_countries": [],
        "high_risk_business_categories": [],
        "high_risk_age_groups": [],
        "high_risk_occupations": [],
        "high_risk_score": [],

        # Composite clusters for entities with multiple risk factors
        "super_high_risk": [],  # Multiple high-risk factors
        "intermediaries": [],   # Potential intermediaries
        "offshore_candidates": [],  # Likely to have offshore connections
        "structuring_candidates": [],  # Likely to engage in structuring
        "fraudulent": [],
    }

    # Helper function to add entity and its accounts to clusters
    def add_entity_and_accounts_to_cluster(cluster_name: str, entity_id: str):
        """Add both the entity and all its accounts to the specified cluster"""
        clusters[cluster_name].append(entity_id)

        # Find all accounts owned by this entity
        for neighbor_id in graph_generator.graph.neighbors(entity_id):
            neighbor_data = graph_generator.graph.nodes.get(neighbor_id, {})
            if neighbor_data.get("node_type") == NodeType.ACCOUNT:
                clusters[cluster_name].append(neighbor_id)

        # Also check if this entity owns businesses that have accounts
        if graph_generator.graph.nodes[entity_id].get("node_type") == NodeType.INDIVIDUAL:
            for owned_business_id in graph_generator.graph.neighbors(entity_id):
                business_data = graph_generator.graph.nodes.get(
                    owned_business_id, {})
                if business_data.get("node_type") == NodeType.BUSINESS:
                    # Add business accounts too
                    for business_account_id in graph_generator.graph.neighbors(owned_business_id):
                        business_account_data = graph_generator.graph.nodes.get(
                            business_account_id, {})
                        if business_account_data.get("node_type") == NodeType.ACCOUNT:
                            clusters[cluster_name].append(business_account_id)

    # Single pass through all nodes to build all clusters
    for node_id, data in graph_generator.graph.nodes(data=True):
        node_type = data.get("node_type")
        country = data.get("country_code")
        risk_score = data.get("risk_score", 0.0)

        # Handle None risk_score values explicitly
        if risk_score is None:
            risk_score = 0.0

        # Skip accounts for most clusters (focus on individuals/businesses)
        if node_type == NodeType.ACCOUNT:
            continue

        risk_factors = []  # Track all risk factors for this entity

        # Check individual risk factors
        if country in HIGH_RISK_COUNTRIES:
            add_entity_and_accounts_to_cluster("high_risk_countries", node_id)
            risk_factors.append("high_risk_country")

        if node_type == NodeType.BUSINESS and data.get("is_high_risk_category", False):
            add_entity_and_accounts_to_cluster(
                "high_risk_business_categories", node_id)
            risk_factors.append("high_risk_business")

        age_group = data.get("age_group")
        if age_group and str(age_group).upper() in HIGH_RISK_AGE_GROUPS:
            add_entity_and_accounts_to_cluster("high_risk_age_groups", node_id)
            risk_factors.append("high_risk_age")

        occupation = data.get("occupation")
        if occupation and occupation in HIGH_RISK_OCCUPATIONS:
            add_entity_and_accounts_to_cluster(
                "high_risk_occupations", node_id)
            risk_factors.append("high_risk_occupation")

        if risk_score >= min_risk_score:
            add_entity_and_accounts_to_cluster("high_risk_score", node_id)
            risk_factors.append("high_risk_score")

        # Build composite clusters based on multiple risk factors

        # Super high risk: entities with 3+ risk factors
        if len(risk_factors) >= 3:
            add_entity_and_accounts_to_cluster("super_high_risk", node_id)

        # Intermediaries: comprehensive criteria for potential money mules/intermediaries
        is_intermediary = False

        # Young adults (18-24) are often recruited as intermediaries
        if age_group and str(age_group).upper() == "EIGHTEEN_TO_TWENTY_FOUR":
            is_intermediary = True

        # Elderly (65+) can be vulnerable to being used as intermediaries
        elif age_group and str(age_group).upper() == "SIXTY_FIVE_PLUS":
            is_intermediary = True

        # High-risk occupations with access to financial systems
        elif occupation and occupation in HIGH_RISK_OCCUPATIONS:
            is_intermediary = True

        # Individuals in high-risk countries (easier to use as intermediaries)
        elif node_type == NodeType.INDIVIDUAL and country in HIGH_RISK_COUNTRIES:
            is_intermediary = True

        # Small businesses in high-risk categories (often used as fronts)
        elif (node_type == NodeType.BUSINESS and
              data.get("is_high_risk_category", False) and
              data.get("number_of_employees", 0) <= 10):
            is_intermediary = True

        if is_intermediary:
            add_entity_and_accounts_to_cluster("intermediaries", node_id)

        # Offshore candidates: entities likely to have or use offshore accounts
        is_offshore_candidate = False

        # High-paid occupations often have offshore accounts
        if occupation and occupation in ["Banker", "Investment banker", "Financial trader",
                                         "Lawyer", "Chartered accountant", "IT consultant"]:
            is_offshore_candidate = True

        # Businesses in high-risk categories often use offshore structures
        elif (node_type == NodeType.BUSINESS and
              data.get("business_category") in ["Private Banking", "Investment Banking",
                                                "Trust Services", "Currency Exchange"]):
            is_offshore_candidate = True

        # Entities from high-risk countries
        elif country in HIGH_RISK_COUNTRIES:
            is_offshore_candidate = True

        # High overall risk score
        elif risk_score >= 0.7:
            is_offshore_candidate = True

        if is_offshore_candidate:
            add_entity_and_accounts_to_cluster("offshore_candidates", node_id)

        # Structuring candidates: entities likely to engage in transaction structuring
        is_structuring_candidate = False

        # Cash-intensive businesses
        cash_businesses = ["Casinos", "Currency Exchange", "Check Cashing Services",
                           "Convenience Stores", "Gas Stations", "Bars and Nightclubs",
                           "Pawn Shops", "Laundromats"]
        if (node_type == NodeType.BUSINESS and
                data.get("business_category") in cash_businesses):
            is_structuring_candidate = True

        # Individuals with financial backgrounds who know the rules
        elif (node_type == NodeType.INDIVIDUAL and occupation and
              any(keyword in occupation.lower() for keyword in
                  ["bank", "financial", "accountant", "trader"])):
            is_structuring_candidate = True

        # Multiple risk factors (sophisticated actors)
        elif len(risk_factors) >= 2:
            is_structuring_candidate = True

        if is_structuring_candidate:
            add_entity_and_accounts_to_cluster(
                "structuring_candidates", node_id)

    # Deduplicate all clusters (since entities might be added multiple times)
    for cluster_name in clusters:
        clusters[cluster_name] = list(set(clusters[cluster_name]))

    # Log cluster sizes for debugging
    cluster_summary = [(k, len(v)) for k, v in clusters.items() if v]
    logger.info(f"Built entity clusters: {cluster_summary}")

    # Log breakdown of entities vs accounts in clusters
    for cluster_name, entities in clusters.items():
        if entities:
            entity_count = len([e for e in entities if graph_generator.graph.nodes.get(
                e, {}).get("node_type") in [NodeType.INDIVIDUAL, NodeType.BUSINESS]])
            account_count = len([e for e in entities if graph_generator.graph.nodes.get(
                e, {}).get("node_type") == NodeType.ACCOUNT])
            logger.info(
                f"  {cluster_name}: {entity_count} entities + {account_count} accounts = {len(entities)} total")

    # Log overlap statistics for insight
    total_entities = len([n for n, d in graph_generator.graph.nodes(data=True)
                          if d.get("node_type") in [NodeType.INDIVIDUAL, NodeType.BUSINESS]])
    super_high_risk_entities = [e for e in clusters["super_high_risk"]
                                if graph_generator.graph.nodes.get(e, {}).get("node_type") in [NodeType.INDIVIDUAL, NodeType.BUSINESS]]
    super_high_risk_count = len(super_high_risk_entities)
    if total_entities > 0:
        logger.info(f"Super high-risk entities: {super_high_risk_count}/{total_entities} "
                    f"({super_high_risk_count/total_entities*100:.1f}%)")

    return clusters
