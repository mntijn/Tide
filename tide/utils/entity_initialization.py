import datetime
import logging
from typing import Dict, Any

from ..datastructures.enums import NodeType
from ..datastructures.attributes import OwnershipAttributes
from ..entities import Account
from ..utils.business import map_occupation_to_business_category, get_random_business_category
from ..utils.accounts import process_individual, process_business
from ..utils.faker_instance import reset_faker_seed

logger = logging.getLogger(__name__)


def initialize_entities(graph_generator):
    """Initialize all entities (institutions, individuals, businesses, accounts) for the graph.

    Args:
        graph_generator: The GraphGenerator instance containing all necessary state
    """
    logger.info("Starting entity generation...")

    # Reset random seeds if defined for reproducibility
    if graph_generator.random_seed:
        logger.info(
            f"Deterministic mode enabled with seed {graph_generator.random_seed}")
        graph_generator.random_instance.seed(graph_generator.random_seed)
        # Also reset global random module
        import random
        random.seed(graph_generator.random_seed)
        # Reset Faker instance
        reset_faker_seed()

    sim_start_date = graph_generator.time_span["start_date"]

    # Generate entity data sequentially
    logger.info("Generating institutions...")
    institutions_data = graph_generator.institution.generate_data()

    logger.info("Generating individuals...")
    individuals_data = graph_generator.individual.generate_data()

    # Create accounts for Institutions
    institution_countries = {}
    if institutions_data:
        for common_attrs, specific_attrs in institutions_data:
            try:
                institution_id = graph_generator._add_node(NodeType.INSTITUTION, common_attrs,
                                                           specific_attrs, creation_date=None)
                institution_countries[institution_id] = common_attrs["country_code"]
            except Exception as e:
                logger.error(f"Error creating institution node: {str(e)}")

    all_institution_ids = graph_generator.all_nodes.get(
        NodeType.INSTITUTION, [])

    graph_generator.account = Account(
        graph_generator.params, all_institution_ids, institution_countries)

    # Create accounts for Individuals and create owned businesses
    num_owned_businesses_created = 0
    if individuals_data:
        for ind_common_attrs, ind_specific_attrs in individuals_data:
            try:
                # Add the individual node
                ind_id = graph_generator._add_node(
                    NodeType.INDIVIDUAL,
                    ind_common_attrs,
                    ind_specific_attrs,
                )

                # Determine if this individual's occupation suggests business ownership
                occupation_str: str = ind_specific_attrs.get("occupation", "")
                suggested_category = map_occupation_to_business_category(
                    occupation_str)

                # Create business if occupation suggests one, or randomly
                should_create_business = (suggested_category is not None or
                                          graph_generator.random_instance.random() < graph_generator.random_business_probability)

                if should_create_business:
                    # Use suggested category or random one
                    business_category = suggested_category or get_random_business_category()

                    business_data_tuple = graph_generator.business.generate_age_consistent_business_for_individual(
                        individual_age_group=ind_specific_attrs["age_group"],
                        sim_start_date=sim_start_date,
                        business_category_override=business_category,
                        owner_occupation=occupation_str,
                        owner_risk_score=ind_common_attrs.get(
                            "risk_score", 0.0),
                        owner_country=ind_common_attrs.get("country_code"),
                    )

                    bus_creation_date, bus_common_attrs, bus_specific_attrs = business_data_tuple

                    bus_id = graph_generator._add_node(
                        NodeType.BUSINESS,
                        bus_common_attrs,
                        bus_specific_attrs,
                        creation_date=bus_creation_date,
                    )

                    ownership_attrs = OwnershipAttributes(
                        ownership_start_date=bus_creation_date.date(),
                        ownership_percentage=100.0,
                    )

                    graph_generator._add_edge(ind_id, bus_id, ownership_attrs)
                    num_owned_businesses_created += 1
            except Exception as e:
                logger.error(f"Error processing individual data: {str(e)}")

    # Create additional businesses if needed
    target_business_count = graph_generator.graph_scale.get("businesses", 0)
    if target_business_count > num_owned_businesses_created:
        additional_needed = target_business_count - num_owned_businesses_created

        individual_ids = graph_generator.all_nodes.get(NodeType.INDIVIDUAL, [])
        if not individual_ids:
            logger.warning(
                "No individuals available to own extra businesses. Skipping override.")
        else:
            for i in range(additional_needed):
                try:
                    owner_id = graph_generator.random_instance.choice(
                        individual_ids)
                    owner_data = graph_generator.graph.nodes[owner_id]

                    owner_age_group = owner_data.get("age_group")

                    bus_tuple = graph_generator.business.generate_age_consistent_business_for_individual(
                        individual_age_group=owner_age_group,
                        sim_start_date=sim_start_date,
                        owner_occupation=owner_data.get("occupation", ""),
                        owner_risk_score=owner_data.get("risk_score", 0.0),
                        owner_country=owner_data.get("country_code"),
                    )

                    bus_creation_date, bus_common_attrs, bus_specific_attrs = bus_tuple

                    bus_id = graph_generator._add_node(
                        NodeType.BUSINESS,
                        bus_common_attrs,
                        bus_specific_attrs,
                        creation_date=bus_creation_date,
                    )

                    graph_generator._add_edge(
                        owner_id,
                        bus_id,
                        OwnershipAttributes(
                            ownership_start_date=bus_creation_date.date(),
                            ownership_percentage=100.0,
                        ),
                    )
                    num_owned_businesses_created += 1
                except Exception as e:
                    logger.error(
                        f"Error creating additional business: {str(e)}")

    logger.info(f"Created {num_owned_businesses_created} total businesses")

    # Generate accounts sequentially
    logger.info("Generating accounts...")
    individual_ids = graph_generator.all_nodes.get(NodeType.INDIVIDUAL, [])
    business_ids = graph_generator.all_nodes.get(NodeType.BUSINESS, [])

    # Process all individuals
    for ind_id in individual_ids:
        process_individual(graph_generator, ind_id, None, sim_start_date)

    # Process all businesses
    for bus_id in business_ids:
        process_business(graph_generator, bus_id, None, sim_start_date)

    logger.info("Account generation completed")

    # Create global cash system account
    if graph_generator.cash_account_id is None:
        try:
            graph_generator.cash_account_id = graph_generator._add_node(
                NodeType.ACCOUNT,
                {"country_code": "CASH"},
                {"start_balance": 0.0, "current_balance": 0.0, "currency": "EUR"},
                creation_date=graph_generator.time_span["start_date"],
            )
            logger.info(
                f"Created global cash system account: {graph_generator.cash_account_id}")
        except Exception as e:
            logger.error(
                f"Error creating global cash system account: {str(e)}")

    logger.info("Entity and account generation complete")
