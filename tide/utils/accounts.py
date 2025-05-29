import logging
from ..datastructures.enums import NodeType, AccountCategory
from ..utils.constants import COUNTRY_TO_CURRENCY
from ..datastructures.attributes import OwnershipAttributes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_individual(graph_gen, ind_id, lock, sim_start_date):
    """Process a single individual."""
    try:
        node_data = graph_gen.graph.nodes[ind_id]
        ind_country_code = node_data.get("country_code")

        if not ind_country_code:
            logger.warning(
                f"Skipping individual {ind_id} - missing country_code")
            return

        accounts_and_ownerships = graph_gen.account.generate_accounts_and_ownership_data_for_entity(
            entity_node_type=NodeType.INDIVIDUAL,
            entity_country_code=ind_country_code,
            entity_data=node_data,
            sim_start_date=sim_start_date
        )

        for acc_creation_date, acc_common, acc_specific, owner_specific in accounts_and_ownerships:
            try:
                acc_id = graph_gen._add_node(
                    NodeType.ACCOUNT, acc_common, acc_specific, creation_date=acc_creation_date)
                ownership_instance = OwnershipAttributes(**owner_specific)
                graph_gen._add_edge(ind_id, acc_id, ownership_instance)
            except Exception as e:
                logger.error(
                    f"Error creating account for individual {ind_id}: {str(e)}")

        # Create cash account
        if ind_id not in graph_gen.individual_cash_accounts:
            try:
                cash_account_common = {
                    "country_code": ind_country_code,
                    "is_fraudulent": False,
                }
                cash_account_specific = {
                    "start_balance": 0.0,
                    "current_balance": 0.0,
                    "institution_id": None,
                    "account_category": AccountCategory.CASH,
                    "currency": COUNTRY_TO_CURRENCY[ind_country_code],
                }

                cash_acc_id = graph_gen._add_node(
                    NodeType.ACCOUNT,
                    cash_account_common,
                    cash_account_specific,
                )
                graph_gen._add_edge(
                    ind_id,
                    cash_acc_id,
                    OwnershipAttributes(
                        ownership_start_date=sim_start_date.date(),
                        ownership_percentage=100.0,
                    ),
                )
                graph_gen.individual_cash_accounts[ind_id] = cash_acc_id
            except Exception as e:
                logger.error(
                    f"Error creating cash account for individual {ind_id}: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing individual {ind_id}: {str(e)}")


def process_business(graph_gen, bus_id, lock, sim_start_date):
    """Process a single business."""
    try:
        node_data = graph_gen.graph.nodes[bus_id]
        bus_creation_date = node_data.get("creation_date")
        bus_country_code = node_data.get("country_code")

        if not bus_creation_date or not bus_country_code:
            logger.warning(
                f"Skipping business {bus_id} - missing creation_date, country_code, or address")
            return

        bus_accounts_and_ownerships = graph_gen.account.generate_accounts_and_ownership_data_for_entity(
            entity_node_type=NodeType.BUSINESS,
            entity_creation_date=bus_creation_date,
            entity_country_code=bus_country_code,
            entity_data=node_data,
            sim_start_date=sim_start_date
        )

        for acc_creation_date, acc_common, acc_specific, owner_specific in bus_accounts_and_ownerships:
            try:
                acc_id = graph_gen._add_node(
                    NodeType.ACCOUNT, acc_common, acc_specific, creation_date=acc_creation_date)
                ownership_instance = OwnershipAttributes(**owner_specific)
                graph_gen._add_edge(bus_id, acc_id, ownership_instance)
                logger.debug(
                    f"Created account {acc_id} for business {bus_id} with category {acc_specific.get('account_category')}")
            except Exception as e:
                logger.error(
                    f"Error creating account for business {bus_id}: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing business {bus_id}: {str(e)}")
