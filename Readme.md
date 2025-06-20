# Tide: A Customisable Dataset Generator for Benchmarking Money Laundering Detection

## Config

There are 2 configs:
1. graph.yaml
2. patterns.yaml

Graphs.yaml allows for more general graph settings. Patterns.yaml allows for pattern-specific settings.

## Running the generator
To run, just create a venv, run `pip install -r requirements.txt`, and then `python main.py`

## Docs

<!-- ## Entity creation

Specify the number of individuals and institutions for the graph. Businesses will be generated based on individual attributes, but you can set a minimum number of businesses. Tide will create additional businesses as needed to meet this minimum. -->

## Code Structure

- **`tide/graph_generator.py`**: The main orchestrator for graph generation. It initializes entities and injects both illicit patterns and background activity.
- **`tide/entities/`**: Defines the graph's fundamental building blocks: `Individual`, `Business`, `Account`, and `Institution`.
- **`tide/patterns/`**: Contains implementations of specific money laundering schemes (e.g., `RapidFundMovement`, `UTurnTransactions`).
- **`tide/patterns/background/`**: Generates legitimate-looking "background" financial activity (e.g., salaries, regular payments) to create a realistic dataset.

## Implemented Patterns

This directory contains the logic for generating specific money laundering typologies. Each pattern is defined by its structural components (the types of entities involved) and its temporal behavior (the timing and sequence of transactions).

- **`front_business_activity.py`**: Simulates a front business that receives frequent, large cash deposits and quickly funnels the money to overseas business accounts.
- **`rapid_fund_movement.py`**: Models a scenario where a high-risk individual receives numerous small electronic transfers from various overseas accounts and then rapidly withdraws the aggregated funds in cash.
- **`repeated_overseas_transfers.py`**: Involves a high-risk individual making repeated, structured transfers from a domestic account to multiple accounts in foreign, high-risk jurisdictions.
- **`synchronised_transactions.py`**: Creates a pattern where multiple, seemingly unrelated individuals deposit funds into their own accounts, which are then almost simultaneously transferred to a common recipient.
- **`u_turn_transactions.py`**: Implements a "U-turn" or "round-trip" transaction, where funds are sent from an originator through a series of intermediary accounts (often in offshore jurisdictions) before returning to an account controlled by the originator.

## Experiments

This directory contains scripts for validating the output of the data generator against a series of hypotheses. These experiments ensure the generator is working as expected in terms of correctness, scalability, and reproducibility.

- **`h1_pattern_validation.py`**: Verifies that the generated money laundering patterns adhere to their formal definitions. It checks entity selection, transaction amounts, and the temporal sequence of events for each pattern.
- **`h2_scalability.py`**: Tests the generator's performance as the graph size increases. It measures wall time, memory usage, and other metrics across various scales to ensure the system can handle large datasets.
- **`h3_config_compliance.py`**: Ensures that the statistical properties of the generated dataset (e.g., number of entities, transaction rates) match the user-defined configuration parameters.
- **`h4_reproducibility.py`**: Confirms that running the generator with the same configuration and random seed produces identical datasets, which is crucial for deterministic and repeatable experiments.
