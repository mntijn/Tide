# Tide: A Customisable Dataset Generator for Benchmarking Money Laundering Detection

## Config

There are 2 configs:
1. graph.yaml
2. patterns.yaml

Graphs.yaml allows for more general graph settings. Patterns.yaml allows for pattern-specific settings.

## Running the generator
To run, just create a venv, run `pip install -r requirements.txt`, and then `python main.py`

## Docs

## Entity creation

For the size of the graph, at least specify individuals and institutions. Individuals will be generated and in accordance with their attributes, businesses will be created. If you want you can overwrite this by specifying the minimum amount of businesses you want in the dataset. Tide wil create new businesses that fill up the gap between the already generated businesses and your specified amount.

## Patterns

Not implemented correctly yet.
