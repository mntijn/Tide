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

Specify the number of individuals and institutions for the graph. Businesses will be generated based on individual attributes, but you can set a minimum number of businesses. Tide will create additional businesses as needed to meet this minimum.

## Patterns

Not implemented correctly yet.
