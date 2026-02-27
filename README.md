# Tide

**A customisable dataset generator for benchmarking money laundering detection.**

Tide generates synthetic financial transaction graphs with realistic topology, entity attributes, and injected money laundering patterns based on known typologies. The generated datasets are designed for training and evaluating Graph Neural Networks (GNNs) on fraud detection tasks.

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd AMLbench
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate a dataset
python main.py --config configs/graph.yaml

# Convert to PyTorch Geometric format (optional)
pip install torch torch-geometric numpy pandas
python tools/csv_to_pytorch.py \
    --nodes generated_nodes.csv \
    --edges generated_transactions.csv \
    --output graph.pt
```

## Configuration

Tide uses two configuration files in `configs/`:

| File | Purpose |
|---|---|
| `graph.yaml` | Graph scale, time span, transaction rates, background pattern weights, risk scoring, fraud selection |
| `patterns.yaml` | Pattern-specific parameters (amounts, timing, layering depth, camouflage probability) |

Additional graph configs are provided for different experimental settings:

- `graph_LI.yaml` — Lower illicit ratio (~0.03%)
- `graph_low_homophily.yaml` — Low fraud-class homophily (dispersed fraud)
- `graph_high_homophily.yaml` — High fraud-class homophily (concentrated fraud)

Output format is controlled by `output.yaml` (CSV, GraphML, Gpickle).

## Output

The generator produces three CSV files:

| File | Contents |
|---|---|
| `generated_nodes.csv` | All entities (individuals, businesses, accounts, institutions) with attributes |
| `generated_edges.csv` | All edges (transactions + ownership) |
| `generated_transactions.csv` | Transaction edges only (recommended for GNN training) |
| `generated_patterns.json` | Metadata for each injected fraud pattern |

Use `tools/csv_to_pytorch.py` to convert CSVs to a PyTorch Geometric `.pt` file. See `docs/PYTORCH_GRAPH_FORMAT.md` for the tensor format specification.

## Implemented Patterns

### Fraud patterns

| Pattern | Description |
|---|---|
| Repeated Overseas Transfers | Structured deposits followed by periodic transfers to overseas accounts |
| Rapid Fund Movement | Multiple inflows from overseas, followed by rapid cash withdrawals |
| Front Business Activity | Cash deposits across multiple bank accounts, transferred to overseas businesses |
| U-Turn Transactions | Funds routed through intermediaries before returning to the originator |
| Synchronised Transactions | Coordinated deposits from multiple entities to a common recipient |

### Background patterns

Tide generates 11 types of legitimate background activity, including random payments, salary payments, high-value transactions, and counter-leakage patterns that overlap with fraud signatures to prevent trivial feature-based detection.

## Project Structure

```
main.py                  Entry point — generates the graph and exports CSVs
tools/
  csv_to_pytorch.py      Converts CSVs to PyTorch Geometric format
configs/                 YAML configuration files
tide/                    Core library
  graph_generator.py     Main generation orchestrator
  entities/              Entity definitions (Individual, Business, Account, Institution)
  patterns/              Fraud pattern implementations
  patterns/background/   Legitimate background activity patterns
  outputs/               CSV export
  utils/                 Shared utilities (clustering, amounts, constants)
scripts/                 Visualization and example scripts
experiments/             Validation experiments (pattern, scalability, reproducibility)
```

## Experiments

The `experiments/` directory contains validation scripts:

- **H1** — Pattern validation: structural and temporal correctness
- **H2** — Scalability: performance across different graph sizes
- **H3** — Configuration compliance: output matches config parameters
- **H4** — Reproducibility: deterministic generation with same seed

## Citation

If you use Tide in your research, please cite:

```bibtex
@inproceedings{tide2025,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TODO},
  year      = {2025}
}
```

## License

See [LICENSE](LICENSE).
