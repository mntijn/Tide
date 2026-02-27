# PyTorch Graph Format Documentation

## Overview

The `generated_graph.pt` file now contains a properly formatted PyTorch dictionary with **all edge attributes** for both transaction and ownership edges, suitable for training Graph Neural Networks including Graph Attention Networks (GATs).

## Problem Solved

**Previous Issue:** The old implementation only captured edge attributes from the first edge sampled, which meant only ownership edge attributes (`edge_type`, `ownership_percentage`, `ownership_start_date`) were visible when an ownership edge was sampled first.

**Solution:** The new implementation properly separates edge types and includes all attributes for both transaction and ownership edges in separate, clearly labeled tensors.

## Data Structure

The `.pt` file contains a dictionary with the following structure:

```python
{
    # Node Information
    'num_nodes': int,                           # Total number of nodes
    'node_ids': List[str],                      # Original node IDs (for mapping back)
    'node_features': torch.Tensor,              # Shape: [num_nodes, num_node_features]
    'node_feature_names': List[str],            # Names of node features

    # Transaction Edges
    'transaction_edge_index': torch.Tensor,     # Shape: [2, num_transaction_edges]
    'transaction_edge_attr': Dict[str, torch.Tensor],  # Dict of feature tensors
    'transaction_edge_attr_names': List[str],   # Names of transaction edge features

    # Ownership Edges
    'ownership_edge_index': torch.Tensor,       # Shape: [2, num_ownership_edges]
    'ownership_edge_attr': Dict[str, torch.Tensor],    # Dict of feature tensors
    'ownership_edge_attr_names': List[str],     # Names of ownership edge features

    # Combined (for models that don't distinguish edge types)
    'edge_index': torch.Tensor,                 # Shape: [2, total_edges]

    # Metadata
    'metadata': {
        'num_transaction_edges': int,
        'num_ownership_edges': int,
        'total_edges': int
    }
}
```

## Node Features

The following node features are extracted and converted to tensors:

- `risk_score` (float)
- `is_fraudulent` (bool → float)
- `incorporation_year` (int → float)
- `number_of_employees` (int → float)
- `is_high_risk_category` (bool → float)
- `is_high_risk_country` (bool → float)

Missing values are filled with 0.0.

## Transaction Edge Attributes

Each transaction edge includes:

- `amount` (float): Transaction amount
- `is_fraudulent` (bool → float): Whether the transaction is fraudulent
- `timestamp` (datetime → float): Unix timestamp
- `time_since_previous` (timedelta → float): Seconds since previous transaction

## Ownership Edge Attributes

Each ownership edge includes:

- `ownership_percentage` (float): Percentage of ownership
- `ownership_start_date` (date → float): Date as ordinal number

## Usage Examples

### Loading the Graph

```python
import torch

# Load the graph data
data = torch.load('generated_graph.pt')

print(f"Nodes: {data['num_nodes']}")
print(f"Transaction edges: {data['metadata']['num_transaction_edges']}")
print(f"Ownership edges: {data['metadata']['num_ownership_edges']}")
```

### Using with PyTorch Geometric

```python
from torch_geometric.data import Data

# Create a PyG Data object for transaction edges
if data['metadata']['num_transaction_edges'] > 0:
    # Combine transaction edge features into a single tensor
    edge_attr_list = []
    for attr_name in data['transaction_edge_attr_names']:
        edge_attr_list.append(
            data['transaction_edge_attr'][attr_name].unsqueeze(1)
        )
    transaction_edge_features = torch.cat(edge_attr_list, dim=1)

    pyg_data = Data(
        x=data['node_features'],
        edge_index=data['transaction_edge_index'],
        edge_attr=transaction_edge_features,
        num_nodes=data['num_nodes']
    )
```

### Using with Graph Attention Networks

```python
from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F

class AMLFraudDetector(nn.Module):
    def __init__(self, num_node_features, num_edge_features):
        super().__init__()
        self.conv1 = GATConv(
            num_node_features,
            64,
            heads=4,
            edge_dim=num_edge_features
        )
        self.conv2 = GATConv(
            64 * 4,
            2,  # Binary classification: fraud or not
            heads=1,
            edge_dim=num_edge_features
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

# Initialize model
model = AMLFraudDetector(
    num_node_features=pyg_data.x.shape[1],
    num_edge_features=pyg_data.edge_attr.shape[1]
)

# Train/inference
output = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
```

## Working with Heterogeneous Graphs

If you want to use both transaction and ownership edges simultaneously (heterogeneous graph):

```python
from torch_geometric.data import HeteroData

hetero_data = HeteroData()

# Add node features
hetero_data['entity'].x = data['node_features']
hetero_data['entity'].num_nodes = data['num_nodes']

# Add transaction edges
if data['metadata']['num_transaction_edges'] > 0:
    trans_edge_attr = torch.cat([
        data['transaction_edge_attr'][name].unsqueeze(1)
        for name in data['transaction_edge_attr_names']
    ], dim=1)

    hetero_data['entity', 'transacts', 'entity'].edge_index = \
        data['transaction_edge_index']
    hetero_data['entity', 'transacts', 'entity'].edge_attr = \
        trans_edge_attr

# Add ownership edges
if data['metadata']['num_ownership_edges'] > 0:
    own_edge_attr = torch.cat([
        data['ownership_edge_attr'][name].unsqueeze(1)
        for name in data['ownership_edge_attr_names']
    ], dim=1)

    hetero_data['entity', 'owns', 'entity'].edge_index = \
        data['ownership_edge_index']
    hetero_data['entity', 'owns', 'entity'].edge_attr = \
        own_edge_attr
```

## Generating the Graph

To generate a new graph with PyTorch export:

1. Ensure `PyTorch: True` in `configs/output.yaml`
2. Run: `python main.py --config configs/graph.yaml`
3. The file will be saved as `generated_graph.pt`

## Example Script

See `load_pytorch_graph_example.py` for a complete working example that demonstrates:
- Loading and inspecting the graph
- Converting to PyTorch Geometric format
- Building a simple GAT model
- Running inference

Run it with:
```bash
python load_pytorch_graph_example.py generated_graph.pt
```

## Edge Index Format

The edge indices follow PyTorch Geometric convention:
- Shape: `[2, num_edges]`
- First row: source node indices
- Second row: destination node indices
- Node indices are 0-based integers mapping to `node_ids` list

Example:
```python
edge_index = torch.tensor([[0, 1, 2],    # source nodes
                          [1, 2, 0]])   # destination nodes
# This represents edges: 0→1, 1→2, 2→0
```

## Troubleshooting

**Q: I'm getting tensor size mismatch errors**
A: Make sure to use the correct edge features for the edge type you're working with. Transaction and ownership edges have different features.

**Q: How do I handle missing values?**
A: Missing values are already converted to 0.0 in the node and edge features.

**Q: Can I use this with DGL instead of PyTorch Geometric?**
A: Yes! You can convert the edge_index and features to DGL format:
```python
import dgl
g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
g.ndata['feat'] = node_features
g.edata['feat'] = edge_features
```

## Changes Made to `main.py`

1. Added `convert_graph_to_pytorch()` function that:
   - Separates transaction and ownership edges
   - Extracts all relevant attributes for each edge type
   - Converts all data to proper PyTorch tensors
   - Provides proper metadata for easy inspection

2. Updated the PyTorch export section to:
   - Use the new conversion function
   - Print detailed summary of exported data
   - Show all edge attribute names for both edge types

3. Added imports: `numpy` and `EdgeType` from datastructures

