"""
Example script demonstrating how to load and use the PyTorch graph data.

This script shows how to:
1. Load the generated_graph.pt file
2. Access node and edge features
3. Use the data with Graph Neural Networks (e.g., PyTorch Geometric)
"""

import torch


def load_and_inspect_graph(filepath: str = "generated_graph.pt"):
    """
    Load and inspect the PyTorch graph data.

    Args:
        filepath: Path to the .pt file
    """
    print(f"Loading graph from: {filepath}")
    data = torch.load(filepath)

    print("\n=== Graph Structure ===")
    print(f"Number of nodes: {data['num_nodes']}")
    print(f"Total edges: {data['metadata']['total_edges']}")
    print(
        f"  - Transaction edges: {data['metadata']['num_transaction_edges']}")
    print(f"  - Ownership edges: {data['metadata']['num_ownership_edges']}")

    print("\n=== Node Features ===")
    print(f"Node features shape: {data['node_features'].shape}")
    print(f"Feature names: {data['node_feature_names']}")
    print(f"Sample node features (first 3 nodes):")
    print(data['node_features'][:3])

    print("\n=== Transaction Edges ===")
    if data['metadata']['num_transaction_edges'] > 0:
        print(
            f"Transaction edge index shape: {data['transaction_edge_index'].shape}")
        print(
            f"Transaction edge attributes: {data['transaction_edge_attr_names']}")
        print(f"Sample transaction edge attributes (first 3 edges):")
        for attr_name in data['transaction_edge_attr_names']:
            attr_values = data['transaction_edge_attr'][attr_name]
            print(
                f"  {attr_name}: shape={attr_values.shape}, sample={attr_values[:3]}")
    else:
        print("No transaction edges in graph")

    print("\n=== Ownership Edges ===")
    if data['metadata']['num_ownership_edges'] > 0:
        print(
            f"Ownership edge index shape: {data['ownership_edge_index'].shape}")
        print(
            f"Ownership edge attributes: {data['ownership_edge_attr_names']}")
        print(f"Sample ownership edge attributes (first 3 edges):")
        for attr_name in data['ownership_edge_attr_names']:
            attr_values = data['ownership_edge_attr'][attr_name]
            print(
                f"  {attr_name}: shape={attr_values.shape}, sample={attr_values[:3]}")
    else:
        print("No ownership edges in graph")

    print("\n=== Combined Edge Index ===")
    print(f"Combined edge index shape: {data['edge_index'].shape}")
    print(f"  (Includes all edges regardless of type)")

    return data


def convert_to_pytorch_geometric(data):
    """
    Example of how to convert the data to PyTorch Geometric format.

    Note: This requires torch_geometric to be installed.
    Install with: pip install torch-geometric

    Args:
        data: The loaded graph data dictionary

    Returns:
        PyTorch Geometric Data object for transaction edges
    """
    try:
        from torch_geometric.data import Data

        # Create a PyG Data object for transaction edges
        # You can create separate Data objects for each edge type if needed

        # For transaction edges:
        if data['metadata']['num_transaction_edges'] > 0:
            # Combine transaction edge features into a single tensor
            edge_attr_list = []
            for attr_name in data['transaction_edge_attr_names']:
                edge_attr_list.append(
                    data['transaction_edge_attr'][attr_name].unsqueeze(1))
            transaction_edge_features = torch.cat(edge_attr_list, dim=1)

            pyg_data = Data(
                x=data['node_features'],
                edge_index=data['transaction_edge_index'],
                edge_attr=transaction_edge_features,
                num_nodes=data['num_nodes']
            )

            print("\n=== PyTorch Geometric Data Object ===")
            print(f"Node features: {pyg_data.x.shape}")
            print(f"Edge index: {pyg_data.edge_index.shape}")
            print(f"Edge features: {pyg_data.edge_attr.shape}")
            print(f"Number of nodes: {pyg_data.num_nodes}")

            return pyg_data
        else:
            print("\nNo transaction edges to convert to PyG format")
            return None

    except ImportError:
        print(
            "\nPyTorch Geometric not installed. Install with: pip install torch-geometric")
        return None


def example_gat_usage(pyg_data):
    """
    Example of how to use the data with a Graph Attention Network.

    Note: This requires torch_geometric to be installed.

    Args:
        pyg_data: PyTorch Geometric Data object
    """
    try:
        from torch_geometric.nn import GATConv
        import torch.nn as nn
        import torch.nn.functional as F

        class SimpleGAT(nn.Module):
            def __init__(self, num_node_features, num_edge_features, hidden_dim=64, num_classes=2):
                super(SimpleGAT, self).__init__()
                # First GAT layer
                self.conv1 = GATConv(
                    num_node_features, hidden_dim, heads=4, edge_dim=num_edge_features)
                # Second GAT layer
                self.conv2 = GATConv(
                    hidden_dim * 4, num_classes, heads=1, edge_dim=num_edge_features)

            def forward(self, x, edge_index, edge_attr):
                x = self.conv1(x, edge_index, edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index, edge_attr)
                return F.log_softmax(x, dim=1)

        # Initialize model
        num_node_features = pyg_data.x.shape[1]
        num_edge_features = pyg_data.edge_attr.shape[1]

        model = SimpleGAT(num_node_features, num_edge_features)

        print("\n=== Example GAT Model ===")
        print(f"Input node features: {num_node_features}")
        print(f"Input edge features: {num_edge_features}")
        print(f"\nModel architecture:")
        print(model)

        # Example forward pass
        model.eval()
        with torch.no_grad():
            output = model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
            print(f"\nOutput shape: {output.shape}")
            print(
                f"Output represents log probabilities for {output.shape[1]} classes")

        return model

    except ImportError:
        print("\nPyTorch Geometric not installed. Cannot demonstrate GAT usage.")
        return None


if __name__ == "__main__":
    import sys

    # Load and inspect the graph
    filepath = sys.argv[1] if len(sys.argv) > 1 else "generated_graph.pt"

    try:
        data = load_and_inspect_graph(filepath)

        # Convert to PyTorch Geometric format
        pyg_data = convert_to_pytorch_geometric(data)

        # Example GAT usage
        if pyg_data is not None:
            example_gat_usage(pyg_data)

    except FileNotFoundError:
        print(f"\nError: File '{filepath}' not found.")
        print("Please generate the graph first by running:")
        print("  python main.py --config configs/graph.yaml")
        print("\nMake sure PyTorch export is enabled in configs/output.yaml")
    except Exception as e:
        print(f"\nError loading graph: {e}")
        import traceback
        traceback.print_exc()
