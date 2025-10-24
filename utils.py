import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.utils as utils
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import seaborn as sns

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj, train_test_split_edges, get_laplacian

def prepare_gtnvae_data(root, dataset_name='Cora', pos_dim=16):

    dataset = Planetoid(root=root, name=dataset_name)
    data = dataset[0]
    data_split = train_test_split_edges(data)
    N = data.num_nodes

    # Dense node features
    x = data.x.unsqueeze(0)

    # Dense adjacencies
    def to_dense_binary(edge_index):
        adj = to_dense_adj(edge_index, max_num_nodes=N)[0]
        return (adj > 0).float()

    train_adj = to_dense_binary(data_split.train_pos_edge_index).unsqueeze(0)
    val_adj   = to_dense_binary(data_split.val_pos_edge_index).unsqueeze(0)
    test_adj  = to_dense_binary(data_split.test_pos_edge_index).unsqueeze(0)

    # Positional encodings from train edges
    edge_index, edge_weight = get_laplacian(data_split.train_pos_edge_index, normalization='sym', num_nodes=N)
    L = torch.sparse_coo_tensor(edge_index, edge_weight, (N, N)).to_dense()
    eigval, eigvec = torch.linalg.eigh(L)
    pos_enc = eigvec[:, 1:pos_dim+1].unsqueeze(0)

    return x, pos_enc, train_adj, val_adj, test_adj, data.y

def balanced_edge_indices(adj_true, num_samples=5000):
    """
    Returns equal numbers of positive and negative edge indices.
    Ensures class balance and guards against small graphs.
    """
    # Flatten and find all positive/negative indices
    pos_idx = (adj_true.view(-1) == 1).nonzero(as_tuple=False).view(-1)
    neg_idx = (adj_true.view(-1) == 0).nonzero(as_tuple=False).view(-1)

    # How many to sample per class
    num_pos = min(len(pos_idx), num_samples // 2)
    num_neg = num_pos  # strict equality
    if num_pos == 0 or num_neg == 0:
        raise ValueError("No positive or negative edges to sample from.")

    # Random balanced sample
    pos_idx = pos_idx[torch.randperm(len(pos_idx))[:num_pos]]
    neg_idx = neg_idx[torch.randperm(len(neg_idx))[:num_neg]]
    idx = torch.cat([pos_idx, neg_idx])

    return idx, num_pos, num_neg


def sampled_vae_loss(adj_pred, adj_true, mu, logvar, beta, num_samples=5000):
    """
    Balanced random BCE + KL loss for VGAE-style models.
    """
    idx, num_pos, num_neg = balanced_edge_indices(adj_true, num_samples)
    y_true = adj_true.view(-1)[idx]
    y_pred = adj_pred.view(-1)[idx]

    # BCE over balanced sample
    bce = F.binary_cross_entropy(y_pred, y_true)
    kl  = -beta * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + kl, bce, kl


def sampled_metrics(adj_pred, adj_true, num_samples=5000):
    """
    Computes ROC-AUC, AP, and Accuracy on balanced edge samples.
    """
    idx, num_pos, num_neg = balanced_edge_indices(adj_true, num_samples)
    y_true = adj_true.view(-1)[idx].cpu().numpy()
    y_pred = adj_pred.view(-1)[idx].detach().cpu().numpy()

    roc = roc_auc_score(y_true, y_pred)
    ap  = average_precision_score(y_true, y_pred)
    acc = ((torch.tensor(y_pred) > 0.5).float().numpy() == y_true).mean()
    return roc, ap, acc

def model_in_out(model_comp, inputs):
    output = model_comp(*inputs)
    output_mean = output.mean()
    output_std = output.std()
    print(f"output: {output}\noutput_mean:{output_mean}\noutput_std:{output_std}")
    return output

def model_debug_auto(model_comp, inputs, verbose=True):
    # Run forward pass
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    
    output = model_comp(*inputs)
    
    def summarize_tensor(name, tensor):
        if isinstance(tensor, torch.Tensor):
            return {
                "name": name,
                "shape": tuple(tensor.shape),
                "dtype": tensor.dtype,
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item()
            }
        elif isinstance(tensor, (list, tuple)):
            return [summarize_tensor(f"{name}[{i}]", t) for i, t in enumerate(tensor)]
        elif isinstance(tensor, dict):
            return {k: summarize_tensor(f"{name}.{k}", v) for k, v in tensor.items()}
        else:
            return {name: str(type(tensor))}

    # Summarize input(s) and output(s)
    input_summary = [summarize_tensor(f"input[{i}]", inp) for i, inp in enumerate(inputs)]
    output_summary = summarize_tensor("output", output)

    if verbose:
        print("=== INPUT SUMMARY ===")
        for s in input_summary:
            print(s)
        print("\n=== OUTPUT SUMMARY ===")
        print(output_summary)
    
    return output, input_summary, output_summary