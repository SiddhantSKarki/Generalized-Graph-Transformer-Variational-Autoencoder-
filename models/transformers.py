import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class InputEmbedding(nn.Module):
    def __init__(self, d_node_in, d_pos, d_hidden):
        super().__init__()
        # Linear projections for node, and positional features
        self.node_proj = nn.Linear(d_node_in, d_hidden)
        self.pos_proj  = nn.Linear(d_pos, d_hidden)

    def forward(self, node_feats, pos_enc):
        """
        node_feats: [B, N, d_node_in]
        pos_enc:    [B, N, d_pos] (e.g., Laplacian eigenvectors)
        """
        B, N, input_dim = node_feats.size()
        # A0 αi + a0
        h_hat0 = self.node_proj(node_feats)              
        lam0 = self.pos_proj(pos_enc) # C0 λi + c0

        h0 = h_hat0 + lam0        
        # (B, N, d_hidden)
        return h0

class GraphTransformerHead(nn.Module):
    def __init__(self, d_input, d_k):
        super().__init__()
        self.d_k = d_k

        # Q, K, V projections (one per head)
        self.W_Q = nn.Linear(d_input, self.d_k)
        self.W_K = nn.Linear(d_input, self.d_k)
        self.W_V = nn.Linear(d_input, self.d_k)

    def forward(self, h, adj):
        """
        h:   [B, N, d_input] node embeddings
        adj: [B, N, N] adjacency matrix (0/1)
        """
        B, N, input_dim = h.shape

        Q = self.W_Q(h) # [B, N, d_k]
        K = self.W_K(h) # [B, N, d_k]
        V = self.W_V(h) # [B, N, d_k]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # add self-loops
        eye = torch.eye(N, device=adj.device).unsqueeze(0)
        adj_mask = torch.clamp(adj + eye, max=1.0)

        # mask non-edges
        # scores = scores.masked_fill(adj_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        h_head = torch.matmul(attn, V)
        return h_head, attn

class GraphTransformerLayer(nn.Module):
    def __init__(self, d_hidden, n_heads=4, dropout=0.1, norm_type="layer"):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        
        assert d_hidden % n_heads == 0, "d_hidden must be divisible by n_heads"

        # heads
        # each head outputs are: (B, N, d_hidden // n_heads)
        self.heads = nn.ModuleList([
            GraphTransformerHead(d_hidden, d_hidden // n_heads)
            for _ in range(n_heads)
        ])

        # inputs: (B, N, (d_hidden * n_heads) // n_heads) = (B, N, d_hidden)
        # output projection after concatenation
        self.W_O = nn.Linear(d_hidden, d_hidden)

        # ff_block
        self.ffn = nn.Sequential(
            nn.Linear(d_hidden, 2 * d_hidden),
            nn.ReLU(),
            nn.Linear(2 * d_hidden, d_hidden)
        )


        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)

        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj):
        """
        h:   [B, N, d_hidden] node embeddings
        adj: [B, N, N] adjacency matrix (0/1)
        """
        B, N, input_dim = h.shape
        heads = []
        attentions = []

        # ---- multi-head attention ----
        for head in self.heads:
            h_head, attn = head(h, adj) # [B, N, d_hidden // n_heads]
            heads.append(h_head)
            attentions.append(attn)

        # concatenate all head outputs
        # [B, N, d_hidden * n_heads // n_heads] = [B, N, d_hidden]
        self.attentions = torch.stack(attentions, dim=1)  # [B, n_heads, N, N]
        h_concat = torch.cat(heads, dim=-1)  # [B, N, d_hidden]
        h_attn = self.W_O(h_concat) # (B, N, d_hidden)

        # ---- Add & Norm ----
        h_norm = self.norm1(h + h_attn) # (B, N, d_hidden)

        # ---- Feed Forward ----
        h_ffn = self.ffn(h_norm) # (B, N, d_hidden)
        h_out = self.norm2(h_norm + h_ffn) # (B, N, d_hidden)

        return h_out # (B, N, d_hidden)
    

# paper link: https://arxiv.org/pdf/1911.06455
class GraphTransformerNetwork(nn.Module):
    def __init__(self, d_node_in, d_pos, d_hidden, n_layers=4, n_heads=4):
        super().__init__()
        self.embed = InputEmbedding(d_node_in, d_pos, d_hidden)
        self.layers = nn.ModuleList([
            GraphTransformerLayer(d_hidden, n_heads=n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, node_feats, pos_enc, adj):
        """
        Forward pass of the GraphTransformer.

        node_feats: [B, N, d_node_in] input node features
        pos_enc:    [B, N, d_pos] positional encodings
        adj:        [B, N, N] adjacency matrix (0/1)
        """
        h = self.embed(node_feats, pos_enc) # (B, N, d_hidden)

        for layer in self.layers:
            h = layer(h, adj)
        return h
    
def dot_product_decode(Z):
    # clamp for numerical stability
    return torch.sigmoid(torch.matmul(Z, Z.transpose(-1, -2)).clamp(-10, 10))


class GTN_VAE(nn.Module):
    def __init__(self, input_dim, pos_dim, hidden_dim, n_layers=4, n_heads=4):
        super(GTN_VAE, self).__init__()
        self.transformer_encoder = GraphTransformerNetwork(d_node_in=input_dim, d_pos=pos_dim, d_hidden=hidden_dim, n_layers=n_layers, n_heads=n_heads)

        self.mlp_mean = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_logvar = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, x, pos_enc, adj):
        """
        x: (B, N, input_dim) node features
        pos_enc: (B, N, hidden1_dim) positional encodings (e.g
        Laplacian eigenvectors)
        adj: (B, N, N) adjacency matrix (0/1)
        return: sampled latent vectors (B, N, hidden2_dim)
        """
        # x: (B, N, input_dim)
        # pos_enc: (B, N, hidden1_dim)
        # adj: (B, N, N)
        encoded = self.transformer_encoder(x, pos_enc, adj)
        return encoded

    def reparameterize(self, mu, logvar):
        # mu, logvar: (B, N, hidden2_dim)
        gaussian_noise = torch.randn_like(logvar)
        # (B, N, hidden2_dim)
        sampled_z = mu + gaussian_noise * torch.exp(0.5 * logvar)  # reparameterization trick
        # (B, N, hidden2_dim)
        return sampled_z


    def forward(self, x, pos_enc, adj):
        encoded = self.encode(x, pos_enc, adj)
        # in: (B, N, hidden1_dim)
        # out: (B, N, hidden2_dim)
        self.z_mean = self.mlp_mean(encoded)
        # in: (B, N, hidden1_dim)
        # out: (B, N, hidden2_dim)
        self.z_logvar = self.mlp_logvar(encoded)

        z = self.reparameterize(self.z_mean, self.z_logvar)
        # (B, N, hidden2_dim)

        adj_recon = dot_product_decode(z)
        # (B, N, N)
        # problem: diagonal entries are always 1, so mask them out now
        adj_recon = adj_recon.masked_fill(torch.eye(adj_recon.size(1)).bool().to(adj_recon.device), 0.0)
        return adj_recon