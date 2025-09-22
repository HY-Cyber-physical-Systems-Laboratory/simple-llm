"""
TreeSelfAttentionLM — a tiny language model showing how to combine
self‑attention with a tree structure (e.g., dependency parses).

Key ideas
- Compute pairwise tree distances from parent indices per sequence
- Bucketize distances and turn them into a learnable relative bias per head
- Add that bias to the attention logits before softmax
- (Optional) restrict attention to a tree neighborhood by masking large distances

Batch format
- tokens: LongTensor [B, L] (token ids)
- parents: LongTensor [B, L] where parents[b, i] in [-1, 0..L-1]; -1 means root
- attn_mask: BoolTensor [B, L] True for valid tokens (padding=False)

This is a minimal, educational implementation — not optimized for speed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# -------------------------------
# Utilities: tree distance per sequence
# -------------------------------

def pairwise_tree_distance(parent_idx: torch.Tensor) -> torch.Tensor:
    """
    parent_idx: LongTensor [L] with values in [-1, 0..L-1]
    Returns distance matrix [L, L] where dist[i, j] is the length of the
    shortest path between nodes i and j in the undirected version of the tree.
    If nodes are disconnected (shouldn't happen in a tree), distance = large.
    """
    L = parent_idx.shape[0]
    device = parent_idx.device
    # Build adjacency list (undirected)
    adj = [[] for _ in range(L)]
    for child in range(L):
        p = int(parent_idx[child].item())
        if p >= 0 and p < L:
            adj[p].append(child)
            adj[child].append(p)
    # BFS from every node (O(L^2)) — fine for small L
    dmat = torch.full((L, L), 1e6, device=device, dtype=torch.long)
    for s in range(L):
        # BFS
        dist = torch.full((L,), 1e6, device=device, dtype=torch.long)
        q = [s]
        dist[s] = 0
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v in adj[u]:
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        dmat[s] = dist
    # Ensure diagonal zero
    dmat.fill_diagonal_(0)
    return dmat

# -------------------------------
# Relative bias on tree distances
# -------------------------------

@dataclass
class TreeBiasConfig:
    num_heads: int
    num_buckets: int = 8
    max_distance: int = 8  # distances >= max_distance fall into the last bucket

class TreeRelativeBias(nn.Module):
    def __init__(self, cfg: TreeBiasConfig):
        super().__init__()
        self.cfg = cfg
        self.bias = nn.Parameter(torch.zeros(cfg.num_heads, cfg.num_buckets))
        nn.init.normal_(self.bias, std=0.02)

    def _bucketize(self, d: torch.Tensor) -> torch.Tensor:
        """Map integer distances to buckets [0..num_buckets-1].
        Uses linear buckets up to max_distance, then clamps.
        d: LongTensor [...]
        """
        b = torch.clamp(d, max=self.cfg.max_distance)
        # distances in [0..max_distance] map to [0..max_distance],
        # then we collapse the tail into the last bucket.
        b = torch.where(b >= self.cfg.max_distance, torch.full_like(b, self.cfg.max_distance), b)
        # If num_buckets <= max_distance+1, we clamp accordingly.
        if self.cfg.num_buckets <= self.cfg.max_distance + 1:
            b = torch.clamp(b, max=self.cfg.num_buckets - 1)
        else:
            # Leave room for potential custom mapping; for now same as clamp
            b = torch.clamp(b, max=self.cfg.num_buckets - 1)
        return b

    def forward(self, parents: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        parents: LongTensor [B, L]
        pad_mask: BoolTensor [B, L] — True for valid tokens, False for pads (optional)
        Returns bias: FloatTensor [B, H, L, L]
        """
        B, L = parents.shape
        device = parents.device
        H = self.cfg.num_heads
        # Compute per-example distance matrices and stack
        dists = []
        for b in range(B):
            d = pairwise_tree_distance(parents[b])  # [L, L] long
            dists.append(d)
        D = torch.stack(dists, dim=0)  # [B, L, L]
        buckets = self._bucketize(D)  # [B, L, L]
        # Gather bias per head
        # bias table: [H, num_buckets] -> expand to [B, H, L, L]
        bias = self.bias[:, buckets]  # [H, B, L, L]
        bias = bias.permute(1, 0, 2, 3).contiguous()  # [B, H, L, L]
        # Mask pads if provided
        if pad_mask is not None:
            # Prevent attending *from* pads and *to* pads by setting -inf bias
            neg_inf = torch.finfo(torch.float32).min
            valid = pad_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L] for keys
            bias = bias.masked_fill(~valid, neg_inf)
            valid_q = pad_mask.unsqueeze(1).unsqueeze(3)  # [B,1,L,1] for queries
            bias = bias.masked_fill(~valid_q, neg_inf)
        return bias

# -------------------------------
# Multi-head self-attention with tree bias
# -------------------------------

class MHSelfAttentionTree(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, tree_bias: TreeRelativeBias):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.tree_bias = tree_bias

    def forward(self, x: torch.Tensor, parents: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        B, L, D = x.shape
        H = self.n_heads
        qkv = self.qkv(x).view(B, L, 3, H, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, H, L, d_head]
        # scaled dot product
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,L,L]
        # Add tree relative bias
        bias = self.tree_bias(parents, pad_mask)  # [B,H,L,L]
        attn_logits = attn_logits + bias
        # Also mask out pads explicitly for keys if provided (redundant but safe)
        if pad_mask is not None:
            neg_inf = torch.finfo(attn_logits.dtype).min
            key_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            attn_logits = attn_logits.masked_fill(~key_mask, neg_inf)
        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        y = torch.matmul(attn, v)  # [B,H,L,d_head]
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.resid_drop(self.out(y))
        return y

# -------------------------------
# Transformer block
# -------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, tree_bias: TreeRelativeBias):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHSelfAttentionTree(d_model, n_heads, dropout, tree_bias)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, parents, pad_mask=None):
        x = x + self.attn(self.ln1(x), parents, pad_mask)
        x = x + self.mlp(self.ln2(x))
        return x

# -------------------------------
# The tiny LM
# -------------------------------

class TreeSelfAttentionLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4, n_layers: int = 4,
                 d_ff: int = 768, dropout: float = 0.1, max_len: int = 512,
                 num_buckets: int = 8, max_tree_distance: int = 8):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        tb_cfg = TreeBiasConfig(num_heads=n_heads, num_buckets=num_buckets, max_distance=max_tree_distance)
        self.tree_bias = TreeRelativeBias(tb_cfg)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, self.tree_bias) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor, parents: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None):
        """
        tokens: [B, L]
        parents: [B, L]
        attn_mask: [B, L] bool — True where valid tokens (not padding)
        targets: [B, L] (next-token labels) optional
        """
        B, L = tokens.shape
        device = tokens.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.tok(tokens) + self.pos(pos_ids)
        for blk in self.blocks:
            x = blk(x, parents, attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, V = 2, 10, 100
    # Fake batch
    tokens = torch.randint(0, V, (B, L))
    # A toy parent array: a simple chain 0<-1<-2<-... with root at 0
    parents = torch.tensor([[ -1, 0,1,2,3,4,5,6,7,8],
                            [ -1, 0,1,2,3,4,5,6,7,8]])
    # All tokens valid
    mask = torch.ones(B, L, dtype=torch.bool)

    model = TreeSelfAttentionLM(vocab_size=V, d_model=128, n_heads=4, n_layers=2, d_ff=256)
    logits, loss = model(tokens, parents, mask, targets=tokens)  # dummy LM task
    print("logits:", logits.shape, "loss:", float(loss))

    # Single training step
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    opt.zero_grad(); loss.backward(); opt.step()
    print("step done")

# -------------------------------
# Tree-GNN (GAT-style) replacement for MLP
# -------------------------------

class TreeAdjacency:
    """Build adjacency masks from parent indices (undirected)."""
    @staticmethod
    def build(parents: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        parents: LongTensor [B, L]
        pad_mask: BoolTensor [B, L] (True for valid tokens)
        returns adj: BoolTensor [B, L, L] where adj[b,i,j]=True if edge exists
        """
        B, L = parents.shape
        device = parents.device
        adj = torch.zeros(B, L, L, dtype=torch.bool, device=device)
        for b in range(B):
            for child in range(L):
                p = int(parents[b, child].item())
                if 0 <= p < L:
                    adj[b, child, p] = True
                    adj[b, p, child] = True
        # self loops (common in GAT)
        eye = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0).expand(B, L, L)
        adj = adj | eye
        if pad_mask is not None:
            # remove any edges to/from pads
            valid = pad_mask.bool()
            for b in range(B):
                v = valid[b]
                adj[b, ~v, :] = False
                adj[b, :, ~v] = False
        return adj

class TreeGATLayer(nn.Module):
    """
    Lightweight multi-head graph attention over a tree (adjacency mask).
    This is a masked attention restricted to neighborhood edges (i,j in E).
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, parents: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm
        h = self.ln(x)
        B, L, D = h.shape
        H = self.n_heads
        q = self.Wq(h).view(B, L, H, self.d_head).transpose(1, 2)  # [B,H,L,d]
        k = self.Wk(h).view(B, L, H, self.d_head).transpose(1, 2)
        v = self.Wv(h).view(B, L, H, self.d_head).transpose(1, 2)
        # Local attention logits
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,L,L]
        # Mask to tree edges only
        adj = TreeAdjacency.build(parents, pad_mask)  # [B,L,L]
        mask = adj.unsqueeze(1)  # [B,1,L,L]
        neg_inf = torch.finfo(logits.dtype).min
        logits = torch.where(mask, logits, torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype))
        if pad_mask is not None:
            key_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
            logits = logits.masked_fill(~key_mask, neg_inf)
        attn = torch.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)
        y = torch.matmul(attn, v)  # [B,H,L,d]
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.resid_drop(self.out(y))
        return x + y

class TreeGNNFeedForward(nn.Module):
    """Stack of TreeGAT layers to replace vanilla MLP."""
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([TreeGATLayer(d_model, n_heads, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, parents: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for lyr in self.layers:
            x = lyr(x, parents, pad_mask)
        return x

class TransformerBlockTreeGNN(nn.Module):
    """
    Transformer block where the FeedForward is replaced by a Tree-GNN (GAT-style).
    """
    def __init__(self, d_model: int, n_heads: int, d_ff_dummy: int, dropout: float, tree_bias: TreeRelativeBias,
                 tree_gnn_heads: Optional[int] = None, tree_gnn_layers: int = 2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHSelfAttentionTree(d_model, n_heads, dropout, tree_bias)
        self.tree_gnn = TreeGNNFeedForward(d_model, n_heads=tree_gnn_heads or n_heads, n_layers=tree_gnn_layers, dropout=dropout)

    def forward(self, x, parents, pad_mask=None):
        x = x + self.attn(self.ln1(x), parents, pad_mask)
        x = self.tree_gnn(x, parents, pad_mask)
        return x

class TreeSelfAttentionLM_GAT(nn.Module):
    """
    Language model using (1) self-attention with tree-relative bias and
    (2) a Tree-GNN (GAT-style) in place of the vanilla MLP.
    """
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4, n_layers: int = 4,
                 dropout: float = 0.1, max_len: int = 512, num_buckets: int = 8, max_tree_distance: int = 8,
                 tree_gnn_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.tree_gnn_layers = tree_gnn_layers
        
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        tb_cfg = TreeBiasConfig(num_heads=n_heads, num_buckets=num_buckets, max_distance=max_tree_distance)
        self.tree_bias = TreeRelativeBias(tb_cfg)
        self.blocks = nn.ModuleList([
            TransformerBlockTreeGNN(d_model, n_heads, d_ff_dummy=d_model*4, dropout=dropout, tree_bias=self.tree_bias,
                                     tree_gnn_heads=n_heads, tree_gnn_layers=tree_gnn_layers)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor, parents: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None):
        B, L = tokens.shape
        device = tokens.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.tok(tokens) + self.pos(pos_ids)
        for blk in self.blocks:
            x = blk(x, parents, attn_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss
