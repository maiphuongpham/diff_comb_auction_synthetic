# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CustomNNConv(nn.Module):
#     def __init__(self, in_channels, out_channels, edge_dim, hidden=32, aggr='mean'):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.aggr = aggr
        
#         # edge_mlp generates a [out × in] matrix for each edge
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(edge_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, out_channels * in_channels)
#         )

#         # Initialize weights (you can change to xavier_normal_ or others)
#         for layer in self.edge_mlp:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 nn.init.zeros_(layer.bias)

#     def forward(self, x, edge_index, edge_attr):
#         """
#         x: [N, F_in] node features
#         edge_index: [2, E] source-to-target edges
#         edge_attr: [E, D] edge features
#         """
#         N = x.size(0)
#         E = edge_index.size(1)

#         row, col = edge_index  # row = target, col = source

#         # Compute weight matrices: [E, out_channels, in_channels]
#         edge_weights = self.edge_mlp(edge_attr)
#         edge_weights = edge_weights.view(-1, self.out_channels, self.in_channels)

#         # Source features: [E, in_channels]
#         x_source = x[col]  # source nodes

#         if torch.isnan(edge_attr).any():
#             raise ValueError("edge_attr has NaNs")

#         if torch.isnan(x_source).any():
#             raise ValueError("x_source has NaNs")

#         if torch.isnan(edge_weights).any():
#             raise ValueError("edge_weights has NaNs")

#         # Weighted message: [E, out_channels]
#         messages = torch.bmm(edge_weights, x_source.unsqueeze(-1)).squeeze(-1)

#         if torch.isnan(messages).any():
#             raise ValueError("messages have NaNs after bmm")

#         # Aggregate to each target node
#         out = torch.zeros(N, self.out_channels, device=x.device)
#         out.index_add_(0, row, messages)  # sum aggregation

#         if torch.isnan(out).any():
#             raise ValueError("out has NaNs after aggregation")

#         if self.aggr == 'mean':
#             deg = torch.bincount(row, minlength=N).clamp(min=1).unsqueeze(-1)
#             out = out / deg

#         return x + out

import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, hidden=64, aggr='mean', dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.dropout = nn.Dropout(dropout)

        # Edge-conditioned weight generator
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_channels * in_channels)
        )
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Post-aggregation transformation (helps reintroduce nonlinearity & diversity)
        self.post_mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        for layer in self.post_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Optional projection if dimensions differ
        self.res_proj = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, edge_index, edge_attr):
        N = x.size(0)
        row, col = edge_index

        # Edge-conditioned weight matrices
        edge_weights = self.edge_mlp(edge_attr)  # [E, out*in]
        edge_weights = edge_weights.view(-1, self.out_channels, self.in_channels)

        x_source = x[col]  # [E, in]
        messages = torch.bmm(edge_weights, x_source.unsqueeze(-1)).squeeze(-1)  # [E, out]

        # Aggregate
        out = torch.zeros(N, self.out_channels, device=x.device)
        out.index_add_(0, row, messages)
        if self.aggr == 'mean':
            deg = torch.bincount(row, minlength=N).clamp(min=1).unsqueeze(-1)
            out = out / deg

        # Post-aggregation MLP
        out = self.post_mlp(self.dropout(out))

        # Residual connection (with projection if needed)
        out = F.relu(out + self.res_proj(x))

        return out


class CustomGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0.0, aggr='mean'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.aggr = aggr
        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.attn = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn)

        final_out_dim = heads * out_channels if concat else out_channels
        self.res_proj = (
            nn.Linear(in_channels, final_out_dim)
            if in_channels != final_out_dim else nn.Identity()
        )

    def forward(self, x, edge_index):
        N = x.size(0)
        H = self.heads
        D = self.out_channels

        x_proj = self.linear(x).view(N, H, D)  # [N, H, D]
        row, col = edge_index

        x_i = x_proj[row]
        x_j = x_proj[col]

        attn_input = torch.cat([x_i, x_j], dim=-1)
        alpha = F.leaky_relu((attn_input * self.attn).sum(dim=-1))
        # alpha = alpha - alpha.max(dim=0, keepdim=True)[0] # for numerical stability
        alpha = torch.clamp(alpha, max=64.0).nan_to_num(0.0)  # avoid huge exp
        alpha = torch.exp(alpha)
        # if alpha is nan or inf, raise an error
        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            raise ValueError("alpha has NaNs or Infs")
        alpha_sum = torch.zeros(N, H, device=x.device).index_add_(0, row, alpha)
        alpha = alpha / (alpha_sum[row] + 1e-9)

        messages = x_j * alpha.unsqueeze(-1)
        out = torch.zeros(N, H, D, device=x.device)
        out.index_add_(0, row, messages)

        if not self.concat:
            out = out.mean(dim=1)  # [N, D]
        else:
            out = out.view(N, H * D)

        out = self.dropout(out)
        return F.elu(out + self.res_proj(x))
