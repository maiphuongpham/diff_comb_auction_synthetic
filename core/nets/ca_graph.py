import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from core.nets.ca_net import CANet
from core.utils import get_bundle_bid
from layers.blocks import CrossAttention
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from layers.graphconv import CustomNNConv, CustomGATConv


class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)

        self.init_weights = nn.init.xavier_uniform_
        self.init_weights(self.proj.weight)
        self.init_weights(self.gate.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        return self.proj(x) * torch.sigmoid(self.gate(x))

def safe_mean(tensor, dim, keepdim=False):
    return torch.nan_to_num(tensor.mean(dim=dim, keepdim=keepdim), nan=0.0)

class CAGCN(CANet):
    def __init__(self, model_config, device):
        super().__init__(model_config, device)

    def init(self):
        self.graph_type = "conflict"  # "bipartite", "tripartite" or "conflict"
        self.alloc_layers = nn.ModuleList([])
        self.pay_layers = nn.ModuleList([])

        self.create_constants()
        self.create_allocation_layers()
        self.create_payment_layers()

    def create_constants(self):
        init = None
        if self.config.net.init == "gu":
            init = nn.init.xavier_uniform_
        elif self.config.net.init == "gn":
            init = nn.init.xavier_normal_
        self.init_weights = init

        activation = nn.ReLU()
        if self.config.net.activation == "tanh":
            activation = nn.Tanh()
        self.activation = activation

        self.num_agents = int(self.config.num_agents)
        self.num_items = int(self.config.num_items)
        self.num_bundles = int(self.config.num_bundles)
        self.incident_matrix = self.gen_incident_matrix().to(self.device)
        self.get_bundle_bid = get_bundle_bid

        self.num_a_layers = self.config.net.num_a_layers
        self.num_p_layers = self.config.net.num_p_layers
        self.num_a_hidden_units = self.config.net.num_a_hidden_units
        self.num_p_hidden_units = self.config.net.num_p_hidden_units

        # Raw input_dim for agent/bundle embeddings
        self.embedding_dim = 1 + self.num_items

        # Final input_dim for GCN layer depends on graph type
        if self.graph_type == "conflict":
            # Conflict node = [bid, agent_emb (D-1), bundle_emb (D-1)] = 1 + 2*(D-1)
            self.input_dim = 1 + 2 * (self.embedding_dim - 1)
        else:
            self.input_dim = self.embedding_dim

        self.num_in = self.num_agents * self.num_bundles
        self.ln = self.config.net.layer_norm
        self.bn = self.config.net.batch_norm

        self.agent_embedding = nn.Parameter(torch.randn(self.num_agents, self.embedding_dim))
        self.bundle_embedding = nn.Parameter(torch.randn(self.num_bundles, self.embedding_dim))
        self.item_embedding = nn.Parameter(torch.randn(self.num_items, self.embedding_dim))



    def create_allocation_layers(self):
        self.create_input_alloc_layer()
        self.create_body_alloc_layer()
        self.create_head_alloc_layer()

    def create_body_alloc_layer(self):
        # self.conv1 = pyg_nn.GCNConv(self.input_dim, self.num_a_hidden_units)
        self.lin0 = nn.Linear(self.input_dim, self.num_a_hidden_units).to(self.device)
        # self.lin0 = GLU(self.input_dim, self.num_a_hidden_units).to(self.device)

        nn.init.xavier_uniform_(self.lin0.weight)
        nn.init.zeros_(self.lin0.bias)

        self.conv1 = CustomGATConv(self.num_a_hidden_units, self.num_a_hidden_units, heads=1, concat=False).to(self.device)
        self.conv2 = CustomGATConv(self.num_a_hidden_units, self.num_a_hidden_units, heads=1, concat=False).to(self.device)
        # self.conv3 = CustomGATConv(self.num_a_hidden_units, self.num_a_hidden_units, heads=1, concat=False).to(self.device)

        self.cross_attn_bundle_item = CrossAttention(embed_dim=self.num_a_hidden_units)
        self.cross_attn_agent_bundle = CrossAttention(embed_dim=self.num_a_hidden_units)


    def create_head_alloc_layer(self):
        if self.graph_type == "conflict":
            self.fc_alloc = nn.Linear(self.num_a_hidden_units * self.num_bundles, self.num_bundles).to(self.device)
            self.fc_alloc2 = nn.Linear(self.num_a_hidden_units * self.num_bundles, self.num_bundles).to(self.device)
            self.fc_item = nn.Linear(self.num_a_hidden_units * self.num_agents, self.num_items).to(self.device)
        else:
            self.fc_alloc = nn.Linear(self.num_a_hidden_units, self.num_bundles).to(self.device)
            self.fc_alloc2 = nn.Linear(self.num_a_hidden_units, self.num_bundles).to(self.device)
            self.fc_item = nn.Linear(self.num_a_hidden_units, self.num_items).to(self.device)

        self.init_weights(self.fc_alloc.weight)
        nn.init.zeros_(self.fc_alloc.bias)

        self.init_weights(self.fc_alloc2.weight)
        nn.init.zeros_(self.fc_alloc2.bias)
        
        self.init_weights(self.fc_item.weight)
        nn.init.zeros_(self.fc_item.bias)
        
        self.alloc_layers.append(self.fc_item)
        # make self.fc_pay deeper with several layers
        self.fc_pay = nn.Sequential(
            nn.Linear(self.num_bundles*self.num_agents, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_agents)
        ).to(self.device)
        self.init_weights(self.fc_pay[0].weight)
        nn.init.zeros_(self.fc_pay[0].bias)
        self.init_weights(self.fc_pay[2].weight)
        nn.init.zeros_(self.fc_pay[2].bias)

    def forward(self, x, c, tau=1, return_true=False):

        # normalize the input
        
        # x = (x - global_min) / (global_max - global_min + 1e-5)
        
        x = self.get_bundle_bid(x, c, self.incident_matrix)
        # x = torch.clamp(x, min=0.0)
        valuations = x
        
        batch_size = x.shape[0]

        if self.graph_type == "bipartite":
            data = self.vectorized_convert_batch_to_bipartite_graphs(x, self.incident_matrix).to(self.device)
        elif self.graph_type == "conflict":
            data = self.vectorized_convert_batch_to_conflict_graph(x, self.incident_matrix).to(self.device)
        else:  # tripartite
            data = self.vectorized_convert_batch_to_tripartite_graphs(x, self.incident_matrix).to(self.device)
        
        # input layer
        x = self.lin0(data.x)

        x = self.conv1(x, data.edge_index)
        x = self.conv2(x, data.edge_index)
        # x = self.conv3(x, data.edge_index)
        
        # head layer
        if self.graph_type == "conflict":
            # [B * (A*M), H] → [B, A, M, H]
            x = x.view(batch_size, self.num_agents, self.num_bundles, -1)
            # mean-pool or flatten for allocation prediction
            x_flat = x.view(batch_size, self.num_agents, -1)
            bidder_scores = self.fc_alloc(x_flat).view(batch_size, self.num_agents, self.num_bundles)
            agents_x = x_flat  # used for payment computation
        else:
            x = x.view(batch_size, self.num_agents + self.num_bundles + self.num_items, -1)
            agents_x = x[:, :self.num_agents]
            bundles_x = x[:, self.num_agents:self.num_agents+self.num_bundles]

            if self.graph_type == "tripartite":
                items_x = x[:, self.num_agents+self.num_bundles:]
                attn_out_bi, _ = self.cross_attn_bundle_item(bundles_x, items_x, items_x)
                bundles_x = bundles_x + attn_out_bi

            attn_out_ab, _ = self.cross_attn_agent_bundle(agents_x, bundles_x, bundles_x)
            agents_x = agents_x + attn_out_ab
            bidder_scores = self.fc_alloc(agents_x)

        alloc = F.softmax(bidder_scores / self.config.temp, dim=2)
        alloc2 = F.softmax(self.fc_alloc2(agents_x) / self.config.temp, dim=1)
        alloc = torch.min(alloc, alloc2)
        # alloc = sinkhorn(alloc)

        if self.graph_type != "conflict":
            
            item_scores = self.fc_item(bundles_x)
            item_probs = F.softmax(item_scores / self.config.temp, dim=1)
            item_probs = torch.clamp(item_probs, min=1e-12)
            item_bundle = item_probs * self.incident_matrix
            item_bundle = item_bundle.transpose(1, 2)
            masked_bundle = torch.where(item_bundle > 0, item_bundle, item_bundle.new_tensor(float('inf')))
            min_item_bundle = masked_bundle.min(dim=1).values.unsqueeze(1)
            final_alloc = min_item_bundle * alloc
            
        else:
            # === Enforce item validity constraints ===
            # item_scores from agent-view instead of bundles_x (conflict graph has no bundles_x)
            x_flat = x.view(batch_size, self.num_bundles, -1)
            item_scores = self.fc_item(x_flat)
            item_scores = item_scores.view(batch_size, self.num_bundles, self.num_items)
            item_probs = F.softmax(item_scores / self.config.temp / 2, dim=1)
            item_probs = torch.clamp(item_probs, min=1e-12)
            item_bundle = item_probs * self.incident_matrix
            item_bundle = item_bundle.transpose(1, 2)
            masked_bundle = torch.where(item_bundle > 0, item_bundle, item_bundle.new_tensor(float('inf')))
            min_item_bundle = masked_bundle.min(dim=1).values.unsqueeze(1)
            final_alloc = min_item_bundle * alloc

            # # check final alloc sum of col and row
            # alloc_sum_col = final_alloc.sum(dim=1, keepdim=True)
            # alloc_sum_row = final_alloc.sum(dim=2, keepdim=True)

            # check item-wise sum
            item_sum = final_alloc @ self.incident_matrix
            item_sum = item_sum.sum(dim=1, keepdim=True)

            # print("Final alloc sum col:", alloc_sum_col[0].cpu().detach().numpy().round(3))
            # print("Final alloc sum row:", alloc_sum_row[0].cpu().detach().numpy().round(3))
            # print("Item-wise sum:", item_sum[0].cpu().detach().numpy().round(3))


        # ---- Payments (shared head) ----
        pay = self.fc_pay(valuations.view(batch_size, -1))
        pay = torch.sigmoid(pay)

        # --- normalized payments/utility base (nonnegative) ---
        val_for_pay = torch.clamp(valuations, min=0.0)
        matrix_dot = (final_alloc * val_for_pay).sum(dim=-1)   # <-- use final_alloc
        final_pay = pay * matrix_dot

        if not return_true:
            return final_alloc, final_pay

        # --- unnormalized branch ---
        # Synthetic benchmarks train directly in valuation space (no dollar
        # rescaling), so the unnormalized valuations are the inputs themselves.
        unnorm_valuations = valuations.to(self.device)

        unnorm_val_for_pay = torch.clamp(unnorm_valuations, min=0.0)
        true_matrix_dot = (final_alloc * unnorm_val_for_pay).sum(dim=-1)  # <-- CRITICAL: use final_alloc here
        true_pay = pay * true_matrix_dot

        return final_alloc, final_pay, true_pay, unnorm_valuations

    def gen_incident_matrix(self):
        possible_bundles = list(itertools.product([0, 1], repeat=self.num_items))
        
        if self.config.airport_case:
            # Keep non-empty bundles with even number of 1s (i.e., even slot count)
            possible_bundles = [
                bundle for bundle in possible_bundles
                if any(bundle) and sum(bundle) % 2 == 0
            ]
        else:
            # Keep all non-empty bundles
            possible_bundles = [bundle for bundle in possible_bundles if any(bundle)]
        
        incident_matrix = np.array(possible_bundles)
        return torch.tensor(incident_matrix, dtype=torch.float)

    def vectorized_convert_batch_to_tripartite_graphs(self, x_batch, incident_matrix):
        """
        Constructs tripartite graphs (Agent ↔ Bundle ↔ Item) for a batch.

        Args:
            x_batch: Tensor of shape [B, A, M] - agent valuations for bundles
            incident_matrix: [M, I] - binary indicator of items in each bundle

        Returns:
            torch_geometric.data.Data
        """
        B, A, M = x_batch.shape  # batch size, agents, bundles
        I = incident_matrix.size(1)
        D = self.input_dim
        device = x_batch.device

        # ==== Node Features ====
        agent_feat = self.agent_embedding.unsqueeze(0).expand(B, -1, -1)    # [B, A, D]
        bundle_feat = self.bundle_embedding.unsqueeze(0).expand(B, -1, -1)  # [B, M, D]
        item_feat = self.item_embedding.unsqueeze(0).expand(B, -1, -1)      # [B, I, D]

        node_features = torch.cat([agent_feat, bundle_feat, item_feat], dim=1)  # [B, A+M+I, D]
        node_features = node_features.reshape(B * (A + M + I), D)

        # ==== Edge Index and Attributes ====
        # Build base edges (Agent ↔ Bundle) and (Bundle ↔ Item) for a single sample
        agent_bundle_edges = []
        for a in range(A):
            for b in range(M):
                agent_bundle_edges.append([a, A + b])
                agent_bundle_edges.append([A + b, a])  # bidirectional

        bundle_item_edges = []
        for b in range(M):
            for i in range(I):
                if incident_matrix[b, i] > 0:
                    bundle_item_edges.append([A + b, A + M + i])
                    bundle_item_edges.append([A + M + i, A + b])

        edge_index_base = torch.tensor(agent_bundle_edges + bundle_item_edges, dtype=torch.long).t()  # [2, E_base]

        # Repeat for batch with offsets
        edge_indices = []
        edge_attrs = []

        for i in range(B):
            offset = i * (A + M + I)
            ei = edge_index_base + offset

            edge_indices.append(ei)

            # Use valuation for agent↔bundle, 1.0 for bundle↔item
            val_matrix = x_batch[i]  # [A, M]
            ab_edge_attr = val_matrix.flatten()
            ab_edge_attr = torch.cat([ab_edge_attr, ab_edge_attr])  # bidirectional

            bi_edge_attr = torch.ones(len(bundle_item_edges), device=device)

            edge_attrs.append(torch.cat([ab_edge_attr, bi_edge_attr]))

        edge_index = torch.cat(edge_indices, dim=1)
        edge_attr = torch.cat(edge_attrs)

        # Batch indicator for torch_geometric
        batch = torch.arange(B, device=device).repeat_interleave(A + M + I)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

    def vectorized_convert_batch_to_bipartite_graphs(self, x_batch, incident_matrix):
        """
        Efficiently constructs a batched bipartite graph from input valuations.
        
        Args:
            x_batch (Tensor): shape [B, num_bidders, num_bundles]
            incident_matrix (Tensor): shape [num_bundles, num_items]
        
        Returns:
            torch_geometric.data.Data: Batched graph
        """
        B, A, M = x_batch.shape  # batch, bidders, bundles
        I = self.incident_matrix.size(1)  # items
        incident_matrix = torch.nan_to_num(incident_matrix, nan=0.0)

        device = x_batch.device

        # === Node Features ===
        # bidder_bid_features = x_batch.mean(dim=2, keepdim=True)           # [B, A, 1]
        # bidder_padding = torch.zeros((B, A, I), device=device)            # [B, A, I]
        # bidder_padding = torch.nan_to_num(bidder_padding)
        # bidder_features = torch.cat([bidder_bid_features, bidder_padding], dim=2)  # [B, A, 1+I]

        # bundle_bid_features = x_batch.mean(dim=1, keepdim=True).transpose(1, 2)     # [B, M, 1]
        # bundle_comp_features = incident_matrix.unsqueeze(0).expand(B, M, I).to(device)  # [B, M, I]
        # bundle_features = torch.cat([bundle_bid_features, bundle_comp_features], dim=2)  # [B, M, 1+I]

        # === Learned Node Embeddings ===
        B, A, M = x_batch.shape
        agent_emb = self.agent_embedding.unsqueeze(0).expand(B, -1, -1)   # [B, A, input_dim]
        bundle_emb = self.bundle_embedding.unsqueeze(0).expand(B, -1, -1) # [B, M, input_dim]
        node_features = torch.cat([agent_emb, bundle_emb], dim=1)         # [B, A+M, input_dim]
        node_features = node_features.reshape(B * (A + M), self.input_dim) 
        if torch.isnan(node_features).any():
            raise ValueError("node_features has NaNs after embedding")


        # === Edge Index and Edge Attr ===
        edge_list = []
        for b in range(A):
            for m in range(M):
                edge_list.append([b, A + m])
                edge_list.append([A + m, b])
        edge_index_base = torch.tensor(edge_list, dtype=torch.long).t()  # [2, num_edges]

        edge_indices = []
        edge_attrs = []
        for i in range(B):
            offset = i * (A + M)
            ei = edge_index_base + offset
            edge_indices.append(ei)

            x = x_batch[i]  # [A, M]
            ea = x.flatten()
            ea = torch.cat([ea, ea])  # both directions
            edge_attrs.append(ea)

        edge_index = torch.cat(edge_indices, dim=1)  # [2, total_edges]
        edge_attr = torch.cat(edge_attrs)            # [total_edges]
        batch = torch.arange(B, device=device).repeat_interleave(A + M)  # [B*(A+M)]

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    
    def vectorized_convert_batch_to_conflict_graph(self, x_batch, incident_matrix):
        """
        Constructs a conflict graph where each node is a (bidder, bundle) pair,
        and edges connect pairs with overlapping items.

        Args:
            x_batch: [B, A, M] tensor of bids
            incident_matrix: [M, I] binary tensor showing bundle-item composition

        Returns:
            torch_geometric.data.Data
        """
        B, A, M = x_batch.shape
        I = incident_matrix.size(1)
        D = self.embedding_dim  # Not self.input_dim here!
        device = x_batch.device

        # === Node Features ===
        bids = x_batch.view(B, A * M, 1)  # [B, A*M, 1]

        bundle_feat = self.bundle_embedding.unsqueeze(0).unsqueeze(0).expand(B, A, M, D)  # [B, A, M, D]
        bundle_feat = bundle_feat.reshape(B, A * M, D)

        agent_feat = self.agent_embedding.unsqueeze(0).unsqueeze(2).expand(B, A, M, D)  # [B, A, M, D]
        agent_feat = agent_feat.reshape(B, A * M, D)

        # Skip first dim from agent/bundle features (don't repeat bid val)
        node_features = torch.cat([bids, agent_feat[..., 1:], bundle_feat[..., 1:]], dim=-1)  # [B, A*M, input_dim]
        node_features = node_features.view(B * A * M, -1)

        # === Conflict Edges ===
        incident_bin = incident_matrix.bool()
        bundle_conflict = (incident_bin.unsqueeze(1) & incident_bin.unsqueeze(0)).any(-1).float()  # [M, M]
        conflict_idx = bundle_conflict.nonzero(as_tuple=False)
        conflict_idx = conflict_idx[conflict_idx[:, 0] != conflict_idx[:, 1]]  # remove self-loops

        edge_index = []
        for a in range(A):
            base = a * M
            edges = conflict_idx + base
            edge_index.append(edges.t())

        edge_index = torch.cat(edge_index, dim=1)  # [2, E_total_per_batch]
        edge_index = edge_index.unsqueeze(0).repeat(B, 1, 1)  # [B, 2, E]
        for i in range(B):
            edge_index[i] += i * A * M
        edge_index = edge_index.view(2, -1)

        edge_attr = torch.ones(edge_index.size(1), device=device)
        batch = torch.arange(B, device=device).repeat_interleave(A * M)

        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


    
def sinkhorn(logits, n_iters=10, tau=1.0):
    # logits: [B, A, M]
    log_probs = logits / tau
    for _ in range(n_iters):
        log_probs = log_probs - log_probs.logsumexp(dim=2, keepdim=True)  # Row norm
        log_probs = log_probs - log_probs.logsumexp(dim=1, keepdim=True)  # Col norm
    return log_probs.exp()


