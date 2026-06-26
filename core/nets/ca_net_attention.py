import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from layers.blocks import CrossAttention, AttentionHead, MHAttentionBody, MLPHead, PositionalEncoding
from layers.exchangeable_layer import Exchangeable
from core.nets.ca_net import CANet
from core.utils import get_bundle_bid


class CANetFormer(CANet):
    def __init__(self, model_config, device):
        super().__init__(model_config, device)

    def init(self):
        self.hid = self.config.net.hid
        self.hid_att = self.config.net.hid_att
        self.n_layers = self.config.net.n_attention_layers
        self.n_heads = self.config.net.n_attention_heads

        self.num_agents = self.config.num_agents
        self.num_items = self.config.num_items
        self.num_bundles = self.config.num_bundles

        if self.config.cc2 and self.config.val_model == "mix_val":
                self.incident_matrix = self.cc2_incident_matrix()
                self.get_bundle_bid = self._get_bundle_bid
        else:
            self.incident_matrix = self.gen_incident_matrix()
            self.get_bundle_bid = get_bundle_bid
            
        if self.config.net.activation_att.lower() == 'relu':
            self.activation = F.relu
        elif self.config.net.activation_att.lower() == 'tanh':
            self.activation = F.tanh
        else:
            raise NotImplementedError

        self.create_layers()   

    def create_layers(self):
        self.create_input_layers()
        self.create_body_layers()
        self.create_head_layers()

    def create_input_layers(self):
        self.input_layer_item = Exchangeable(1, self.hid, add_channel_dim=True).to(self.device)
        self.input_layer_agent = Exchangeable(1, self.hid, add_channel_dim=True).to(self.device)

        if self.config.net.pos_enc:
            if self.config.net.pos_enc_part > 1:
                pos_enc_part_layer = PositionalEncoding(self.config.net.pos_enc_part, self.hid, item_wise=False)
                self.input_layer_agent = nn.Sequential(self.input_layer_agent, pos_enc_part_layer)
            if self.config.net.pos_enc_item > 1:
                pos_enc_item_layer = PositionalEncoding(self.config.net.pos_enc_item, self.hid, item_wise=True)
                self.input_layer_agent = nn.Sequential(self.input_layer_agent, pos_enc_item_layer)
                self.input_layer_item = nn.Sequential(self.input_layer_item, pos_enc_item_layer)

    def create_body_layers(self):
        self.body_layers_item = nn.ModuleList(
            [MHAttentionBody(self.n_heads, self.hid, self.hid_att) for _ in range(self.n_layers)]
        ).to(self.device)

        self.body_layers_agent = nn.ModuleList(
            [MHAttentionBody(self.n_heads, self.hid, self.hid_att) for _ in range(self.n_layers)]
        ).to(self.device)

    def create_head_layers(self):
        self.cross_attn = CrossAttention(embed_dim=self.hid)
        self.head_layer = AttentionHead(hid=self.hid, config=self.config).to(self.device)

    def forward(self, x, c, return_intermediates=False, tau=None, return_true=False):
        """
        Args:
            x: [batch_size, num_agents, num_bundles]
            c: [batch_size, num_agents, num_bundles]
            return_intermediates: bool
            tau: temperature for softmax
        """
        x = self.get_bundle_bid(x, c, self.incident_matrix)
        x = torch.clamp(x, min=1e-5)  # Ensure values are non-negative
        valuations = x


        x_item = torch.bmm(x.transpose(1, 2), x[:, :, :self.num_items]).transpose(1, 2)
        x_item = self.activation(self.input_layer_item(x_item))

        x = self.activation(self.input_layer_agent(x))

        for i in range(self.n_layers):
            x_item = self.activation(self.body_layers_item[i](x_item)) # [-1, self.num_items, self.num_bundles, self.hid]
            x_agent = self.activation(self.body_layers_agent[i](x)) # [-1, self.num_agents, self.num_bundles, self.hid]

        # x_item = torch.bmm(x_agent.mean(-1).transpose(1, 2), x_agent.mean(-1)[:, :, :self.num_items]).transpose(1, 2)
        # x_item = self.activation(self.input_layer_item(x_item))

        # Cross Attention
        x_agent_sum = x_agent.sum(dim=1, keepdim=True)
        attn_output, _ = self.cross_attn(
            x_item.reshape(-1, self.num_items, self.hid), 
            x_agent_sum.reshape(-1, 1, self.hid), x_agent_sum.reshape(-1, 1, self.hid))
        attn_output = attn_output.reshape(-1, self.num_items, self.num_bundles, self.hid)
        x_item = x_item + attn_output

        alloc, pay = self.head_layer(x_item, x_agent, self.incident_matrix, tau)

        matrix_dot = (alloc * valuations).sum(dim=-1)
        final_pay = pay * matrix_dot

        if return_true:
            return alloc, final_pay, final_pay, valuations
        if return_intermediates:
            return alloc, final_pay, pay
        return alloc, final_pay
    
    def gen_incident_matrix(self):
        possible_bundles = list(itertools.product([0, 1], repeat=self.num_items))
        possible_bundles = [bundle for bundle in possible_bundles if any(bundle)] 
        self.num_bundles = len(possible_bundles)   
        incident_matrix = np.array(possible_bundles)
        incident_matrix = np.broadcast_to(incident_matrix, (1, self.num_bundles, self.num_items))
        return torch.tensor(incident_matrix).float().to(self.device)
    
    def cc2_incident_matrix(self):
        # Action Primitives: Analyse, Remove, Restore
        incident_matrix = np.array([
            [1, 0, 0],  # Analyse
            [0, 1, 0],  # Remove
            [0, 0, 1],  # Restore
            [1, 0, 1],  # Analyse + Restore
            [1, 1, 0],  # Analyse + Remove
            [1, 1, 1],  # All
            # Optional:
            # [0, 1, 1],  # Remove + Restore
        ])
        incident_matrix = np.broadcast_to(incident_matrix, (1, self.num_bundles, self.num_items))
        return torch.tensor(incident_matrix).float().to(self.device)

    def _get_bundle_bid(self, x, c, i):
        c = c.repeat(int(x.size(0) / c.size(0)), 1, 1)
        i = torch.tensor(i, dtype=torch.float32, device=x.device)
        # print("i", i)
        x = torch.bmm(x, i.expand(x.size(0), -1, -1).transpose(1, 2))
        # x_3 = x.sum(dim=-1) + c[:, :, 0]
        # x = torch.cat([x, x_3.unsqueeze(-1)], dim=-1)
        x[:,:,3] = x[:,:,3] + c[:,:,0]
        x[:,:,4] = x[:,:,4] + c[:,:,1]
        x[:,:,5] = x[:,:,5] + c[:,:,2]
        return x