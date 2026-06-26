import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.utils import get_bundle_bid

class CANet(nn.Module):
    def __init__(self, model_config, device):
        super(CANet, self).__init__()
        self.config = model_config
        self.device = device
        self.init()

    def init(self):
        self.alloc_layers = nn.ModuleList([])
        self.pay_layers = nn.ModuleList([])

        self.create_constants()
        self.create_allocation_layers()
        self.create_payment_layers()

    def create_constants(self):
        if self.config.net.init == "None":
            init = None
        elif self.config.net.init == "gu":
            init = nn.init.xavier_uniform_
        elif self.config.net.init == "gn":
            init = nn.init.xavier_normal_
        self.init_weights = init

        if self.config.net.activation == "tanh":
            activation = nn.Tanh()
        elif self.config.net.activation == "relu":
            activation = nn.ReLU()
        self.activation = activation

        self.num_agents = self.config.num_agents
        self.num_items = self.config.num_items
        self.num_bundles = self.config.num_bundles
        self.incident_matrix = self.gen_incident_matrix()

        self.num_a_layers = self.config.net.num_a_layers
        self.num_p_layers = self.config.net.num_p_layers

        self.num_a_hidden_units = self.config.net.num_a_hidden_units
        self.num_p_hidden_units = self.config.net.num_p_hidden_units

        # self.num_in = self.num_agents * self.num_items
        # self.num_a_output = (self.num_agents + 1) * (self.num_items + 1)
        self.num_in = self.num_agents * self.num_bundles
        self.num_a_output = (self.num_agents) * (self.num_bundles)
        self.num_a_output_item = (self.num_items) * (self.num_bundles)

        self.ln = self.config.net.layer_norm
        self.bn = self.config.net.batch_norm

        self.dropout_a = nn.Dropout(p=0.00)

        # bin_score = torch.nn.Parameter(torch.tensor(1.))
        # self.register_parameter('bin_score', bin_score)

    def create_allocation_layers(self):
        self.create_input_alloc_layer()
        self.create_body_alloc_layer()
        self.create_head_alloc_layer()
        if self.ln:
            self.create_ln_alloc_layers()
        if self.bn:
            self.create_bn_alloc_layers()

    def create_input_alloc_layer(self):
        alloc_first_layer = nn.Linear(self.num_in, self.num_a_hidden_units).to(self.device)
        self.init_weights(alloc_first_layer.weight)
        nn.init.zeros_(alloc_first_layer.bias)
        self.alloc_layers.append(alloc_first_layer)

    def create_body_alloc_layer(self):
        for i in range(1, self.num_a_layers - 1):
            alloc_new_layer = nn.Linear(self.num_a_hidden_units, self.num_a_hidden_units).to(self.device)
            self.init_weights(alloc_new_layer.weight)
            nn.init.zeros_(alloc_new_layer.bias)
            self.alloc_layers.append(alloc_new_layer)

    def create_head_alloc_layer(self):
        alloc_output_layer1 = nn.Linear(self.num_a_hidden_units, self.num_a_output).to(self.device)
        self.init_weights(alloc_output_layer1.weight)
        nn.init.zeros_(alloc_output_layer1.bias)
        self.alloc_layers.append(alloc_output_layer1)

        alloc_output_layer2 = nn.Linear(self.num_a_hidden_units, self.num_a_output).to(self.device)
        self.init_weights(alloc_output_layer2.weight)
        nn.init.zeros_(alloc_output_layer2.bias)
        self.alloc_layers.append(alloc_output_layer2)

        alloc_output_layer3 = nn.Linear(self.num_a_hidden_units, self.num_a_output_item).to(self.device)
        self.init_weights(alloc_output_layer3.weight)
        nn.init.zeros_(alloc_output_layer3.bias)
        self.alloc_layers.append(alloc_output_layer3)

    def create_ln_alloc_layers(self):
        self.a_lns = nn.ModuleList([])
        for i in range(self.num_a_layers - 1):
            # layer = nn.LayerNorm(self.num_a_hidden_units, eps=1e-3).to(self.device)
            layer = nn.SpectralNorm(nn.LayerNorm(self.num_a_hidden_units).to(self.device))
            self.a_lns.append(layer)

    def create_bn_alloc_layers(self):
        self.a_bns = nn.ModuleList([])
        for i in range(self.num_a_layers - 1):
            layer = nn.BatchNorm1d(self.num_a_hidden_units).to(self.device)
            self.a_bns.append(layer)

    def create_payment_layers(self):
        self.create_input_payment_layer()
        self.create_body_payment_layer()
        self.create_head_payment_layer()
        if self.ln:
            self.create_ln_payment_layers()
        if self.bn:
            self.create_bn_payment_layers()

    def create_input_payment_layer(self):
        pay_first_layer = nn.Linear(self.num_in, self.num_p_hidden_units).to(self.device)
        self.init_weights(pay_first_layer.weight)
        nn.init.zeros_(pay_first_layer.bias)
        self.pay_layers.append(pay_first_layer)

    def create_body_payment_layer(self):
        for i in range(1, self.num_p_layers - 1):
            pay_new_layer = nn.Linear(self.num_p_hidden_units, self.num_p_hidden_units).to(self.device)
            self.init_weights(pay_new_layer.weight)
            nn.init.zeros_(pay_new_layer.bias)
            self.pay_layers.append(pay_new_layer)

    def create_head_payment_layer(self):
        pay_output_layer = nn.Linear(self.num_p_hidden_units, self.num_agents).to(self.device)
        self.init_weights(pay_output_layer.weight)
        nn.init.zeros_(pay_output_layer.bias)
        self.pay_layers.append(pay_output_layer)

    def create_ln_payment_layers(self):
        self.p_lns = nn.ModuleList([])
        for i in range(self.num_p_layers - 1):
            layer = nn.LayerNorm(self.num_p_hidden_units, eps=1e-3).to(self.device)
            self.p_lns.append(layer)

    def create_bn_payment_layers(self):
        self.p_bns = nn.ModuleList([])
        for i in range(self.num_p_layers - 1):
            layer = nn.BatchNorm1d(self.num_p_hidden_units).to(self.device)
            self.p_bns.append(layer)

    def forward(self, x, c, return_intermediates=False, tau=1.0, return_true=False):
        x = get_bundle_bid(x, c, self.incident_matrix)

        x_in = x.view([-1, self.num_in])  # reshape to vector

        alloc = self.forward_th_allocation(x_in)
        pay = self.forward_th_payment(x_in)

        # final layer
        matrix_dot = (alloc * x).sum(dim=-1)
        final_pay = pay * matrix_dot

        if return_true:
            return alloc, final_pay, final_pay, x
        if return_intermediates:
            return alloc, final_pay, pay
        return alloc, final_pay

    def forward_th_allocation(self, x):
        alloc = self.alloc_layers[0](x)
        if self.ln:
            alloc = self.a_lns[0](alloc)
        if self.bn:
            alloc = self.a_bns[0](alloc)
        alloc = self.activation(alloc)
        for k in range(1, self.num_a_layers - 3):
            alloc = self.alloc_layers[k](alloc)
            if self.ln:
                alloc = self.a_lns[k](alloc)
            if self.bn:
                alloc = self.a_bns[k](alloc)
            alloc = self.activation(alloc)
        alloc = self.dropout_a(alloc)
        temp = self.config.temp
        # if hasattr(self.config, 'bundle') and self.config.bundle is not None:
        agent_bundle1 = self.alloc_layers[-3](alloc)
        agent_bundle2 = self.alloc_layers[-2](alloc)
        agent_bundle1 = F.softmax(agent_bundle1.view([-1, self.num_agents, self.num_bundles]) / temp, dim=1)
        agent_bundle2 = F.softmax(agent_bundle2.view([-1, self.num_agents, self.num_bundles]) / temp, dim=-1)
        agent_bundle = torch.min(agent_bundle1, agent_bundle2)#[:, :-1, :-1]
        item_bundle = self.alloc_layers[-1](alloc)
        item_bundle = F.softmax(item_bundle.view([-1, self.num_items, self.num_bundles]) / temp / 2, dim=-1)
        item_bundle = torch.clamp(item_bundle, min=1e-12)
        # i = i.expand(item_bundle.size(0), -1, -1)

        if isinstance(self.incident_matrix, torch.Tensor):
            i = self.incident_matrix.clone().detach().to(dtype=torch.float32, device=x.device)
        else:
            i = torch.tensor(self.incident_matrix, dtype=torch.float32, device=x.device)
        i_transpose = i.transpose(1, 2)
        item_bundle = item_bundle * i_transpose

        masked_bundle = torch.where(item_bundle > 0, item_bundle, item_bundle.new_tensor(float('inf')))
        item_bundle = masked_bundle.min(dim=1).values.unsqueeze(1)
        alloc = item_bundle * agent_bundle
        # item = torch.bmm(alloc, i.expand(item_bundle.size(0), -1, -1))
        # assert torch.all(item.sum(dim=1) <= 1), f"item allocation {item.sum(dim=-1)} exceeds 1"
        # assert torch.all(alloc.sum(dim=1) <= 1), f"agent allocation {alloc.sum(dim=1)} exceeds 1"
        # assert torch.all(alloc.sum(dim=2) <= 1), f"bundle allocation {alloc.sum(dim=2)} exceeds 1"
        return alloc

    def forward_th_payment(self, x):
        pay = self.pay_layers[0](x)
        if self.ln:
            pay = self.p_lns[0](pay)
        if self.bn:
            pay = self.p_bns[0](pay)
        pay = self.activation(pay)
        for i in range(1, self.num_p_layers - 1):
            pay = self.pay_layers[i](pay)
            if self.ln:
                pay = self.p_lns[i](pay)
            if self.bn:
                pay = self.p_bns[i](pay)
            pay = self.activation(pay)

        pay = self.pay_layers[-1](pay)
        pay = torch.sigmoid(pay)

        return pay
    
    def log_sinkhorn_iterations(self, Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)

    def log_optimal_transport(self, scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
        """ Perform Differentiable Optimal Transport in Log-space for stability"""
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m*one).to(scores), (n*one).to(scores)

        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                            torch.cat([bins1, alpha], -1)], 1)

        norm = - (ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        Z = Z - norm  # multiply probabilities by M+N
        return Z

    def gen_incident_matrix(self):
        possible_bundles = list(itertools.product([0, 1], repeat=self.num_items))
        possible_bundles = [bundle for bundle in possible_bundles if any(bundle)] 
        self.num_bundles = len(possible_bundles)   
        incident_matrix = np.array(possible_bundles)
        incident_matrix = np.broadcast_to(incident_matrix, (1, self.num_bundles, self.num_items))
        return incident_matrix