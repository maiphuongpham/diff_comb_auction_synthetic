import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from networkx.algorithms import bipartite
from scipy.sparse import csr_matrix
import random
from core.utils import get_bundle_bid#, gen_incident_matrix


def oracle(data, language="marginal", n_best_items=None, incident_matrix=None):
    """
    Allocation oracle.
    :param data: tensor with bids (valuations) in auctions shaped as (batch_size, n_agents, n_items)
    :param language: bidding language, either
        'additive' for heterogenous goods and additive utilities,
        'marginal' for homogenous goods and marginally decreasing utilities
            (utility of a bundle is sum of item-wise utilities, s.t. each new item produces less utility) or
        'unit-demand' for heterogenous goods and unit demand (utility of a bundle is max of item-wise utilities)
        * 'hierarchical' for hierarchical bundles?
    :param n_best_items: number of items to allocate, the default is all items
    :return: allocation, i.e. binary tensor with same shape as valuation
    """
    size, n_participants, n_items = data.size()
    if n_best_items is None:
        n_best_items = n_items
    n_best_items = min(n_best_items, n_items)

    if language == "additive":
        allocation = torch.argmax(data, 1, True)
        allocation = F.one_hot(allocation)[:, 0].permute((0, 2, 1))
        allocation = topk_allocation(data, allocation, n_best_items)

    elif language == "marginal":
        allocation = torch.zeros((size, n_participants)).long()
        for i, auction in enumerate(data):
            auction_copy = auction.clone()
            v, idx = auction_copy[:, 0], torch.zeros((n_participants,)).long()
            for j in range(n_best_items):
                part = torch.argmax(v).item()
                idx[part] += 1
                if j != n_items - 1:
                    v[part] = auction_copy[part, idx[part]] - auction_copy[part, idx[part] - 1]
            allocation[i] = idx
        allocation = torch.zeros(size, n_participants, n_items + 1).scatter_(2, allocation.unsqueeze(-1), 1)[:, :, 1:]

    elif language == "unit-demand":
        n_best_items = min(n_best_items, n_participants)
        allocation = torch.zeros(size, n_participants, n_items)
        for i, auction in enumerate(data):
            allocation_cur = torch.zeros(n_participants, n_items)
            graph = csr_matrix(-auction.detach().numpy())
            graph = bipartite.from_biadjacency_matrix(graph)
            matching = bipartite.matching.minimum_weight_full_matching(graph)
            for part in range(n_participants):
                if part in matching.keys():
                    item = matching[part] - n_participants
                    allocation_cur[part, item] = 1
            allocation[i] = allocation_cur
        allocation = topk_allocation(data, allocation, n_best_items)

    elif language == "combinatorial":
        # make sure incident_matrix is specified
        if incident_matrix is None:
            raise ValueError("Incident matrix must be provided for combinatorial language")
        # if incident_matrix is 3D, make it 2D
        if len(incident_matrix.shape) == 3:
            incident_matrix = incident_matrix[0, :, :]
        n_best_items = min(n_best_items, n_participants)
        n_bundles = n_items
        allocation = torch.zeros(size, n_participants, n_items)
        for i, auction in enumerate(data):
            allocation_cur = torch.zeros(n_participants, n_items)
            graph = nx.Graph()
            # Add nodes with bidder and bundle indices
            for bidder in range(n_participants):
                for bundle in range(n_bundles):
                    node = f"bidder_{bidder}_bundle_{bundle}"
                    weight = auction.detach().cpu().numpy()[bidder, bundle]
                    graph.add_node(node, weight=weight)

                    for other_bidder in range(n_participants):
                        if other_bidder != bidder:
                            conflicting_node = f"bidder_{other_bidder}_bundle_{bundle}"
                            graph.add_edge(node, conflicting_node)
                            
                    for other_bundle in range(n_bundles):
                        if other_bundle != bundle:
                            conflicting_node = f"bidder_{bidder}_bundle_{other_bundle}"
                            graph.add_edge(node, conflicting_node)
            
            for j in range(n_bundles):
                for k in range(j + 1, n_bundles):                    
                    for bidder_i in range(n_participants):
                        for bidder_j in range(n_participants):
                            if np.any(incident_matrix[j] & incident_matrix[k]):
                            # if j == k or np.any(incident_matrix[j] & incident_matrix[k]):

                                node_i = f"bidder_{bidder_i}_bundle_{j}"
                                node_j = f"bidder_{bidder_j}_bundle_{k}"
                                graph.add_edge(node_i, node_j)

            matching = solve_maximum_independent_set(graph)
            matching = {int(node.split("_")[1]): int(node.split("_")[3]) for node in matching}
            for part in range(n_participants):
                if part in matching.keys():
                    item = matching[part] #- n_participants
                    allocation_cur[part, item] = 1
            allocation[i] = allocation_cur
        # move allocation to data device
        allocation = allocation.to(data.device)
        allocation = topk_allocation(data, allocation, n_best_items)

    else:
        raise NotImplementedError("Only 'additive', 'marginal', and 'unit-demand' languages are implemented")

    return allocation


def topk_allocation(data, allocation, n_best_items):
    threshold = torch.topk((data * allocation).reshape(data.shape[0], -1), n_best_items, -1)[0][:, -1].view(-1, 1, 1)
    allocation[data < threshold] = 0
    return allocation


def get_v(data, allocation):
    """
    :param data: tensor with bids (valuations) in auctions shaped as (batch_size, n_agents, n_items)
    :param allocation: efficient allocation, output of oracle, binary tensor shaped as (batch_size, n_agents, n_items)

    :return: total value of objects gained by each agent in each auction with respect to the efficient allocations,
        tensor shaped as (batch_size, n_agents)
    """
    return torch.sum(data * allocation, 2)


def delete_agent(x, i):
    mask = torch.ones(x.size()[1]).int()
    mask[i] = 0
    return x[:, mask.bool()]


def get_v_sum_but_i(v):
    return torch.cat([torch.sum(delete_agent(v, i), dim=1).view(-1, 1) for i in range(v.size()[1])], dim=1)

def prepare_auctions(auctions, language="marginal"):
    if language == "additive":
        pass
    elif language == "marginal":
        auctions = torch.sort(auctions, dim=2, descending=True)[0]
        auctions = torch.cumsum(
            auctions, 2
        )  ### this is different from the paper, but the marginal utilities can increase otherwise
    elif language == "unit-demand":
        pass
    else:
        raise NotImplementedError("Only 'additive', 'marginal', and 'unit-demand' languages are implemented")
    return auctions

def get_batch(auctions, batch_size=256, return_idx=False):
    batch_size = min(batch_size, len(auctions))
    # idx = torch.randperm(len(auctions))[:batch_size]
    idx = torch.Tensor(random.sample(range(len(auctions)), batch_size)).long()
    if not return_idx:
        return auctions[idx]
    else:
        return idx, auctions[idx]


def get_representation(data, language="marginal"):
    """
    :param data: tensor with bids (valuations) in auctions shaped as (batch_size, n_agents, n_items)
    :return: auction representations that are used as input for DeepMindNet, tensor shaped as (batch_size, n_agents, n_items, n_channels)
    """
    old_shape = list(data.shape)
    n_items = old_shape[2]
    representation = [data]

    for i in range(n_items):
        allocation = oracle(data, language, i + 1).float()
        utility = data * allocation
        representation.extend([allocation, utility])

    representation = torch.stack(representation, -1)
    return representation

def VCG(batch, language="combinatorial", incident_matrix=None):
    """
    VCG efficient and truthful mechanism.
    :param batch: tensor with bids (valuations) in auctions shaped as (batch_size, n_agents, n_items)
    :return: VCG prices t, tensor shaped as (batch_size, n_agents)
    """
    # convert batch to numpy if needed
    # if isinstance(batch, torch.Tensor):
    #     batch = batch.detach().cpu().numpy()
    # Ensure batch is a 3D numpy array
    allocation = oracle(batch, language, incident_matrix=incident_matrix)
    v = get_v(batch, allocation)
    v_sum_but_i = get_v_sum_but_i(v)

    h = []
    for i in range(batch.shape[1]):
        batch_cur = delete_agent(batch, i)
        allocation_cur = oracle(batch_cur, language, incident_matrix=incident_matrix)
        v_cur = get_v(batch_cur, allocation_cur)
        v_sum_cur = v_cur.sum(dim=-1).view(-1, 1)
        h.append(v_sum_cur)
    h = torch.cat(h, dim=1)

    t = h - v_sum_but_i
    
    payment_frac = (t / batch.sum(dim=2))
    return allocation, payment_frac


def solve_maximum_independent_set(graph):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    # Create a Gurobi model
    model = gp.Model("Maximum_Independent_Set", env=env)

    # Create a variable for each vertex in the graph
    x = model.addVars(graph.nodes(), vtype=GRB.BINARY, name="x")

    # Set the objective: maximize the sum of selected vertices
    # model.setObjective(gp.quicksum(x[v] for v in graph.nodes()), GRB.MAXIMIZE) # unweighted
    model.setObjective(gp.quicksum(graph.nodes[v]['weight'] * x[v] for v in graph.nodes()), GRB.MAXIMIZE) # weighted

    # Add constraints: for each edge (i, j), x[i] + x[j] <= 1
    for u, v in graph.edges():
        model.addConstr(x[u] + x[v] <= 1, name=f"edge_{u}_{v}")

    # Optimize the model
    model.optimize()

    # Extract the solution
    independent_set = [v for v in graph.nodes() if x[v].x > 0.5]

    # print(independent_set)

    return independent_set


# # from core.base.base_generator_ca import BaseGenerator

# with open('/scratch/maip/auction/canet/core/data/valuation_without_connection_3.json', 'r') as f:
#     values = json.load(f)

# def _generate_airport_data_per_step(dict_of_values):
#     """
#     Converts nested bundle valuation dictionaries into a 2D numpy array.

#     Input:
#         dict_of_values: dict {
#             'B6': { '0': {'slots': [...], 'value': v1}, ... },
#             'F9': { '0': {'slots': [...], 'value': v2}, ... },
#             ...
#         }

#     Output:
#         matrix: np.array of shape (num_bidders, num_bundles)
#         bidder_list: list of bidder IDs
#         bundle_indices: list of bundle indices used as columns
#     """
#     bidder_list = sorted(dict_of_values.keys())
#     all_bundle_indices = sorted({
#         int(b) for v in dict_of_values.values() for b in v.keys()
#     })

#     matrix = np.zeros((len(bidder_list), len(all_bundle_indices)))

#     for i, bidder in enumerate(bidder_list):
#         for j, bundle_idx in enumerate(all_bundle_indices):
#             bundle_str = str(bundle_idx)
#             if bundle_str in dict_of_values[bidder]:
#                 value = dict_of_values[bidder][bundle_str].get("value", 0.0)
#                 matrix[i, j] = float(value)

#     return matrix #, bidder_list, all_bundle_indices

# def generate_airport(list_of_dicts):
#     """
#     Args:
#         list_of_dicts: List of dictionaries containing decoded values.
#     Returns:
#         X: List
#     """
#     X_list = []

#     for dict_of_values in list_of_dicts:
#         X_list.append(_generate_airport_data_per_step(dict_of_values))
#     return np.array(X_list)

# def gen_incident_matrix():
#     num_items = 4
#     possible_bundles = list(itertools.product([0, 1], repeat=num_items))
    
#     # Keep non-empty bundles with even number of 1s (i.e., even slot count)
#     possible_bundles = [
#         bundle for bundle in possible_bundles
#         if any(bundle) and sum(bundle) % 2 == 0
#     ]
#     incident_matrix = np.array(possible_bundles)
#     return incident_matrix