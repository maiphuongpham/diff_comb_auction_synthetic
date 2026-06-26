import subprocess
import os
import itertools
import numpy as np
import torch

# Clipping valuation values functions
from core.clip_ops.clip_ops import *

# Data generator classes
from core.data import *


CLIPS = {
    'ca_uniform_01': lambda x: clip_op_01(x),
    'ca_uniform_12': lambda x: clip_op_12(x),
    'ca_uniform_12_15': lambda x: clip_op_12_15(x),
}

GENERATORS = {
    'ca_uniform_01': ca_uniform_01_generator.Generator,
    'ca_uniform_12': ca_uniform_12_generator.Generator,
    'ca_uniform_12_15': ca_uniform_12_15_generator.Generator,
}


def get_path_and_file(setting_path):
    path, file = os.path.split(setting_path)
    # Convert a directory path (possibly with subfolders, e.g.
    # "run_configs/airport") into a dotted module path so configs can be
    # organized into subdirectories.
    path = path.replace(os.sep, ".").replace("/", ".")
    return path, os.path.splitext(file)[0]


def get_objects(setting_path):
    '''
    Get objects from configuration file
    '''
    path, setting_name = get_path_and_file(setting_path)
    import_obj = __import__(path, fromlist=[setting_name])
    cfg = getattr(import_obj, setting_name).cfg
    clip_op = CLIPS[cfg.distribution_type]
    generator = GENERATORS[cfg.distribution_type]
    return cfg, clip_op, generator, setting_name


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], encoding="utf-8"
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

# def get_bundle_bid(x, c, i):
#     c = c.repeat(int(x.size(0) / c.size(0)), 1, 1)
#     i = torch.tensor(i, dtype=torch.float32, device=x.device)
#     if x.size()[-1] == 2:
        
#         # print("i", i)
#         x = torch.bmm(x, i.expand(x.size(0), -1, -1).transpose(1, 2))
#         # x_3 = x.sum(dim=-1) + c[:, :, 0]
#         # x = torch.cat([x, x_3.unsqueeze(-1)], dim=-1)
#         x[:,:,2] = x[:,:,2] + c[:,:,0]
#     elif x.size()[-1] == 3:
#         x = torch.bmm(x, i.expand(x.size(0), -1, -1).transpose(1, 2))
#         x[:,:,3] = x[:,:,3] + c[:,:,0]
#         x[:,:,4] = x[:,:,4] + c[:,:,1]
#         x[:,:,5] = x[:,:,5] + c[:,:,2]
#         x[:,:,6] = x[:,:,6] + c[:,:,3]
#     elif x.size()[-1] == 4: # 15 bundles (4 items)
#         x = torch.bmm(x, i.expand(x.size(0), -1, -1).transpose(1, 2))
#         x[:,:,4] = x[:,:,4] + c[:,:,0]
#         x[:,:,5] = x[:,:,5] + c[:,:,1]
#         x[:,:,6] = x[:,:,6] + c[:,:,2]
#         # x[:,:,7] = x[:,:,7] + c[:,:,3]
#         # x[:,:,8] = x[:,:,8] + c[:,:,4]
#         # x[:,:,9] = x[:,:,9] + c[:,:,5]
#         # x[:,:,10] = x[:,:,10] + c[:,:,6]
#         # x[:,:,11] = x[:,:,11] + c[:,:,7]
#         # x[:,:,12] = x[:,:,12] + c[:,:,8]
#         # x[:,:,13] = x[:,:,13] + c[:,:,9]
#         # x[:,:,14] = x[:,:,14] + c[:,:,10]
#     elif x.size()[-1] == 5:  # 31 bundles (5 items)
#         x = torch.bmm(x, i.expand(x.size(0), -1, -1).transpose(1, 2))
#         # Add the combinatorial constant to appropriate bundles
#         x[:, :, 5]  = x[:, :, 5]  + c[:, :, 0]  # Bundle 1
#         x[:, :, 6]  = x[:, :, 6]  + c[:, :, 1]  # Bundle 2
#         x[:, :, 7]  = x[:, :, 7]  + c[:, :, 2]  # Bundle 3
#         x[:, :, 8]  = x[:, :, 8]  + c[:, :, 3]  # Bundle 4
#         x[:, :, 9]  = x[:, :, 9]  + c[:, :, 4]  # Bundle 5
#         x[:, :, 10] = x[:, :, 10] + c[:, :, 5]  # Bundle 6
#         x[:, :, 11] = x[:, :, 11] + c[:, :, 6]  # Bundle 7
#         x[:, :, 12] = x[:, :, 12] + c[:, :, 7]  # Bundle 8
#         x[:, :, 13] = x[:, :, 13] + c[:, :, 8]  # Bundle 9
#         x[:, :, 14] = x[:, :, 14] + c[:, :, 9]  # Bundle 10
#         x[:, :, 15] = x[:, :, 15] + c[:, :, 10] # Bundle 11
#         x[:, :, 16] = x[:, :, 16] + c[:, :, 11] # Bundle 12
#         x[:, :, 17] = x[:, :, 17] + c[:, :, 12] # Bundle 13
#         x[:, :, 18] = x[:, :, 18] + c[:, :, 13] # Bundle 14
#         x[:, :, 19] = x[:, :, 19] + c[:, :, 14] # Bundle 15
#         x[:, :, 20] = x[:, :, 20] + c[:, :, 15] # Bundle 16
#         x[:, :, 21] = x[:, :, 21] + c[:, :, 16] # Bundle 17
#         x[:, :, 22] = x[:, :, 22] + c[:, :, 17] # Bundle 18
#         x[:, :, 23] = x[:, :, 23] + c[:, :, 18] # Bundle 19
#         x[:, :, 24] = x[:, :, 24] + c[:, :, 19] # Bundle 20
#         x[:, :, 25] = x[:, :, 25] + c[:, :, 20] # Bundle 21
#         x[:, :, 26] = x[:, :, 26] + c[:, :, 21] # Bundle 22
#         x[:, :, 27] = x[:, :, 27] + c[:, :, 22] # Bundle 23
#         x[:, :, 28] = x[:, :, 28] + c[:, :, 23] # Bundle 24
#         x[:, :, 29] = x[:, :, 29] + c[:, :, 24] # Bundle 25
#         x[:, :, 30] = x[:, :, 30] + c[:, :, 25] # Bundle 26
#     return x

def get_bundle_bid(x, c, i):
    # Repeat c to match x's batch dimension
    c = c.repeat(int(x.size(0) / c.size(0)), 1, 1)

    if isinstance(i, torch.Tensor):
        i = i.clone().detach().to(dtype=torch.float32, device=x.device)
    else:
        i = torch.tensor(i, dtype=torch.float32, device=x.device)
    x = torch.bmm(x, i.expand(x.size(0), -1, -1).transpose(1, 2))

    # Add constants from c to the tail of the bundle outputs
    start_idx = x.size(-1) - c.size(-1)  # Determine where constants go
    x[:, :, start_idx:start_idx + c.size(-1)] += c

    return x

def gen_incident_matrix(config):

    if config.cc2:
        if config.val_model == "mix_val":
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
            incident_matrix = np.broadcast_to(incident_matrix, (1, config.num_bundles, config.num_items))
            return incident_matrix #torch.tensor(incident_matrix).float().to(self.device)
    
    possible_bundles = list(itertools.product([0, 1], repeat=config.num_items))
    if config.airport_case:
        possible_bundles = [
            bundle for bundle in possible_bundles
            if any(bundle) and sum(bundle) % 2 == 0
        ]

    else:
        
        # Keep all non-empty bundles
        possible_bundles = [bundle for bundle in possible_bundles if any(bundle)]
    config.num_bundles = len(possible_bundles)   
    incident_matrix = np.array(possible_bundles)
    incident_matrix = np.broadcast_to(incident_matrix, (1, config.num_bundles, config.num_items))
    return incident_matrix #torch.tensor(incident_matrix).float().to(self.device)