import os
from copy import deepcopy

from core.configs.default_config import cfg
from core.clip_ops.clip_ops import *
from core.data import *

cfg = deepcopy(cfg)
__C = cfg

# Plot
__C.plot.bool = False

# Auction params
__C.num_agents = 2
__C.num_items = 5
__C.num_bundles = 31

# RegretNet
__C.net.num_a_layers = 6
__C.net.num_p_layers = 6

# RegretFormer
__C.net.pos_enc = True
__C.net.pos_enc_part = 1
__C.net.pos_enc_item = 31

# Distribution type - 'uniform_01' or 'uniform_416_47' or 'ca_uniform_12' or ca_uniform_01
__C.distribution_type = "ca_uniform_01"
__C.min = 0
__C.max = 1
