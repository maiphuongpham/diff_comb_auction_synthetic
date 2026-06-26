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
__C.num_items = 2
__C.num_bundles = 3

# RegretFormer
# __C.net.pos_enc = True
# __C.net.pos_enc_part = 1
# __C.net.pos_enc_item = 3

# Distribution type - 'uniform_01' or 'uniform_416_47' or 'ca_uniform_12'
__C.distribution_type = "ca_uniform_12"
__C.min = 1
__C.max = 2
