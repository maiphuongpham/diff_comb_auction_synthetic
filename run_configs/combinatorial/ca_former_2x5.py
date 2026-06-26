from copy import deepcopy

from core.configs.ca_2x5_uniform_config import cfg

cfg = deepcopy(cfg)

# copy params from base config
__C = cfg
__C.setting = "ca_2x5_uniform"

# Type of net - RegretNet, RegretFormer or EquivariantNet or CANet
__C.architecture = "CANetFormer"

# Attention params
__C.net.hid_att = 4#16
__C.net.hid = 8#32