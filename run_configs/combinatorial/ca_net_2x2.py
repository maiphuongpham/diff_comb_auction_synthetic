from copy import deepcopy

from core.configs.ca_2x2_uniform_config import cfg

cfg = deepcopy(cfg)

# copy params from base config
__C = cfg
__C.setting = "ca_2x2_uniform"

# Type of net - RegretNet, RegretFormer or EquivariantNet or CANet
__C.architecture = "CANet"
