from copy import deepcopy

from core.configs.ca01_2x5_uniform_config import cfg

cfg = deepcopy(cfg)

# copy params from base config
__C = cfg
__C.setting = "ca01_2x5_uniform"

# Type of net - RegretNet, RegretFormer or EquivariantNet or CANet
__C.architecture = "CANetFormer"
