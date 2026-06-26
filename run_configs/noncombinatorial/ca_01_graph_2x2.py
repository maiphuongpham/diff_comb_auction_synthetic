from copy import deepcopy

from core.configs.ca01_2x2_uniform_config import cfg

cfg = deepcopy(cfg)

# copy params from base config
__C = cfg
__C.setting = "ca01_2x2_uniform"

# Type of net - RegretNet, RegretFormer or EquivariantNet or CANet
__C.architecture = "CAGCN"

__C.net.hid = 128
__C.train.batch_size = 128
__C.train.learning_rate = 0.001
__C.train.rgt_target_end = 0.00005
__C.train.print_iter = 100