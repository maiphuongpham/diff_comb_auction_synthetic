from copy import deepcopy

from core.configs.ca1215_2x3_uniform_config import cfg

cfg = deepcopy(cfg)

# copy params from base config
__C = cfg
__C.setting = "ca1215_2x3_uniform"

# Type of net - RegretNet, RegretFormer or EquivariantNet or CANet
__C.architecture = "CAGCN"

__C.net.hid = 128
__C.train.batch_size = 128
__C.train.learning_rate = 0.001
__C.train.rgt_target_end = 0.0001
__C.train.print_iter = 100