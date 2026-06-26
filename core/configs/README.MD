# Description of configuration parameters

## Configuration file structure

1. The base configuration file is `default_config`. It describes all the parameters for running experiments and is used as a starting point of all higher-level configs. Some of the parameters are overwritten in the higher-level configs.
2. Auction-type configuration files - all other configs in this folder. They overwrite some parameters from the base config by describing a specific auction -- number of items and participants, bid distribution type, and neural network hyperparameters.
3. Ready-to-run configuration files - all configs from `run_configs` folder. They overwrie some parameters from the auction-type configs, i.e. the type of model to use (`architecture`).


## Parameters
`architecture` - type of the network: "CANet", or "CAFormer". [Default: None, specified in ready-to-run config]

#### Auction parameters

`num_agents` - the number of participants in an auction [Default: None, specified in auction-type config]

`num_items` - the number of items in an auction [Default: None, specified in auction-type config]

`num_bundles` - the number of bundles in an auction [Default: None, specified in auction-type config]

`distribution_type` - type of the bid distribution, "ca_uniform_12" (setting (B) in the paper), "ca_uniform_01" (setting (A) in the paper) or "ca_uniform_12_15" (setting (C) in the paper) [Default: None, specified in auction-type config]

`min` - min possible value of bid [Default: None, specified in auction-type config]

`max` - max possible value of bid [Default: None, specified in auction-type config]


#### Train process parameters

`train.seed` - random seed fixation [Default: 0]

`train.restore_iter` - iter from which training begins. Set restore_iter = 0 for random network initialization. Set restore_iter > 0 for starting training from restore_iter (requires a saved model) [Default: 0]

`train.max_iter` - number of training iterations (1 iteration = optimizer step on 1 batch) [Default: 100000]

`train.learning_rate` - learning rate for optimizer [Default: 4e-4]

`train.data` - fixed or online generated data. If online, set adv_reuse to False [Default: "fixed"]

`train.num_batches` - number of batches sampled to generate the training dataset [Default: 1250]

`train.batch_size` - size of a single training batch [Default: 512]

`train.adv_reuse` - cache misreports after misreport optimization [Default: True]

`train.num_misreports` - number of misreport initializations during training [Default: 1]

`train.gd_iter` - number of gradient steps for misreport computation during training [Default: 50]

`train.gd_lr` - learning rate of gradient descent for misreport computation during training [Default: 0.1]

`train.gd_lr_step` - learning rate step size (scheduler) for misreport computation during training; no scheduling if set to one [Default: 1]

`train.w_rgt_init_val` - initial value of the regret weight in the loss function (Lagrangian coefficient $\gamma$) [Default: 1]

`train.rgt_target_start` - target regret value (budget) at the beginning of training [Default: 0.05]

`train.rgt_target_end` - target regret value (budget) by the end of training; the budget exponentially converges from rgt_target_start to rgt_target_end during first 2/3 of training; set rgt_target_start to rgt_target_end to turn target regret scheduling off (may hinder performance for low rgt_target_end values) [Default: 0.001]

`train.rgt_lr` - learning rate of dual gradient descent; specifies how quickly Lagrange coefficient $\gamma$ is updated for the network to match the regret budget [Default: 0.01]

`train.rgt_ratio` - the target regret - revenue adjustment factor [Default: 1]

`train.save_iter` - frequency at which the models are saved [Default: 100000]

`train.print_iter` - print frequency of statistics during training [Default: 100]


#### Validation params

`val.num_misreports` - number of misreport initializations during validation [Default: 1]

`val.gd_iter` - number of gradient steps for misreport computation during validation (typically higher than during training for higher precision) [Default: 1000]

`val.gd_lr` - learning rate of gradient descent for misreport computation during validation [Default: 0.1]

`val.gd_lr_step` - learning rate step size (scheduler) for misreport computation during valudation [Default: 1]

`val.num_batches` - number of batches sampled to generate the validation dataset [Default: 128]

`val.batch_size` - size of a single validation batch [Default: 32]

`val.print_iter` - frequency at which validation is performed and the results are printed [Default: 25000]

`val.data` - fixed data or online generated. If online, set adv_reuse to False [Default: "online"]


#### Neural networks architecture
All neural networks parameters have default values specified in `default.cfg` and might be rewritten in auction-type configs.

##### Common parameters

`net.init` - weights initialization, (g - glorot, h - he) + (u - uniform, n - normal) [Default: "gu"]

##### CAFormer

`net.hid_att` - dimensionality of linear projections of each head in multi-head self-attention [Default: 32]

`net.hid` - hidden size of all layers in RegretFormer [Default: 64]

`net.n_attention_layers` - number of attention layers [Default: 1]

`net.n_attention_heads` - number of heads in each attention layer [Default: 2]

`net.activation_att` - activation function ("relu" or "tanh") [Default: "tanh"]

`net.pos_enc` - boolean, whether to apply positional encoding; should be used to learn auctions with asymmetric participants or items; PE from Vaswani et al. (2017) is applied [Default: False]

`net.pos_enc_part` - Number of participants to encode the positions of; should be set to (the maximal) number of participants to learn auctions with asymmetric participants [Default: 1]

`net.pos_enc_item` - Number of bundles to encode the positions of; should be set to (the maximal) number of bundles to learn auctions with asymmetric bundles [Default: 1]


##### CANet

`net.activation` - activation function ("tanh", "sigmoid", "relu") [Default: "tanh"]

`net.num_a_layers` - number of fully-connected layers in the allocation network [Default: 3]

`net.num_a_hidden_units` - hidden size of each allocation layer [Default: 100]

`net.num_p_layers` - number of fully-connected layers in the payment network  [Default: 3]

`net.num_p_hidden_units` - hidden size of each payment layer [Default: 100]

`net.layer_norm` - boolean, whether to apply nn.LayerNorm to outputs of each layer [Default: False]

`net.batch_norm` - boolean, whether to apply nn.BatchNorm1d to outputs of each layer [Default: False]

#### Additional parameters
`C.temp` - the softmax temperature [Default: 15]