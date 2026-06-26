import torch

from core.nets.ca_net import CANet
from core.nets.ca_net_attention import CANetFormer
from core.nets.ca_graph import CAGCN
from core.trainer.trainer import Trainer
from core.utils import get_objects


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(setting):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"DEVICE: {device}")

    cfg, clip_op, generator, setting_name = get_objects(setting)
    cfg.setting = cfg.get('setting', setting_name)
    seed = cfg.train.seed
    if cfg.save_data is not None:
        cfg.save_data = cfg.save_data.split("/setting")[0]
        cfg.save_data = cfg.save_data + f"/setting_{cfg.setting}/seed_{seed}"

    if cfg.architecture == "CANet":
        net = CANet(cfg, device).to(device)
    elif cfg.architecture == "CANetFormer":
        net = CANetFormer(cfg, device).to(device)
    elif cfg.architecture == "CAGCN":
        net = CAGCN(cfg, device).to(device)
    print("number of parameters, net =", count_parameters(net))
    generators = [generator(cfg, "train"), generator(cfg, "val")]
    trainer = Trainer(cfg, net, clip_op, device)

    trainer.train(generators, seed)
