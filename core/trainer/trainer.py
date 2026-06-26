import itertools
import json
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import wandb
import random
from core.plot import plot
from core.utils import get_bundle_bid, gen_incident_matrix
from core.mechanisms import VCG
from core.trainer.metrics import jain_fairness_loss, fairness_metrics

# === Plot Heatmaps ===
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Trainer(object):
    def __init__(self, configuration, net, clip_op_lambda, device):
        self.net = net
        self.config = configuration

        #Change config:
        if hasattr(self.config.train, "fair_target_start"):
            self.config.train.fair_target_end = self.config.train.fair_target_start
        #------------------------------------
        self.device = device
        self.mode = "train"
        self.clip_op_lambda = clip_op_lambda

        self.writer = SummaryWriter(self.config.save_data)

        wandb.init(project="optimaler", config=self.config)

        self.init_componenents()

        if self.config.cc2 and self.config.val_model == "mix_val":
            self.get_bundle_bid = self._get_bundle_bid
        else:
            self.get_bundle_bid = get_bundle_bid
        
        self.incident_matrix = gen_incident_matrix(self.config)

    def _get_bundle_bid(self, x, c, i):
        c = c.repeat(int(x.size(0) / c.size(0)), 1, 1)
        if isinstance(i, torch.Tensor):
            i = i.clone().detach().to(dtype=torch.float32, device=x.device)
        else:
            i = torch.tensor(i, dtype=torch.float32, device=x.device)
        # print("i", i)
        x = torch.bmm(x, i.expand(x.size(0), -1, -1).transpose(1, 2))
        # x_3 = x.sum(dim=-1) + c[:, :, 0]
        # x = torch.cat([x, x_3.unsqueeze(-1)], dim=-1)
        x[:,:,3] = x[:,:,3] + c[:,:,0]
        x[:,:,4] = x[:,:,4] + c[:,:,1]
        x[:,:,5] = x[:,:,5] + c[:,:,2]
        return x

    def init_componenents(self):
        self.create_constants()

        self.create_params_to_train()

        self.create_optimizers()

        self.create_masks()

        self.save_config()

    def create_constants(self):
        self.x_shape = dict()
        self.x_shape["train"] = [self.config.train.batch_size, self.config.num_agents, self.config.num_items]
        self.x_shape["val"] = [self.config.val.batch_size, self.config.num_agents, self.config.num_items]

        self.adv_shape = dict()
        self.adv_shape["train"] = [
            self.config.num_agents,
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]
        self.adv_shape["val"] = [
            self.config.num_agents,
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]

        self.adv_var_shape = dict()
        self.adv_var_shape["train"] = [
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]
        self.adv_var_shape["val"] = [
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
            self.config.num_items,
        ]

        self.u_shape = dict()
        self.u_shape["train"] = [
            self.config.num_agents,
            self.config.train.num_misreports,
            self.config.train.batch_size,
            self.config.num_agents,
        ]
        self.u_shape["val"] = [
            self.config.num_agents,
            self.config.val.num_misreports,
            self.config.val.batch_size,
            self.config.num_agents,
        ]

        self.w_rgt = self.config.train.w_rgt_init_val
        self.rgt_target = self.config.train.rgt_target_start
        self.rgt_target_mult = (self.config.train.rgt_target_end / self.config.train.rgt_target_start) ** (
            1.5 / self.config.train.max_iter
        )
        if hasattr(self.config.train, "fair_target_start"):
            self.fair_target = self.config.train.fair_target_start
            self.fair_target_mult = (self.config.train.fair_target_end / self.config.train.fair_target_start) ** (
                1.5 / self.config.train.max_iter
            )
            self.w_fair = 0.5
        else:
            self.fair_target = 0.0
            self.fair_target_mult = 1.0
            self.w_fair = 0.0

    def create_params_to_train(self, train=True, val=True):
        # Trainable variable for find best misreport using gradient by inputs
        self.adv_var = dict()
        if train:
            self.adv_var["train"] = torch.zeros(
                self.adv_var_shape["train"], requires_grad=True, device=self.device
            ).float()
        if val:
            self.adv_var["val"] = torch.zeros(self.adv_var_shape["val"], requires_grad=True, device=self.device).float()

    def create_optimizers(self, train=True, val=True):
        self.opt1 = optim.Adam(self.net.parameters(), self.config.train.learning_rate)

        # Optimizer for best misreport find
        self.opt2 = dict()
        if train:
            self.opt2["train"] = optim.Adam([self.adv_var["train"]], self.config.train.gd_lr)
        if val:
            self.opt2["val"] = optim.Adam([self.adv_var["val"]], self.config.val.gd_lr)

        self.sc_opt2 = dict()
        if train:
            self.sc_opt2["train"] = optim.lr_scheduler.StepLR(self.opt2["train"], 1, self.config.train.gd_lr_step)
        if val:
            self.sc_opt2["val"] = optim.lr_scheduler.StepLR(self.opt2["val"], 1, self.config.val.gd_lr_step)

    def create_masks(self, train=True, val=True):
        self.adv_mask = dict()
        if train:
            self.adv_mask["train"] = np.zeros(self.adv_shape["train"])
            self.adv_mask["train"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0
            self.adv_mask["train"] = tensor(self.adv_mask["train"]).float()

        if val:
            self.adv_mask["val"] = np.zeros(self.adv_shape["val"])
            self.adv_mask["val"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0
            self.adv_mask["val"] = tensor(self.adv_mask["val"]).float()

        self.u_mask = dict()
        if train:
            self.u_mask["train"] = np.zeros(self.u_shape["train"])
            self.u_mask["train"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
            self.u_mask["train"] = tensor(self.u_mask["train"]).float()

        if val:
            self.u_mask["val"] = np.zeros(self.u_shape["val"])
            self.u_mask["val"][np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
            self.u_mask["val"] = tensor(self.u_mask["val"]).float()

    def save_config(self):
        print(self.writer.log_dir)
        print(type(self.config))
        with open(self.writer.log_dir + "/config.json", "w") as f:
            json.dump(self.config, f)

    def mis_step(self, x, c):
        """
        Find best misreport step using gradient by inputs, trainable inputs: self.adv_var variable.
        IMPORTANT: evaluate misreport utility under TRUE valuations (not the misreported ones).
        """
        mode = self.mode

        self.opt2[mode].zero_grad()

        # Get misreports
        x_mis, misreports = self.get_misreports_grad(x)

        # Run net for misreports
        a_mis, p_mis = self.net(misreports, c)
        x_mis = self.get_bundle_bid(x_mis, c, self.incident_matrix)
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)

        # Calculate loss value
        u_mis = -(utility_mis.view(self.u_shape[mode]) * self.u_mask[mode].to(self.device)).sum()

        # Make a step
        u_mis.backward()
        self.opt2[mode].step()
        self.sc_opt2[mode].step()




    def to_dollars_from_fraction(self, alloc, pay_norm, x_norm, x_dollar):
        """
        Convert normalized payments to dollar payments by using the
        model-implied fraction (pay_norm / allocated_value_norm),
        then applying that fraction to allocated dollar value.

        Shapes:
        alloc     : [B, A, M]
        pay_norm  : [B, A]
        x_norm    : [B, A, M]  (normalized valuations fed to net after get_bundle_bid)
        x_dollar  : [B, A, M]  (unnormalized/dollar valuations from net(..., return_true=True))
        Returns:
        pay_dollar       : [B, A]
        revenue_dollar   : scalar tensor
        welfare_dollar   : scalar tensor  (sum of allocated dollar values)
        """
        # allocated value in normalized space
        val_norm = (alloc * x_norm).sum(dim=-1)                          # [B, A]
        frac = (pay_norm / (val_norm + 1e-12)).clamp(min=0.0, max=1.0)   # [B, A]

        # allocated value in dollar space
        val_dollar = (alloc * x_dollar).sum(dim=-1)                       # [B, A]
        pay_dollar = frac * val_dollar                                    # [B, A]

        revenue_dollar = pay_dollar.sum(dim=-1).mean()                    # scalar
        welfare_dollar = val_dollar.sum(dim=-1).mean()                    # scalar
        return pay_dollar, revenue_dollar, welfare_dollar

    def compute_objective(self, revenue, welfare, fairness=None):
        """
        Compute the objective function based on revenue and welfare.
        """
        # Example: simple linear combination of revenue and welfare
        # You can modify this to fit your specific objective function
        if hasattr(self.config.train, 'objective'):
            if self.config.train.objective == "revenue":
                return revenue
            elif self.config.train.objective == "welfare":
                return welfare
            elif self.config.train.objective == "50-50":
                return 0.5 * revenue + 0.5 * welfare
            elif self.config.train.objective == "75-25":
                return 0.75 * revenue + 0.25 * welfare
            elif self.config.train.objective == "25-75":
                return 0.25 * revenue + 0.75 * welfare
            elif self.config.train.objective == "wel-fair":
                return welfare
        else:
            return revenue
    
    def train_op(self, x, c):
        self.opt1.zero_grad()

        x_mis, misreports = self.get_misreports(x)

        alloc_true, pay_true, _pay_true, _vals = self.net(x, c, return_true=True)
        a_mis, p_mis, _p_mis, _vals_mis = self.net(misreports, c, return_true=True)

        item_sum = alloc_true.detach().cpu().numpy() @ self.incident_matrix
        self.item_sum = item_sum.sum(axis=1, keepdims=True)

        x = self.get_bundle_bid(x, c, self.incident_matrix)
        x_mis = self.get_bundle_bid(x_mis, c, self.incident_matrix)

        rgt = self.compute_regret(x, alloc_true, pay_true, x_mis, a_mis, p_mis).sum()

        revenue = self.compute_rev(pay_true)
        welfare = (x * alloc_true).sum(dim=-1).sum(dim=-1).mean()

        u_batch = (alloc_true * x).sum(dim=-1) - pay_true
        u = u_batch.mean(dim=0)

        fairness_loss = jain_fairness_loss(u)

        _, rev_true_d, wel_true_d = self.to_dollars_from_fraction(
            alloc_true, pay_true, x, _vals
        )
        self._revenue = rev_true_d
        self._welfare = wel_true_d

        obj = self.compute_objective(revenue, welfare)

        g_t_rgt = (rgt / (self.config.train.rgt_ratio * obj + 1)).detach().log().item() - np.log(self.rgt_target)
        self.w_rgt = self.update_adaptive_weight("rgt", g_t_rgt, self.config.train.rgt_lr, self.rgt_target, self.ep)

        if hasattr(self.config.train, "fair_target_start"):
            g_t_fair = (fairness_loss / (self.config.train.fair_ratio * obj + 1)).detach().item() - self.fair_target
            self.w_fair = self.update_adaptive_weight("fair", g_t_fair, self.config.train.fair_lr, self.fair_target, self.ep)
        else:
            self.w_fair = 0.0

        final_loss = - 1 * (torch.log(1 + obj)) + self.w_rgt * torch.clamp(rgt, min=1e-12) + self.w_fair * fairness_loss

        ratio_penalty = torch.tensor(0.0, device=self.device)
        if getattr(self.config.train, "objective", None) in {"welfare", "25-75", "75-25", "50-50", "wel-fair"}:
            gamma = getattr(self.config.train, "rev_to_wel_ratio", 0.5)

            floor = F.relu(0.16 * wel_true_d - rev_true_d)
            cap   = F.relu(rev_true_d - 0.5 * wel_true_d)

            ratio_penalty = gamma * (floor**2 + cap**2)
            final_loss += torch.log(1+ratio_penalty)

        final_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.net.parameters(), 1)
        self.opt1.step()

        loss_component = {"welfare": 1 * (torch.log(1 + obj)), 
                          "rgt": self.w_rgt * torch.clamp(rgt, min=1e-12), 
                          "fairness": self.w_fair * fairness_loss,
                          "ratio_penalty": torch.log(1+ratio_penalty)}
        return final_loss, revenue, welfare, rgt, fairness_loss, loss_component

    def compute_metrics(self, x, c):
        x_mis, misreports = self.get_misreports_grad(x)

        alloc_true, pay_true =self.net (x, c)
        a_mis, p_mis = self.net(misreports, c)
        x = self.get_bundle_bid(x, c, self.incident_matrix)
        x_mis = self.get_bundle_bid(x_mis, c, self.incident_matrix)

        rgt = self.compute_regret_grad(x, alloc_true, pay_true, x_mis, a_mis, p_mis)

        revenue = self.compute_rev(pay_true)
        return revenue, rgt.mean()
    
    def compute_rev(self, pay):
        return pay.sum(dim=-1).mean()

    def compute_utility(self, x, alloc, pay):
        return (alloc * x).sum(dim=-1) - pay

    def compute_regret(self, x, a_true, p_true, x_mis, a_mis, p_mis):
        return self.compute_regret_grad(x, a_true, p_true, x_mis, a_mis, p_mis)

    def compute_regret_grad(self, x, a_true, p_true, x_mis, a_mis, p_mis):
        mode = self.mode

        utility = self.compute_utility(x, a_true, p_true)
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)

        utility_true = utility.repeat(self.config.num_agents * self.config[mode].num_misreports, 1)
        excess_from_utility = F.relu(
            (utility_mis - utility_true).view(self.u_shape[mode]) * self.u_mask[mode].to(self.device)
        )
        rgt = excess_from_utility.max(3)[0].max(1)[0].mean(dim=1)
        return rgt

    def get_misreports(self, x):
        return self.get_misreports_grad(x)

    def get_misreports_grad(self, x):
        mode = self.mode
        adv_mask = self.adv_mask[mode].to(self.device)

        adv = self.adv_var[mode].unsqueeze(0).repeat(self.config.num_agents, 1, 1, 1, 1)
        x_mis = x.repeat(self.config.num_agents * self.config[mode].num_misreports, 1, 1)
        x_r = x_mis.view(self.adv_shape[mode])
        y = x_r * (1 - adv_mask) + adv * adv_mask
        misreports = y.view([-1, self.config.num_agents, self.config.num_items])
        return x_mis, misreports

    def train(self, generator, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)

        self.train_gen, self.val_gen = generator

        iteration = self.config.train.restore_iter

        if iteration > 0:
            model_path = self.writer.log_dir + "/model_{}".format(iteration)
            state_dict = torch.load(model_path)
            self.net.load_state_dict(state_dict)

        max_ws_iters = getattr(self.config.train, "warm_start_iters", 0)
        print("Starting warm-start stage with {} iterations".format(max_ws_iters))
        ws_iter = 0
        while ws_iter < max_ws_iters:
            self.warm_start_epoch(ws_iter)
            ws_iter += 1
            if (ws_iter % 500) == 0:
                self.eval(ws_iter)
        print("Warm-start stage completed.")

        time_elapsed = 0.0

        while iteration < (self.config.train.max_iter):
            self.ep = iteration
            tic = time.time()
            self.train_epoch(iteration)

            toc = time.time()
            time_elapsed += toc - tic

            iteration += 1
            self.writer.add_scalar("Train/epoch time", time_elapsed, iteration / 1000)

            wandb.log({"epoch time": time_elapsed}, step=iteration)

            if (iteration + 1) % self.config.train.save_iter == 0:
                self.save(iteration + 1)

            if (iteration % self.config.val.print_iter) == 0:
                self.eval(iteration)

    def warm_start_epoch(self, iteration):
        self.mode = "train"
        self.net.train()

        X, _, C, perm = next(self.train_gen.gen_func)

        x = torch.from_numpy(X).float().to(self.device)
        c = torch.from_numpy(C).float().to(self.device)

        alloc, pay = self.net(x, c)

        x = self.get_bundle_bid(x, c, self.incident_matrix)
        alloc_vcg, pay_vcg = VCG(x, incident_matrix=self.incident_matrix)

        loss_alloc = F.mse_loss(alloc, alloc_vcg)
        loss_pay = F.mse_loss(pay, pay_vcg)

        loss = 10*(10*loss_alloc + loss_pay)
        self.opt1.zero_grad()
        loss.backward()
        self.opt1.step()

        if (iteration % 50) == 0:
            print("Warm-start iteration {}".format(iteration))
            print("Alloc: {}".format(alloc[0].detach().cpu().numpy()))
            print("Pay: {}".format(pay[0].detach().cpu().numpy()))
            print("VCG Alloc: {}".format(alloc_vcg[0].detach().cpu().numpy()))
            print("VCG Pay: {}".format(pay_vcg[0].detach().cpu().numpy()))
            print("Loss: {}".format(loss.item()))

    def train_epoch(self, iteration):
        self.mode = "train"
        self.net.train()

        X, ADV, C, perm = next(self.train_gen.gen_func)

        x = torch.from_numpy(X).float().to(self.device)

        c = torch.from_numpy(C).float().to(self.device)
        self.adv_var["train"].data = torch.as_tensor(ADV, dtype=torch.float32, device=self.device)

        self.misreport_cycle(x, c)

        if self.config.train.data == "fixed" and self.config.train.adv_reuse:
            self.train_gen.update_adv(perm, self.adv_var["train"].data.cpu())
        net_loss, train_revenue, train_welfare, train_regret, train_fairness, loss_component = self.train_op(x, c)
        self.rgt_target = max(self.rgt_target * self.rgt_target_mult, self.config.train.rgt_target_end)
        if hasattr(self.config.train, "fair_target_start"):
            self.fair_target = max(self.fair_target * self.fair_target_mult, self.config.train.fair_target_end)

        if (iteration % self.config.train.print_iter) == 0:
            print("Iteration {}".format(iteration))
            print(
                "Train revenue: {}, Train welfare: {},  regret: {}, fairness: {},  net loss: {} , w: {}".format(
                    round(float(train_revenue), 5),
                    round(float(train_welfare), 5),
                    round(float(train_regret), 5),
                    round(float(train_fairness), 5),
                    round(float(net_loss), 5),
                    round(float(self.w_rgt.item()), 4),
                )
            )
            print(loss_component)
            print("Unnormalized revenue: {}, welfare: {}".format(
                round(float(self._revenue), 5),
                round(float(self._welfare), 5)
            ))
            
            print("item-wise allocation: {}".format(
                self.item_sum[0])
            )

            wandb.log({"train_revenue": self._revenue, "train_welfare": self._welfare, 
                       "train_fairness": train_fairness,"train_regret": train_regret, 
                       "net_loss": net_loss,
                       "w_rgt": self.w_rgt, "w_fair": self.w_fair}, step=iteration)

    def misreport_cycle(self, x, c):
        mode = self.mode

        for _ in range(self.config[mode].gd_iter):
            self.mis_step(x, c)

            self.clip_op_lambda(self.adv_var[mode])

        for param_group in self.opt2[mode].param_groups:
            param_group["lr"] = self.config[mode].gd_lr

        self.opt2[mode].state = defaultdict(dict)

    def save(self, iteration):
        torch.save(self.net.state_dict(), self.writer.log_dir + "/model_{}".format(iteration))

    def eval(self, iteration):
        print("Validation on {} iteration".format(iteration))
        self.mode = "val"
        self.net.eval()

        self.eval_grad(iteration)

        if self.config.plot.bool:
            self.plot()

    def eval_grad(self, iteration):
        val_revenue = 0
        val_regret = 0

        unnormed_val_revenue = 0
        unnormed_val_welfare = 0

        all_alloc_true, all_alloc_mis, all_alloc_greedy = [], [], []
        all_pay_true, all_pay_true_d = [], []

        vcg_revenue = 0
        vcg_welfare = 0

        fairness_sums = {k: 0.0 for k in ["jain_index", "cv", "gini", "soft_min_utility", "1_minus_jain"]}
        vcg_fairness_sums = {k: 0.0 for k in fairness_sums.keys()}

        for batch_idx in range(self.config.val.num_batches):
            X, ADV, C, _ = next(self.val_gen.gen_func)
            x = torch.from_numpy(X).float().to(self.device)
            c = torch.from_numpy(C).float().to(self.device)
            adv = torch.as_tensor(ADV, dtype=torch.float32, device=self.device)

            self.adv_var["val"].data = adv
            self.misreport_cycle(x, c)

            x_val = self.get_bundle_bid(x, c, self.incident_matrix)

            x_mis_val, misreports = self.get_misreports_grad(x)
            x_mis_val = x_mis_val.view(
                x.shape[0] * self.config.num_agents * self.config.val.num_misreports,
                self.config.num_agents,
                self.config.num_items,
            )
            c_repeat = c.repeat(x_mis_val.size(0) // c.size(0), 1, 1)
            x_mis_val = self.get_bundle_bid(x_mis_val, c_repeat, self.incident_matrix)

            with torch.no_grad():
                alloc_true, pay_true, pay_true_d, x_true_d = self.net(x, c, return_true=True)
                alloc_mis, pay_mis, pay_mis_d, _ = self.net(misreports, c, return_true=True)

            pay_true_d, rev_true_d, wel_true_d = self.to_dollars_from_fraction(
                alloc_true, pay_true, x_val, x_true_d
            )

            all_alloc_true.append(alloc_true.detach().cpu().numpy())
            all_alloc_mis.append(alloc_mis.detach().cpu().numpy())
            all_alloc_greedy.append(torch.argmax(x_val, dim=-1).detach().cpu().numpy())
            all_pay_true.append(pay_true.detach().cpu().numpy())
            all_pay_true_d.append(pay_true_d.detach().cpu().numpy())

            u_val = ((alloc_true * x_val).sum(dim=-1) - pay_true).mean(dim=0)
            fairness_dict = fairness_metrics(u_val)
            for k in fairness_sums.keys():
                fairness_sums[k] += fairness_dict[k]

            alloc_vcg, pay_vcg = VCG(x_val, incident_matrix=self.incident_matrix)
            u_vcg = ((alloc_vcg * x_val).sum(dim=-1) - pay_vcg).mean(dim=0)
            vcg_fair_dict = fairness_metrics(u_vcg)
            for k in vcg_fairness_sums.keys():
                vcg_fairness_sums[k] += vcg_fair_dict[k]

            val_revenue += self.compute_rev(pay_true)
            val_regret += self.compute_regret_grad(x_val, alloc_true, pay_true, x_mis_val, alloc_mis, pay_mis).mean()

            unnormed_val_revenue += rev_true_d
            unnormed_val_welfare += wel_true_d

            pay_vcg_d, rev_vcg_d, wel_vcg_d = self.to_dollars_from_fraction(
                alloc_vcg, pay_vcg, x_val, x_true_d
            )
            vcg_revenue += rev_vcg_d
            vcg_welfare += wel_vcg_d

        n_batches = float(self.config.val.num_batches)
        val_revenue /= n_batches
        val_regret /= n_batches
        unnormed_val_revenue /= n_batches
        unnormed_val_welfare /= n_batches
        vcg_revenue /= n_batches
        vcg_welfare /= n_batches

        avg_fairness = {k: v / n_batches for k, v in fairness_sums.items()}
        avg_vcg_fairness = {k: v / n_batches for k, v in vcg_fairness_sums.items()}

        print("\n=== Validation Summary ===")
        print(f"Val revenue (norm): {val_revenue:.5f}, regret_grad (norm): {val_regret:.5f}")
        print(f"Val revenue ($): {unnormed_val_revenue:.2f}, Welfare ($): {unnormed_val_welfare:.2f}")
        print(f"[VAL] VCG revenue ($): {vcg_revenue:.2f}, Welfare ($): {vcg_welfare:.2f}")

        print("\n=== Fairness Metrics (Model) ===")
        for k, v in avg_fairness.items():
            print(f"{k}: {v:.4f}")

        print("\n=== Fairness Metrics (VCG) ===")
        for k, v in avg_vcg_fairness.items():
            print(f"{k}: {v:.4f}")

        avg_alloc_true = np.mean(np.concatenate(all_alloc_true, axis=0), axis=0)
        avg_payment_true = np.mean(np.concatenate(all_pay_true, axis=0), axis=0)
        avg_payment_true_d = np.mean(np.concatenate(all_pay_true_d, axis=0), axis=0)

        print("\n=== Average Allocation Across Batches ===")
        print(avg_alloc_true)
        print("\n=== Average Normalized Payments Across Batches ===")
        print(avg_payment_true)
        print("\n=== Average Dollar Payments Across Batches ===")
        print(avg_payment_true_d)

        wandb.log({
            "val_revenue": val_revenue,
            "val_regret": val_regret,
            "val_unnormed_revenue": unnormed_val_revenue,
            "val_unnormed_welfare": unnormed_val_welfare,
            **{f"fairness/{k}": v for k, v in avg_fairness.items()},
            **{f"fairness_vcg/{k}": v for k, v in avg_vcg_fairness.items()}
        }, step=iteration)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.alloc_net.load_state_dict(checkpoint['alloc_net'])
        self.pay_net.load_state_dict(checkpoint['pay_net'])

        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def update_adaptive_weight(self, name, grad_value, lr, target, step):
        """
        Generic Adam-style adaptive weight update for any auxiliary loss term.
        
        Args:
            name (str): e.g. 'rgt' or 'fair'
            grad_value (float): log-scale gradient signal
            lr (float): learning rate for this adaptive weight
            target (float): desired target value (e.g. regret or fairness target)
            step (int): current iteration index
        Returns:
            weight (torch.Tensor): updated adaptive weight
        """
        # Initialize state if missing
        m_name, v_name, w_name = f"m_w_{name}", f"v_w_{name}", f"w_{name}"
        if not hasattr(self, m_name):
            setattr(self, m_name, torch.tensor(0., device=self.device))
            setattr(self, v_name, torch.tensor(0., device=self.device))
            setattr(self, w_name, torch.tensor(0.5, device=self.device))

        m = getattr(self, m_name)
        v = getattr(self, v_name)
        w = getattr(self, w_name)

        # Compute moments
        m = 0.9 * m + 0.1 * grad_value
        v = 0.999 * v + 0.001 * (grad_value ** 2)

        # Bias correction
        m_hat = m / (1 - 0.9 ** (step + 1))
        v_hat = v / (1 - 0.999 ** (step + 1))

        # Adam-style parameter update
        w = torch.clamp(
            w + lr * m_hat / (torch.sqrt(v_hat) + 1e-8),
            min=0.1, max=5.0
        )

        # Save back
        setattr(self, m_name, m)
        setattr(self, v_name, v)
        setattr(self, w_name, w)

        return w

    