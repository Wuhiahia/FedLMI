import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple


class HWVA:
    def __init__(self,
                 cid: int,
                 loss: nn.Module,
                 train_data: List[Tuple],
                 batch_size: int,
                 rand_percent: int,
                 layer_idx: int = 0,
                 eta: float = 1.0,
                 device: str = 'cpu',
                 threshold: float = 0.1,
                 num_pre_loss: int = 10) -> None:
        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device
        self.weights = None
        self.start_phase = True

    def hierarchical_local_aggregation(self,
                                       global_model: nn.Module,
                                       local_model: nn.Module) -> None:
        for (name_g, p_g), (name_l, p_l) in zip(global_model.named_parameters(), local_model.named_parameters()):
            if p_g.size() != p_l.size():
                print(f"Size mismatch: Global {name_g} {p_g.size()} vs Local {name_l} {p_l.size()}")

        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data) - rand_num)
        rand_loader = DataLoader(self.train_data[rand_idx:rand_idx + rand_num], self.batch_size, drop_last=False)

        params_g = []
        params = []
        param_names = []

        for name, p_g in global_model.named_parameters():
            if 'bn' not in name:
                params_g.append(p_g)
                param_names.append(name)

        for name, p in local_model.named_parameters():
            if 'bn' not in name:
                params.append(p)

        if torch.sum(params_g[0] - params[0]) == 0:
            return

        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        model_t = copy.deepcopy(local_model)
        params_t = [p for name, p in model_t.named_parameters() if 'bn' not in name]

        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        for name, param in model_t.named_parameters():
            if 'bn' not in name and name in [n for n, _ in model_t.named_parameters()][:-self.layer_idx]:
                param.requires_grad = False

        optimizer = torch.optim.SGD(params_tp, lr=0)

        if self.weights is None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
            param_t.data = param + (param_g - param) * weight

        losses = []
        cnt = 0
        while True:
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y)
                loss_value.backward()

                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1

            if not self.start_phase:
                break

            if (len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold) or cnt >= 300:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]), '\tHWVA epochs:', cnt)
                break
            if cnt % 13 == 0:
                print(f"client{self.cid}：{cnt}th")

        self.start_phase = False

        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()