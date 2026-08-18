import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.HWVA import HWVA
import math


class clientFedLMI(object):
    def __init__(self, args, id, train_samples, test_samples):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.is_first = True
        self.disable_hwva = args.disable_hwva

        self.loss = nn.CrossEntropyLoss()

        self.min_samples = None
        self.max_samples = None

        self.mapping_type = args.mapping_type
        self.wd_base = args.wd_base
        self.dp_base = args.dp_base

        self.current_weight_decay = self.compute_adaptive_weight_decay_base()
        self.current_dropout_rate = self.compute_adaptive_dropout_base()

        self.sample_ratio = args.sample_ratio
        self.current_sample_indices = None

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.learning_rate,
                                         weight_decay=self.current_weight_decay)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx

        self.full_train_data = read_client_data(self.dataset, self.id, is_train=True)
        self.HWVA = HWVA(self.id, self.loss, self.full_train_data, self.batch_size, self.rand_percent, self.layer_idx, self.eta, self.device)

    def load_sampled_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        total_samples = len(self.full_train_data)
        sample_size = int(total_samples * self.sample_ratio)

        indices = np.random.choice(total_samples, sample_size, replace=False)
        self.current_sample_indices = indices

        sampled_data = Subset(self.full_train_data, indices)

        return DataLoader(sampled_data, batch_size, drop_last=True, shuffle=True)

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.full_train_data, batch_size, drop_last=True, shuffle=False)

    def train(self):
        if self.is_first is True:
            self.update_model_dropout()
            self.update_optimizer()
            self.is_first = False
        trainloader = self.load_sampled_train_data()
        self.model.train()

        for step in range(self.local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

    def local_initialization(self, received_global_model):
        if not self.disable_hwva:
            self.HWVA.hierarchical_local_aggregation(received_global_model, self.model)
        else:
            local_bn_params = {}
            for name, param in self.model.named_parameters():
                if 'bn' in name:
                    local_bn_params[name] = copy.deepcopy(param.data)

            for name, global_param in received_global_model.named_parameters():
                if 'bn' not in name:
                    if name in self.model.state_dict():
                        self.model.state_dict()[name].copy_(global_param.data)
                    else:
                        print(f"Warning: local model missing non-BN param {name}")

            for name, param in self.model.named_parameters():
                if 'bn' in name and name in local_bn_params:
                    param.data = local_bn_params[name]

    def load_test_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model is None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output, dim=1).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self, model=None):
        trainloader = self.load_train_data()
        if model is None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def update_model_dropout(self):
        if self.mapping_type == "base":
            self.current_dropout_rate = self.compute_adaptive_dropout_base()
        elif self.mapping_type == "line":
            self.current_dropout_rate = self.compute_adaptive_dropout_line()
        elif self.mapping_type == "log":
            self.current_dropout_rate = self.compute_adaptive_dropout_log()
        elif self.mapping_type == "exp":
            self.current_dropout_rate = self.compute_adaptive_dropout_exp()

        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.p = self.current_dropout_rate
        print(f"Client {self.id} 样本量 {self.train_samples}，调整 dropout 率为 {self.current_dropout_rate:.2f}")

    def update_optimizer(self):
        if self.mapping_type == "base":
            self.current_weight_decay = self.compute_adaptive_weight_decay_base()
        elif self.mapping_type == "line":
            self.current_weight_decay = self.compute_adaptive_weight_decay_line()
        elif self.mapping_type == "log":
            self.current_weight_decay = self.compute_adaptive_weight_decay_log()
        elif self.mapping_type == "exp":
            self.current_weight_decay = self.compute_adaptive_weight_decay_exp()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.current_weight_decay
        )

        print(f"Client {self.id} 样本量 {self.train_samples}，调整权重衰减为 {self.current_weight_decay:.6f}")

    def compute_adaptive_weight_decay_base(self):
        return self.wd_base

    def compute_adaptive_dropout_base(self):
        return self.dp_base

    def compute_adaptive_weight_decay_line(self):
        if self.max_samples == self.min_samples:
            return self.wd_base * 0.5

        sample_ratio = (self.train_samples - self.min_samples) / (self.max_samples - self.min_samples)
        wd = self.wd_base * (1 - sample_ratio)
        return max(1e-8, wd)

    def compute_adaptive_dropout_line(self):
        if self.max_samples == self.min_samples:
            return self.dp_base * 0.5

        sample_ratio = (self.train_samples - self.min_samples) / (self.max_samples - self.min_samples)
        dp = self.dp_base * (1 - sample_ratio)
        return max(0.0, min(1.0, dp))

    def compute_adaptive_weight_decay_log(self):
        if self.max_samples == self.min_samples:
            return self.wd_base * 0.5

        if self.min_samples <= 0:
            self.min_samples = 1

        relative_samples = self.train_samples / self.min_samples
        log_relative = math.log(relative_samples + 1e-8)

        max_relative = self.max_samples / self.min_samples
        max_log_relative = math.log(max_relative + 1e-8)

        log_sample_ratio = log_relative / max_log_relative
        log_sample_ratio = max(0.0, min(1.0, log_sample_ratio))

        wd = self.wd_base * (1 - log_sample_ratio)
        return max(1e-8, wd)

    def compute_adaptive_dropout_log(self):
        if self.max_samples == self.min_samples:
            return self.dp_base * 0.5

        if self.min_samples <= 0:
            self.min_samples = 1

        relative_samples = self.train_samples / self.min_samples
        log_relative = math.log(relative_samples + 1e-8)

        max_relative = self.max_samples / self.min_samples
        max_log_relative = math.log(max_relative + 1e-8)

        log_sample_ratio = log_relative / max_log_relative
        log_sample_ratio = max(0.0, min(1.0, log_sample_ratio))

        dp = self.dp_base * (1 - log_sample_ratio)
        return max(0.0, min(1.0, dp))

    def compute_adaptive_weight_decay_exp(self):
        if self.max_samples is None or self.min_samples is None or self.max_samples == self.min_samples:
            return self.wd_base * 0.5

        min_samples = max(self.min_samples, 1)
        current_samples = max(self.train_samples, 1)

        relative_samples = current_samples / min_samples
        log_relative = math.log(relative_samples + 1e-8)

        max_relative = self.max_samples / min_samples
        max_log_relative = math.log(max_relative + 1e-8)

        if max_log_relative == 0:
            log_sample_ratio = 0.0
        else:
            log_sample_ratio = log_relative / max_log_relative
            log_sample_ratio = max(0.0, min(1.0, log_sample_ratio))

        wd = self.wd_base * (1 - log_sample_ratio)
        return max(1e-8, wd)

    def compute_adaptive_dropout_exp(self):
        if self.max_samples is None or self.min_samples is None or self.max_samples == self.min_samples:
            return self.dp_base * 0.5

        min_samples = max(self.min_samples, 1)
        current_samples = max(self.train_samples, 1)

        relative_samples = current_samples / min_samples
        log_relative = math.log(relative_samples + 1e-8)

        max_relative = self.max_samples / min_samples
        max_log_relative = math.log(max_relative + 1e-8)

        if max_log_relative == 0:
            log_sample_ratio = 0.0
        else:
            log_sample_ratio = log_relative / max_log_relative
            log_sample_ratio = max(0.0, min(1.0, log_sample_ratio))

        dp = self.dp_base * (1 - log_sample_ratio)
        return max(0.0, min(1.0, dp))