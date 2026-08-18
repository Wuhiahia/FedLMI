import copy
import statistics
import numpy as np
import torch
import time
from flcore.clients.clientFedLMI import clientFedLMI
from utils.DDWA import DDWAModule
from utils.result_utils import save_results
from utils.data_utils import read_client_data
from threading import Thread


class FedLMI(object):
    def __init__(self, args, times):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)

        self.clients = []
        self.selected_clients = []

        self.disable_ddwa = args.disable_ddwa

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_train_loss = []
        self.rs_test_auc = []
        self.rs_std_acc = []
        self.rs_std_auc = []
        self.current_upload_size = []

        self.hn = args.h5_name

        self.times = times
        self.eval_gap = args.eval_gap
        self.set_clients(args, clientFedLMI)

        all_samples = [client.train_samples for client in self.clients]
        self.min_samples = min(all_samples)
        self.max_samples = max(all_samples)

        for client in self.clients:
            client.min_samples = self.min_samples
            client.max_samples = self.max_samples

        self.ddwa = DDWAModule(num_clients=self.num_clients, device=self.device)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

        save_results(acc=self.rs_test_acc, std_acc=self.rs_std_acc, auc=self.rs_test_auc, std_auc=self.rs_std_auc, loss=self.rs_train_loss, h5name=self.hn, times=self.times)

        print(f"\nBest global accuracy {max(self.rs_test_acc)}")
        print(f'avg time cost {sum(self.Budget[1:]) / len(self.Budget[1:])}')
        print(f'upload size Avg:{statistics.mean(self.current_upload_size) / (1024 * 1024):.2f}MB')

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data))
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.local_initialization(self.global_model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        total_upload_size = 0
        param_size = 4

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples
            client_param_count = 0
            for name, param in client.model.named_parameters():
                if 'bn' not in name:
                    client_param_count += param.numel()
            client_upload_size = client_param_count * param_size
            total_upload_size += client_upload_size
        print(f"第{len(self.Budget)}轮上传总参数量：{total_upload_size / (1024 * 1024):.2f} MB")
        self.current_upload_size.append(total_upload_size)
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        current_round = len(self.Budget)
        print(f"开始第{current_round}轮聚合，客户端数量: {len(self.uploaded_models)}")

        if self.disable_ddwa:
            print("DDWA模块已禁用，使用样本数量加权聚合")
            return self._aggregate_by_sample_size()

        if current_round == 0:
            print("第一轮使用样本数量加权聚合")
            return self._aggregate_by_sample_size()
        elif current_round < 3:
            print(f"第{current_round}轮使用保守DDWA聚合")
            return self._conservative_ddwa_aggregation()
        else:
            print(f"第{current_round}轮使用完整DDWA聚合")
            return self._full_ddwa_aggregation()

    def _aggregate_by_sample_size(self):
        total_samples = sum(client.train_samples for client in self.selected_clients)

        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for i, client in enumerate(self.selected_clients):
            weight = client.train_samples / total_samples
            for (name, client_param), global_param in zip(
                    self.uploaded_models[i].named_parameters(),
                    self.global_model.parameters()):
                if 'bn' not in name:
                    global_param.data += client_param.data.clone() * weight

        print(f"样本数量加权聚合完成，权重范围: [{min([c.train_samples / total_samples for c in self.selected_clients]):.3f}, "
              f"{max([c.train_samples / total_samples for c in self.selected_clients]):.3f}]")

    def _conservative_ddwa_aggregation(self):
        return self._ddwa_aggregation_with_settings(max_iter=20, lr=0.001, eps=1e-6)

    def _full_ddwa_aggregation(self):
        return self._ddwa_aggregation_with_settings(max_iter=100, lr=0.005, eps=1e-8)

    def _ddwa_aggregation_with_settings(self, max_iter=50, lr=0.01, eps=1e-8):
        G = self._compute_safe_deviations(eps)
        if G is None:
            print("偏差计算失败，回退到样本数量加权")
            return self._aggregate_by_sample_size()

        p = self._solve_nash_bargaining_safe(G, max_iter, lr, eps)
        if p is None:
            print("Nash权重求解失败，回退到样本数量加权")
            return self._aggregate_by_sample_size()

        self._apply_aggregation_with_weights(p)
        return True

    def _compute_safe_deviations(self, eps=1e-8):
        try:
            deviations = []
            for i, client_model in enumerate(self.uploaded_models):
                dev_vectors = []
                for (name_g, g_param), (name_l, c_param) in zip(
                        self.global_model.named_parameters(),
                        client_model.named_parameters()):

                    if any(bn_keyword in name_g for bn_keyword in ['bn', 'batchnorm', 'batch_norm']):
                        continue

                    deviation = c_param.data - g_param.data
                    dev_vectors.append(deviation.flatten())

                if not dev_vectors:
                    print(f"警告：客户端{i}没有可聚合的非BN参数")
                    dummy_vec = torch.randn(100, device=self.device) * eps
                    deviations.append(dummy_vec)
                else:
                    deviations.append(torch.cat(dev_vectors))

            if not deviations:
                print("错误：所有客户端偏差为空")
                return None

            max_len = max(len(dev) for dev in deviations)
            padded_deviations = []
            for dev in deviations:
                if len(dev) < max_len:
                    padding = torch.zeros(max_len - len(dev), device=self.device)
                    padded_dev = torch.cat([dev, padding])
                else:
                    padded_dev = dev[:max_len]
                padded_deviations.append(padded_dev)

            G = torch.stack(padded_deviations, dim=1)
            print(f"偏差矩阵G形状: {G.shape}, 范数: {torch.norm(G):.6f}")
            return G

        except Exception as e:
            print(f"偏差计算错误: {e}")
            return None

    def _solve_nash_bargaining_safe(self, G, max_iter, lr, eps):
        try:
            d, K = G.shape
            if K == 0 or d == 0:
                print("偏差矩阵维度异常")
                return None

            g_norm = torch.norm(G)
            if g_norm < eps:
                print(f"偏差矩阵G范数过小({g_norm:.2e})，使用均匀权重")
                return torch.ones(K, device=self.device) / K

            p = torch.ones(K, device=self.device) / K
            p = torch.clamp(p, min=eps)
            p = p / torch.sum(p)

            G_T_G = torch.matmul(G.T, G)
            G_T_G = G_T_G + eps * torch.eye(K, device=self.device)

            for iter in range(max_iter):
                q = torch.matmul(G_T_G, p)
                q = torch.clamp(q, min=eps)

                log_p = torch.log(p + eps)
                log_q = torch.log(q + eps)
                phi = log_p + log_q

                max_phi = torch.max(torch.abs(phi))
                if max_phi < 1e-4:
                    print(f"DDWA收敛于第{iter}次迭代, max_phi={max_phi:.2e}")
                    break

                safe_q = torch.clamp(q, min=eps)
                gradient = torch.matmul(G_T_G, 1.0 / safe_q)

                current_lr = lr / (1 + 0.05 * iter)
                p_new = p - current_lr * gradient

                p_new = torch.clamp(p_new, min=eps)
                p_new = p_new / torch.sum(p_new)

                if torch.any(torch.isnan(p_new)) or torch.any(torch.isinf(p_new)):
                    print(f"第{iter}次迭代权重无效，使用上一步结果")
                    break

                p = p_new

            if torch.any(torch.isnan(p)) or torch.any(torch.isinf(p)):
                print("最终权重无效，使用均匀权重")
                return torch.ones(K, device=self.device) / K

            print(f"DDWA权重求解成功: min={torch.min(p):.4f}, max={torch.max(p):.4f}, std={torch.std(p):.4f}")
            return p

        except Exception as e:
            print(f"Nash权重求解错误: {e}")
            return None

    def _apply_aggregation_with_weights(self, p):
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)

        total_weight = 0.0
        for i, client_model in enumerate(self.uploaded_models):
            weight = p[i].item()
            total_weight += weight

            for (name, client_param), global_param in zip(
                    client_model.named_parameters(),
                    self.global_model.parameters()):

                if any(bn_keyword in name for bn_keyword in ['bn', 'batchnorm', 'batch_norm']):
                    continue

                global_param.data += client_param.data.clone() * weight

        if abs(total_weight - 1.0) > 0.01:
            print(f"权重和异常: {total_weight}, 进行重新归一化")
            for param in self.global_model.parameters():
                param.data = param.data / total_weight

        print(f"DDWA聚合完成，权重范围: [{torch.min(p):.3f}, {torch.max(p):.3f}]")

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            print(f'Client {c.id}: Acc: {ct * 1.0 / ns}, AUC: {auc}')
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            print(f'Client {c.id}: Train loss: {cl * 1.0 / ns}')
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    def evaluate(self):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        std_accs = np.std([a / n for a, n in zip(stats[2], stats[1])])
        std_aucs = np.std([a / n for a, n in zip(stats[3], stats[1])])

        self.rs_test_acc.append(test_acc)
        self.rs_std_acc.append(std_accs)

        self.rs_test_auc.append(train_loss)
        self.rs_std_auc.append(std_aucs)

        self.rs_train_loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(std_accs))
        print("Std Test AUC: {:.4f}".format(std_aucs))