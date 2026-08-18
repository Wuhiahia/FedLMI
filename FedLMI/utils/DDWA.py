import torch
import numpy as np
from typing import List, Tuple


class DDWAModule:
    def __init__(self, num_clients: int, device: str = 'cpu'):
        self.K = num_clients
        self.device = device

    def solve_nash_bargaining(self, G: torch.Tensor, max_iter: int = 50, lr: float = 0.01, eps: float = 1e-8) -> torch.Tensor:
        print(f"DDWA求解: 矩阵形状{G.shape}, 范数{torch.norm(G):.6f}")
        d, K = G.shape
        if K == 0:
            return torch.ones(1, device=self.device)

        G_T_G = torch.matmul(G.T, G) + eps * torch.eye(K, device=self.device)

        p = torch.ones(K, device=self.device) / K
        p = torch.clamp(p, min=eps)

        for iter in range(max_iter):
            q = torch.matmul(G_T_G, p)
            q = torch.clamp(q, min=eps)

            log_p = torch.log(p + eps)
            log_q = torch.log(q + eps)
            phi = log_p + log_q

            if torch.max(torch.abs(phi)) < 1e-4:
                break

            gradient = torch.matmul(G_T_G, 1.0 / q)
            p_new = p - lr * gradient

            p_new = torch.clamp(p_new, min=eps)
            p_new = p_new / torch.sum(p_new)

            p = p_new
        print(f"DDWA权重范围: [{torch.min(p):.4f}, {torch.max(p):.4f}]")
        return p / torch.sum(p)

    def compute_deviations(self, global_model: torch.nn.Module, client_models: List[torch.nn.Module]) -> torch.Tensor:
        deviations = []
        for client_model in client_models:
            dev_vectors = []
            for (name_g, g_param), (name_l, c_param) in zip(
                    global_model.named_parameters(),
                    client_model.named_parameters()):

                if any(bn_keyword in name_g for bn_keyword in ['bn', 'batchnorm', 'batch_norm', 'dropout']):
                    continue

                dev_vectors.append((c_param.data - g_param.data).flatten())

            if dev_vectors:
                deviations.append(torch.cat(dev_vectors))
            else:
                deviations.append(torch.zeros(100, device=self.device))

        return torch.stack(deviations, dim=1)