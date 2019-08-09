import math
import torch


class Adam:
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.parameters = list(parameters)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.state = {}

    def step(self):
        for param in self.parameters:
            grad = param.grad

            if param not in self.state:
                self.state[param] = {}
                self.state[param]['step'] = 0
                self.state[param]['exp_avg'] = torch.zeros_like(param)
                self.state[param]['exp_avg_sq'] = torch.zeros_like(param)
            
            exp_avg, exp_avg_sq = self.state[param]['exp_avg'], self.state[param]['exp_avg_sq']
            beta1, beta2 = self.betas

            self.state[param]['step'] += 1

            if self.weight_decay:
                grad.add_(self.weight_decay, param) # L2 regularization contributes this much additional gradient

            exp_avg.mul_(beta1).add_(1 - beta1, grad)
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
            denominator = exp_avg_sq.sqrt().add_(self.eps)

            bias_correction1 = 1 - beta1 ** self.state[param]['step']
            bias_correction2 = 1 - beta2 ** self.state[param]['step']
            step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

            param.addcdiv_(-step_size, exp_avg, denominator)
