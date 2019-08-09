import torch
from .module import Module


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter('weight', torch.empty(out_features, in_features))
        self.register_parameter('bias', torch.empty(out_features) if bias else None)
        self.reset_parameters()

    def reset_parameters(self):
        std = torch.tensor(2 / sum(self.weight.shape)).sqrt()
        self.weight = torch.randn_like(self.weight) * std
        self.bias = torch.randn_like(self.bias) * std

    def forward(self, input):
        output = input.reshape(-1, self.in_features) @ self.weight.t() + self.bias
        self.save_cache(input)
        return output.reshape(*input.shape[:-1], self.out_features)

    def backward(self, grad_output):
        input = self.load_cache()
        grad_output = grad_output.reshape(-1, self.out_features)
        grad_input = grad_output @ self.weight
        self.weight._grad += grad_output.t() @ input.reshape(-1, self.in_features)
        self.bias._grad += grad_output.sum(0)
        return grad_input
