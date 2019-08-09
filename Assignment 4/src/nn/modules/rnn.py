import math
import torch
from .module import Module


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, nonlinearity='tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity

        self.register_parameter('w_ih', torch.empty(hidden_size, input_size))
        self.register_parameter('b_ih', torch.empty(hidden_size))
        self.register_parameter('w_hh', torch.empty(hidden_size, hidden_size))
        self.register_parameter('b_hh', torch.empty(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        std = 1 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)

    @staticmethod
    def tanh(input):
        input_exp = (-input.abs()).exp().pow(2)
        output = (1 - input_exp) / (1 + input_exp)
        output[input < 0] *= -1
        return output

    def forward(self, input, h_x=None):
        if self.batch_first:
            input = input.permute(1, 0, 2)
        
        seq_len, batch_size, input_size = input.shape
        if h_x is None:
            h_x = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        output = torch.empty(seq_len, batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        for t in range(seq_len):
            x_t = input[t]
            h_x = self.tanh(self.w_ih.data @ x_t.t() + self.b_ih.data.reshape(-1, 1) + self.w_hh.data @ h_x.t() + self.b_hh.data.reshape(-1, 1)).t()
            output[t] = h_x.clone()
        self.save_cache(input_size, input, output)

        if self.batch_first:
            output = output.permute(1, 0, 2)

        return output, h_x

    def backward(self, grad_output):
        if self.batch_first:
            grad_output = grad_output.permute(1, 0, 2)

        seq_len, batch_size = grad_output.size(0), grad_output.size(1)
        input_size, input, output = self.load_cache()
        grad_h_x = torch.zeros(batch_size, self.hidden_size, dtype=grad_output.dtype, device=grad_output.device)
        grad_input = torch.empty(seq_len, batch_size, input_size, dtype=grad_output.dtype, device=grad_output.device)
        for t in range(seq_len-1, -1, -1):
            x_t, h_x = input[t], output[t]
            grad_h_x += grad_output[t]
            grad_z = ((1 - h_x.pow(2)) * grad_h_x).t()
            self.w_ih.grad += grad_z @ x_t
            self.w_hh.grad += grad_z @ h_x
            self.b_ih.grad += grad_z.sum(1)
            self.b_hh.grad += grad_z.sum(1)
            grad_input[t] += grad_z.t() @ self.w_ih.data
            grad_h_x = grad_z.t() @ self.w_hh.data

        if self.batch_first:
            grad_input = grad_input.permute(1, 0, 2)

        return grad_input
