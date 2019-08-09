import math
import numpy as np
import torch


# ====================================================================================================================
# 												BASE CLASS FOR LAYERS
# ====================================================================================================================


class Layer:
	def __init__(self):
		self.params, self.grad, self.buffer = {}, {}, {}
		self.training = True

	def to(self, device):
		self.device = device
		for name in self.params:
			self.params[name] = self.params[name].to(device)
			self.grad[name] = self.grad[name].to(device)
		for name in self.buffer:
			self.buffer[name] = self.buffer[name].to(device)
		return self

	def cpu(self):
		for name in self.params:
			self.params[name] = self.params[name].cpu()
			self.grad[name] = self.grad[name].cpu()
		for name in self.buffer:
			self.buffer[name] = self.buffer[name].cpu()
		return self

	def init_grad(self):
		for name in self.params:
			self.grad[name] = torch.zeros_like(self.params[name])
		return self

	def zero_grad(self):
		for name in self.params:
			self.grad[name].zero_()
		return self



# ====================================================================================================================
# 												LINEAR LAYERS
# ====================================================================================================================



class Linear(Layer):
	def __init__(self, input_size, output_size):
		super().__init__()

		# Glorot Initialization
		std = math.sqrt(2.0 / (input_size + output_size))
		self.params['weights'] = torch.randn(input_size, output_size) * std
		self.params['bias'] = torch.randn(1, output_size) * std
		
		self.init_grad()


	def forward(self, input):
		output = input.clone() @ self.params['weights'] + self.params['bias']
		return output

	def backward(self, input, grad_output):
		grad_input = grad_output.clone() @ self.params['weights'].t()
		self.grad['weights'] += input.t() @ grad_output
		self.grad['bias'] += grad_output.sum(0, keepdim=True)
		return grad_input



# ====================================================================================================================
# 												DROPOUT LAYERS
# ====================================================================================================================



class Dropout(Layer):
	def __init__(self, p=0.5):
		super().__init__()
		self.p = p

	def forward(self, input):
		if not self.training:
			return input
		bernoulli_sampler = torch.distributions.bernoulli.Bernoulli(probs= 1 - self.p)
		mask = bernoulli_sampler.sample(input.shape)
		output = input.clone()
		output[mask == 0] = 0
		output /= (1 - self.p)
		if self.training:
			self.cache = mask
		return output

	def backward(self, input, grad_output):
		mask = self.cache
		delattr(self, 'cache')

		if not self.training:
			return grad_output
		grad_input = grad_output.clone()
		grad_input[self.mask == 0] = 0
		grad_input *= (1 - self.p)
		return grad_input



# ====================================================================================================================
# 												UTILITY LAYERS
# ====================================================================================================================



class Flatten(Layer):
	def forward(self, input):
		return input.clone().reshape(input.shape[0], -1)

	def backward(self, input, grad_output):
		grad_input = grad_output.clone().reshape(input.shape)
		return grad_input



# ====================================================================================================================
# 												CONVOLUTION LAYERS
# ====================================================================================================================



class Conv2d(Layer):
	def __init__(self, input_size, in_channels, out_channels, kernel_size, stride=1, padding=0):
		super().__init__()

		self.in_height, self.in_width = input_size
		self.in_channels, self.out_channels = in_channels, out_channels
		self.kernel_height, self.kernel_width = kernel_size, kernel_size
		self.stride, self.padding = stride, padding

		# Kaiming Normal Initialization
		std = np.sqrt(1.0 / (self.in_channels * self.kernel_height * self.kernel_width)).item()
		self.params['weights'] = torch.rand(self.out_channels, self.in_channels, self.kernel_height, self.kernel_width) * std * 2 - std
		self.params['bias'] = torch.rand(self.out_channels) * std

		self.out_height = int((self.in_height - self.kernel_height) / self.stride) + 1
		self.out_width = int((self.in_width - self.kernel_width) / self.stride) + 1

		self.init_grad()



		i_kernel_indices = np.tile(np.repeat(np.arange(self.kernel_height), self.kernel_width), self.in_channels).reshape(-1, 1).astype(int)
		i_convolve_indices = self.stride * np.repeat(np.arange(self.out_height), self.out_width).reshape(1, -1).astype(int)
		self.i = i_kernel_indices + i_convolve_indices # Using broadcasting to our advantage

		j_kernel_indices = np.tile(np.arange(self.kernel_width), self.kernel_height * self.in_channels).reshape(-1, 1).astype(int)
		j_convolve_indices = self.stride * np.tile(np.arange(self.out_width), self.out_height).reshape(1, -1).astype(int)
		self.j = j_kernel_indices + j_convolve_indices

		self.k = np.repeat(np.arange(self.in_channels), self.kernel_height * self.kernel_width).reshape(-1, 1).astype(int)

		self.i = torch.from_numpy(self.i).long()
		self.j = torch.from_numpy(self.j).long()
		self.k = torch.from_numpy(self.k).long()


	def to(self, device):
		super().to(device)
		self.i.to(device)
		self.j.to(device)
		self.k.to(device)
		return self


	def forward(self, input):
		n = input.shape[0]

		X = input.clone()
		weights = self.params['weights']
		bias = self.params['bias']
		
		X_flattened = X[:, self.k, self.i, self.j].permute(1, 2, 0).reshape(self.in_channels * self.kernel_height * self.kernel_width, -1)
		W_flattened = weights.reshape(self.out_channels, -1)
		output = W_flattened @ X_flattened + bias.reshape(-1, 1)

		output = output.reshape(self.out_channels, self.out_height, self.out_width, n).permute(3, 0, 1, 2)
		if self.training:
			self.cache = X_flattened

		return output

	def backward(self, input, grad_output):
		n = input.shape[0]

		X_flattened = self.cache
		delattr(self, 'cache')

		delta = grad_output.clone()
		weights = self.params['weights']
		bias = self.params['bias']

		delta_flattened = delta.permute(1, 2, 3, 0).reshape(self.out_channels, -1)
		W_flattened = weights.reshape(self.out_channels, -1)
		new_delta_flattened = W_flattened.t() @ delta_flattened
		new_delta_flattened = new_delta_flattened.reshape(self.in_channels * self.kernel_height * self.kernel_width, -1, n).permute(2, 0, 1)

		grad_input = torch.zeros_like(input)
		for out_h_i in range(self.out_height):
			for out_w_i in range(self.out_width):
				grad_input[:, :, out_h_i * self.stride: out_h_i * self.stride + self.kernel_height, out_w_i * self.stride: out_w_i * self.stride + self.kernel_width] \
					+= (grad_output[:, :, out_h_i, out_w_i].t().unsqueeze(2) * W_flattened.unsqueeze(1)).sum(0).reshape(n, self.in_channels, self.kernel_height, self.kernel_width)


		self.grad['bias'] += grad_output.clone().sum(dim=(0, 2, 3))
		self.grad['weights'] += (delta_flattened @ X_flattened.t()).reshape(weights.shape)

		return grad_input



# ====================================================================================================================
# 												POOLING LAYERS
# ====================================================================================================================



# This implementation works only when kernel_size == stride
class MaxPool2d(Layer):
	def __init__(self, kernel_size, stride=None, padding=0):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride if stride is not None else self.kernel_size
		self.padding = padding

	def forward(self, input):
		output1 = input.clone().reshape(input.shape[0], input.shape[1], input.shape[2] // self.kernel_size, self.kernel_size, input.shape[3] // self.kernel_size, self.kernel_size)
		output2, idx2 = output1.max(5)
		output, idx = output2.max(3)
		if self.training:
			self.cache = (output1, output2, idx2, output, idx)
		return output

	def backward(self, input, grad_output):
		output1, output2, idx2, output, idx = self.cache
		delattr(self, 'cache')
		
		grad_output = grad_output.clone()

		grad_output2 = torch.zeros_like(output2).reshape(-1, output2.shape[3])
		grad_output2[torch.arange(grad_output2.shape[0]), idx.reshape(-1)] = grad_output.reshape(-1)
		grad_output2 = grad_output2.reshape(output2.shape).permute(0, 1, 2, 4, 3)

		grad_output1 = torch.zeros_like(output1).reshape(-1, output1.shape[-1])
		grad_output1[torch.arange(grad_output1.shape[0]), idx2.reshape(-1)] = grad_output2.reshape(-1)
		
		grad_input = grad_output1.reshape(input.shape)
		return grad_input


# This implementation works only when kernel_size == stride
class AvgPool2d(Layer):
	def __init__(self, kernel_size, stride=None, padding=0):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride if stride is not None else self.kernel_size
		self.padding = padding

	def forward(self, input):
		output = input.clone().reshape(input.shape[0], input.shape[1], input.shape[2] // self.kernel_size, self.kernel_size, input.shape[3] // self.kernel_size, self.kernel_size).mean((3, 5))
		return output

	def backward(self, input, grad_output):
		grad_input = grad_output.clone().unsqueeze(3).unsqueeze(5).expand(-1, -1, -1, self.kernel_size, -1, self.kernel_size).reshape(input.shape)
		grad_input /= self.kernel_size * self.kernel_size
		return grad_input



# ====================================================================================================================
# 												ACTIVATION FUNCTIONS
# ====================================================================================================================



class ReLU(Layer):
	def forward(self, input):
		output = input.clone().clamp(min=0)
		return output

	def backward(self, input, grad_output):
		grad_input = grad_output.clone()
		grad_input[input < 0] = 0
		return grad_input


# Parametric ReLU (He et al., 2015)
class PReLU(Layer):
	def __init__(self, num_parameters=1, init=0.25):
		super().__init__()
		self.num_parameters = num_parameters
		self.init = init
		self.params['a'] = torch.full((1, num_parameters), self.init)
		self.init_grad()

	def forward(self, input):
		output = input.clamp(min=0) + self.params['a'] * input.clamp(max=0)
		return output

	def backward(self, input, grad_output):
		self.grad['a'][input < 0] += grad_output[input < 0] 
		grad_input = grad_output.clone()
		grad_input[input < 0] *= self.params['a'][input < 0]
		return grad_input



# ====================================================================================================================
# 												NORMALIZATION LAYERS
# ====================================================================================================================



class BatchNorm1d(Layer):
	def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False):
		super().__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		self.affine = affine
		self.track_running_stats = track_running_stats
		
		if self.affine:
			self.params['gamma'] = torch.rand(self.num_features)
			self.params['beta'] = torch.zeros(self.num_features)
		
		if self.track_running_stats:
			self.buffer['running_mean'] = torch.zeros(num_features)
			self.buffer['running_var'] = torch.ones(num_features)
			self.buffer['num_batches_tracked'] = torch.tensor(0, dtype=torch.long)

		self.reset_parameters()

	def reset_parameters(self):
		self.init_grad()
		self.reset_running_stats()
		if self.affine:
			self.params['gamma'].uniform_()
			self.params['beta'].zero_()

	def reset_running_stats(self):
		if self.track_running_stats:
			self.buffer['running_mean'].zero_()
			self.buffer['running_var'].fill_(1)
			self.buffer['num_batches_tracked'].zero_()

	def forward(self, input):
		mean, var = input.mean(0), input.var(0, unbiased=False)

		if self.training and self.track_running_stats:
				self.buffer['num_batches_tracked'] += 1
				exponential_average_factor = 0.0
				if self.momentum is None:
					exponential_average_factor = 1.0 / self.buffer['num_batches_tracked']
				else:
					exponential_average_factor = self.momentum
				
				mean_curr, var_curr = input.mean(0), input.var(0, unbiased=True)

				self.buffer['running_mean'] = (1 - exponential_average_factor) * self.buffer['running_mean'] + exponential_average_factor * mean_curr
				self.buffer['running_var'] = (1 - exponential_average_factor) * self.buffer['running_var'] + exponential_average_factor * var_curr


		output = input.clone()
		output = (output - mean) * torch.rsqrt(var + self.eps)
		if self.training:
			self.cache = mean, var, output
		if self.affine:
			output = output * self.params['gamma'] + self.params['beta']
		return output

	def backward(self, input, grad_output):
		mean, var, output = self.cache
		delattr(self, 'cache')
		n = input.shape[0]

		if self.affine:
			grad_output = grad_output.clone()
			self.grad['beta'] += grad_output.clone().sum(0)
			self.grad['gamma'] += (grad_output.clone() * output).sum(0)
			grad_output *= self.params['gamma']

		grad_input = ((input - mean) * grad_output).sum(0, keepdim=True) * (input - mean) / (var + self.eps)
		grad_input = n * grad_output - grad_output.sum(0) - grad_input
		grad_input *= torch.rsqrt(var + self.eps)
		grad_input /= n

		return grad_input


class BatchNorm2d(BatchNorm1d):
	def forward(self, input):
		input_size = input.shape
		output = super().forward(input.clone().permute(0, 2, 3, 1).reshape(-1, input_size[1]))
		output = output.reshape(input_size[0], input_size[2], input_size[3], input_size[1]).permute(0, 3, 1, 2)
		return output
	
	def backward(self, input, grad_output):
		input_size = input.shape
		grad_input = super().backward(input.clone().permute(0, 2, 3, 1).reshape(-1, input_size[1]), grad_output.permute(0, 2, 3, 1).reshape(-1, input_size[1]))
		grad_input = grad_input.reshape(input_size[0], input_size[2], input_size[3], input_size[1]).permute(0, 3, 1, 2)
		return grad_input
