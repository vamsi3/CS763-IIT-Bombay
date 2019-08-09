import torch

class Model:
	def __init__(self, training=True):
		self.layers = []
		self.training = training

	def add_layer(self, layer):
		layer.training = self.training
		self.layers.append(layer)
		return self

	def forward(self, input):
		activations = [input]
		for layer in self.layers:
			activations.append(layer.forward(activations[-1]))
		if self.training:
			self.cache = activations # Cache between forward and backward passes
		return activations[-1]

	def backward(self, input, grad_output):
		activations = self.cache
		num_layers = len(self.layers)
		for layer_index in range(num_layers-1, -1, -1):
			grad_output = self.layers[layer_index].backward(activations[layer_index], grad_output)
		delattr(self, 'cache')
		return grad_output

	def zero_grad(self):
		for layer in self.layers:
			layer.zero_grad()
		return self

	def train(self):
		self.training = True
		for layer in self.layers:
			layer.training = self.training
		return self

	def eval(self):
		self.training = False
		for layer in self.layers:
			layer.training = self.training
		return self

	# For GPU support
	def to(self, device):
		self.device = device
		for layer in self.layers:
			layer.to(device)
		return self

	def cpu(self):
		for layer in self.layers:
			layer.cpu()
		return self

	__call__ = forward

	# Compatibility with specified API
	clear_grad_param = zero_grad

	def disp_grad_param(self):
		for layer in reversed(self.layers):
			for name in layer.grad:
				print(name.upper(), layer.grad[name], sep='\n')
			print("-"*150)
		return self