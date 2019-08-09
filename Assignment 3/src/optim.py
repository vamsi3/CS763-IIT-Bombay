import math
import torch
from bisect import bisect_right


# ====================================================================================================================
# 												OPTIMIZERS
# ====================================================================================================================


class SGD:
	def __init__(self, model, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
		self.model = model
		self.lr = lr
		self.momentum = momentum
		self.dampening = dampening
		self.weight_decay = weight_decay
		self.nesterov = nesterov

		self.momentum_buffer = {}

	def step(self):
		for layer in self.model.layers:
			for name in layer.params:

				param = layer.params[name]
				grad = layer.grad[name]

				if self.weight_decay:
					grad.add_(self.weight_decay, layer.params[name]) # L2 regularization contributes this much additional gradient

				if self.momentum:
					if layer.params[name] not in self.momentum_buffer:
						self.momentum_buffer[param] = grad.clone()
					else:
						self.momentum_buffer[param].mul_(self.momentum).add_(1 - self.dampening, grad)
					if self.nesterov:
						grad.add_(self.momentum, self.momentum_buffer[param])
					else:
						grad = self.momentum_buffer[param]

				layer.params[name].add_(-self.lr, grad)



# Please note that torch.Tensor methods used in this Adam code are heavily derived and got known to me from PyTorch itself
class Adam:
	def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
		self.model = model
		self.lr = lr
		self.betas = betas
		self.eps = eps
		self.weight_decay = weight_decay

		self.state = {}

	def step(self):
		for idx, layer in enumerate(self.model.layers):
			for name in layer.params:

				param = layer.params[name]
				grad = layer.grad[name]


				if layer.params[name] not in self.state:
					self.state[param] = {}
					self.state[param]['step'] = 0
					self.state[param]['exp_avg'] = torch.zeros_like(param)
					self.state[param]['exp_avg_sq'] = torch.zeros_like(param)
				
				exp_avg, exp_avg_sq = self.state[param]['exp_avg'], self.state[param]['exp_avg_sq']
				beta1, beta2 = self.betas

				self.state[param]['step'] += 1

				if self.weight_decay:
					grad.add_(self.weight_decay, layer.params[name]) # L2 regularization contributes this much additional gradient

				exp_avg.mul_(beta1).add_(1 - beta1, grad)
				exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
				denominator = exp_avg_sq.sqrt().add_(self.eps)

				bias_correction1 = 1 - beta1 ** self.state[param]['step']
				bias_correction2 = 1 - beta2 ** self.state[param]['step']
				step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

				layer.params[name].addcdiv_(-step_size, exp_avg, denominator)



# ====================================================================================================================
# 												LEARNING RATE SCHEDULERS
# ====================================================================================================================



class _LRScheduler:
	def __init__(self, optimizer, last_epoch=-1):
		self.optimizer = optimizer
		self.base_lr = optimizer.lr
		self.last_epoch = last_epoch

	def step(self):
		self.last_epoch += 1
		self.optimizer.lr = self.get_lr()


class StepLR(_LRScheduler):
	def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
		super().__init__(optimizer, last_epoch)
		self.step_size = step_size
		self.gamma = gamma

	def get_lr(self):
		return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class MultiStepLR(_LRScheduler):
	def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
		super().__init__(optimizer, last_epoch)
		self.milestones = milestones
		self.gamma = gamma

	def get_lr(self):
		return self.base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)


class ExponentialLR(_LRScheduler):
	def __init__(self, optimizer, gamma, last_epoch=-1):
		super().__init__(optimizer, last_epoch)
		self.gamma = gamma

	def get_lr(self):
		return self.base_lr * self.gamma ** self.last_epoch


class CosineAnnealingLR(_LRScheduler):
	def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
		super().__init__(optimizer, last_epoch)
		self.T_max = T_max
		self.eta_min = eta_min

	def get_lr(self):
		return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
