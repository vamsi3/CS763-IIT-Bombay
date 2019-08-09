import torch

# ====================================================================================================================
# 												LOSS FUNCTIONS
# ====================================================================================================================

class CrossEntropyLoss:
	def __init__(self):
		pass

	def forward(self, input, target):
		input_max, _ = torch.max(input, dim=1, keepdim=True)
		input_softmax = (input - input_max).exp() # For numerical stability while computing softmax
		input_softmax /= input_softmax.sum(1, keepdim=True)
		loss = -input_softmax[torch.arange(target.shape[0], dtype=torch.long), target].log().mean()
		self.cache = input_softmax
		return loss

	def backward(self, input, target):
		input_softmax = self.cache
		delattr(self, 'cache')
		grad_input = input_softmax / input.shape[0]
		grad_input[torch.arange(target.shape[0], dtype=torch.long), target] -= 1
		return grad_input

	__call__ = forward
