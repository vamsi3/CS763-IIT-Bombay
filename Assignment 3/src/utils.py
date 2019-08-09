import math
import torch

# ====================================================================================================================
# 												DATALOADER
# ====================================================================================================================


class DataLoader:
	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = list(dataset) if type(dataset) == tuple else [dataset]
		self.batch_size = batch_size
		self.shuffle = shuffle

		self.dataset_size = self.dataset[0].shape[0]
		self.batches_outstanding = math.ceil(self.dataset_size / self.batch_size)
		
		if self.shuffle:
			indices = torch.randperm(self.dataset_size)
			self.dataset = [data[indices] for data in self.dataset]

	def __iter__(self):
		return self

	def __next__(self):
		if self.batches_outstanding == 0:
			self.batches_outstanding = math.ceil(self.dataset_size / self.batch_size) # This helps for next epoch to reuse the same dataloader object
			raise StopIteration
		
		self.batches_outstanding -= 1
		batch = [data[self.batches_outstanding * self.batch_size: (self.batches_outstanding + 1) * self.batch_size] for data in self.dataset]
		return tuple(batch) if len(batch) > 1 else batch[0]
