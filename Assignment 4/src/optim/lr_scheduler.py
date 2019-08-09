import math
import torch
from bisect import bisect_right


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
