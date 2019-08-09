from collections import OrderedDict
import torch

class Module:
    def __init__(self):
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        self.training = True

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *grad_output):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        return self.forward(*input, **kwargs)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for key, param in self._parameters.items():
            if param is not None:
                self._parameters[key] = fn(param)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

    def cpu(self):
        self._apply(lambda t: t.cpu())
        return self

    def to(self, device=None):
        self._apply(lambda t: t.to(device))
        return self

    def register_parameter(self, name, param):
        self._parameters[name] = param

    def register_buffer(self, name, tensor):
        self._buffers[name] = tensor

    def add_module(self, name, module):
        self._modules[name] = module

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def buffers(self, recurse=True):
        for name, param in self.named_buffers(recurse=recurse):
            yield param

    def children(self):
        for name, module in self.named_children():
            yield module

    def modules(self):
        for name, module in self.named_modules():
            yield module

    def named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def named_buffers(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        memo = set()
        modules = self.named_modules(
            prefix=prefix) if recurse else ((prefix, self),)
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for key, value in members:
                if value is None or value in memo:
                    continue
                memo.add(value)
                name = f"{module_prefix}{'.' if module_prefix else ''}{key}"
                yield name, value

    def named_children(self):
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = f"{prefix}{'.' if prefix else ''}{name}"
                    for m in module.named_modules(memo, submodule_prefix):
                        yield m

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]

        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]

        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        self.train(False)

    def zero_grad(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
            else:
                param.grad = torch.zeros_like(param)

    def save_cache(self, *cache):
        if self.training:
            self.cache = cache
            if len(cache) == 1:
                self.cache = self.cache[0]

    def load_cache(self):
        if self.training:
            cache = self.cache
            del self.cache
            return cache

    def state_dict(self, prefix=''):
        destination = OrderedDict()
        for name, param in self._parameters.items():
            if param is not None:
                destination[f'{prefix}{name}'] = param.data
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[f'{prefix}{name}'] = buf.data
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(f'{prefix}{name}.')
        return destination

    def load_state_dict(self, state_dict, prefix=''):
        for name in self._parameters:
            self._parameters[name].data.copy_(state_dict[f'{prefix}{name}'])
        for name in self._buffers:
            self._parameters[name].data.copy_(state_dict[f'{prefix}{name}'])
        for name, module in self._modules.items():
            if module is not None:
                module.load_state_dict(state_dict, f'{prefix}{name}.')
