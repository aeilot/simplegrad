from simplegrad.tensor import Tensor

class Parameter(Tensor):

    def __init__(self, data):
        super().__init__(data, requires_grad=True)

    def __repr__(self):
        return f"Parameter({self.data.shape})"

class Module():
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        params = []
        for name, value in self.__dict__.items():
            if isinstance(value, Parameter):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()