from tinygrad.scalar import Value
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # x*w + b
        summ = sum([wi*xi for wi, xi in zip(self.w, x)], self.b)
        out = summ.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs   
    
    def parameters(self):
        # return [ n.parameters() for n in self.neurons ]
        return [ p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin]+nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        # return [ l.parameters() for l in self.layers ]
        return [ p for l in self.layers for p in l.parameters()]