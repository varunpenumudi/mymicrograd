import math

class Value:
    def __init__(self, data, _children=(), op='', label=''):
        self.label = label
        self.data = data
        self.grad = 0.0
        
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data+other.data, (self, other), op='+')

        def _backward(): # grads add up
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data*other.data, (self, other), op='*')

        def _backward(): # grads add up
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int, float values" 
        out = Value(self.data**other, (self, ), op='pow')

        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other): # other * self
        return self * other
    
    def __radd__(self, other): # other * self
        return self + other
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad = math.exp(x) * out.grad

        out._backward = _backward
        return out

    
    def backward(self):
        """ 
        implements back propagation using topological
        sorting for whole math expresssion graph
        """

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# test
if __name__  == "__main__":
    a = Value(2.0, label='a')
    b = Value(-3, label='b')
    print("a: ", a)
    print("b: ", b)

    print("----------------------")
    print("a+b: ", a+b)
    print("a*b: ", a*b)
    print("a-b: ", a-b)
    print("a/b: ", a/b)

    print("--------------------")
    print("a+1, 2+a: ", a+1, 2+a)
    print("a*2, 3*a: ", a*2, 3*a)
