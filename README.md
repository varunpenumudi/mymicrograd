# **mymicrograd**

<p align="center">
   <img src="assets/image.png" width=400 height=250> <img>
<p>

**mymicrograd** is an implementation of Andrej Karpathy’s **micrograd**, a backpropagation and neural networks engine. This project provides a foundation for understanding the core concepts behind neural networks, gradient computation, and backpropagation in a simple, clean codebase.

## **Features**

- **Back Propagation**: Compute gradients for backpropagation through a computational graph.

- **Custom Operations**: Supports basic arithmetic operations along with activation functions (ReLU, Tanh, Exp), and more.

- **Neural Network Building**: Create simple neural networks with layers and neurons, and perform forward and backward passes.

- **Scalar Valued**: This is a Scalar valued engine. Focused on clarity and simplicity, perfect for learning how neural networks and gradient descent work under the hood.

## **Cloning and importing**

1. Clone the repository:

   ```bash
   git clone https://github.com/varunpenumudi/mymicrograd.git
   ```

2. move to the folder and use it

   ```bash
   cd mymicrograd
   ```

## **Usage**

Here's an example of using **mymicrograd** to create values, perform operations, and backpropagate:

```python
from mymicrograd.scalar import Value

# Create two values
a = Value(3.0)
b = Value(2.0)

# Perform operations
c = a * b
d = c + a
e = d.relu()

# Perform backward pass to calculate gradients
e.backward()

# Output results
print(f"Final output: {e.data}")
print(f"Gradient of a: {a.grad}")
print(f"Gradient of b: {b.grad}")
```

This simple code performs a series of operations and computes gradients for the involved variables. The backward pass updates the gradients of the values.

## **Core Classes**

- **Value**: Represents a scalar value, tracks gradients, and supports automatic differentiation for operations like addition, multiplication, division, etc.
- **Neuron**: Represents a simple neuron in a neural network, with weights, biases, and an activation function (currently, `tanh`).
- **Layer**: A fully connected layer that contains multiple neurons.
- **MLP (Multi-Layer Perceptron)**: A simple multi-layer neural network built from multiple layers.

## **Operations Supported**

- **Arithmetic Operations**: `+`, `-`, `*`, `/`
- **Activation Methods**: `relu()`, `tanh()`, `exp()`
- **Backward Pass**: Compute gradients through the computational graph using the `.backward()` method.

## **Why mymicrograd?**

This library is designed for learning and teaching the core concepts behind neural networks and automatic differentiation. If you’ve ever wondered how popular frameworks like TensorFlow or PyTorch work, **mymicrograd** gives you a simple implementation that will help you understand the underlying principles of gradient computation and backpropagation.

## **Acknowledgments**

This project is heavily inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).
