# JAX-Inspired Autodifferentiation Compiler

A JAX-inspired automatic differentiation compiler that implements key JAX features like function transformations (`jit`, `vmap`, `grad`) and operation fusion. Built with PyTorch backend and Metal acceleration for Apple Silicon.

## Features

### 1. Function Transformations
- **Just-In-Time Compilation (jit)**: Cache and reuse computation graphs for improved performance
- **Vectorized Mapping (vmap)**: Automatically vectorize functions across batch dimensions
- **Automatic Differentiation (grad)**: Compute gradients of functions with respect to inputs
- **Composable Transformations**: Stack transformations like `@jit`, `@vmap`, and `@grad`

### 2. Hardware Acceleration
- **Metal Performance Shaders**: Optimized for Apple Silicon
- **Automatic Device Placement**: Seamlessly handles CPU and GPU operations
- **Operation Fusion**: Automatically fuses compatible operations for better performance

### 3. Core Components
- **Computation Graph**: Track operations and dependencies for optimization
- **Automatic Differentiation**: Reverse-mode autodiff with efficient gradient computation
- **Operation Fusion**: Identify and combine operations for better performance

## Installation

```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic Usage
```python
from src.core.tracer import constant, add, mul
from src.transforms.jit import jit
from src.transforms.vmap import vmap
from src.transforms.grad import grad, value_and_grad

# Define a function
def f(x):
    return add(mul(x, x), constant(1.0))

# JIT compilation
f_fast = jit(f)
result = f_fast(2.0)  # Uses cached computation graph

# Vectorization
batch_f = vmap(f)
batch_result = batch_f([1.0, 2.0, 3.0])  # Applies f to each element

# Gradients
df = grad(f)
gradient = df(2.0)  # Computes df/dx at x=2.0

# Combined transformations
@jit
@vmap
@grad
def optimized_f(x):
    return add(mul(x, x), constant(1.0))
```

### Advanced Features

#### Value and Gradient
```python
def loss(params, data):
    # Your model here
    return prediction_error

value_grad_fn = value_and_grad(loss)
(loss_value, gradients), aux = value_grad_fn(params, data)
```

#### Automatic Operation Fusion
Operations are automatically fused when possible:
```python
@jit
def fused_ops(x, y):
    a = add(x, y)
    b = mul(a, a)
    return b  # add and mul operations may be fused
```

## Project Structure
```
.
├── examples/          # Example usage and benchmarks
├── src/
│   ├── core/         # Core autodiff and tracing
│   ├── metal/        # Metal acceleration
│   ├── optimizations/# Graph optimizations
│   └── transforms/   # Function transformations
└── tests/            # Unit tests
```

## Contributing
Feel free to open issues or submit pull requests. Areas of interest:
- Additional function transformations
- More optimization passes
- Extended hardware support
- Performance improvements

## License
MIT License

## Acknowledgments
Inspired by the JAX project and its functional programming approach to automatic differentiation.
