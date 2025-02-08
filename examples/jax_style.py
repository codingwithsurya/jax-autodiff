"""
Example demonstrating JAX-style transformations (jit, vmap) and operation fusion.
"""

from src.core.tracer import constant, add, mul, trace
from src.transforms.jit import jit
from src.transforms.vmap import vmap
import torch
import time

# Basic function
def f(x):
    """Compute x^2 + 1"""
    return add(mul(x, x), constant(1.0))

# JIT-compiled version
@jit
def f_jit(x):
    return add(mul(x, x), constant(1.0))

# Vectorized version
@vmap
def f_vmap(x):
    return add(mul(x, x), constant(1.0))

# Combining transformations
@jit
@vmap
def f_both(x):
    return add(mul(x, x), constant(1.0))

def benchmark_transformations():
    print("Benchmarking JAX-style transformations\n")
    
    # Test data
    x_single = 2.0
    x_batch = torch.linspace(0, 10, 1000)
    
    # 1. Basic function
    print("1. Basic function f(x) = x^2 + 1")
    start = time.time()
    result = f(constant(x_single))
    print(f"Single input: {trace(result)}")
    print(f"Time: {(time.time() - start)*1000:.2f}ms\n")
    
    # 2. JIT-compiled function
    print("2. JIT-compiled function")
    # First call (compilation)
    start = time.time()
    result = f_jit(x_single)
    compile_time = time.time() - start
    # Second call (execution)
    start = time.time()
    result = f_jit(x_single)
    exec_time = time.time() - start
    print(f"Single input: {result}")
    print(f"Compilation time: {compile_time*1000:.2f}ms")
    print(f"Execution time: {exec_time*1000:.2f}ms\n")
    
    # 3. Vectorized function
    print("3. Vectorized function")
    start = time.time()
    result = f_vmap(x_batch)
    print(f"Batch input shape: {x_batch.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Time: {(time.time() - start)*1000:.2f}ms\n")
    
    # 4. Combined JIT and vectorization
    print("4. Combined JIT and vectorization")
    # First call (compilation)
    start = time.time()
    result = f_both(x_batch)
    compile_time = time.time() - start
    # Second call (execution)
    start = time.time()
    result = f_both(x_batch)
    exec_time = time.time() - start
    print(f"Batch input shape: {x_batch.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Compilation time: {compile_time*1000:.2f}ms")
    print(f"Execution time: {exec_time*1000:.2f}ms")

if __name__ == "__main__":
    benchmark_transformations()
