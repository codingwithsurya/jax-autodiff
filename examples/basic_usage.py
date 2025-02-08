"""
Example: basic_usage.py
Demonstrates a simple computation graph, automatic differentiation, and optimization.
"""

from src.core.tracer import constant, add, mul, trace
from src.core.autodiff import compute_gradients
from src.core.compiler import Compiler
from src.utils.visualization import visualize

def compute_expression():
    # Build a simple computation graph: (2 + 3) * 4
    a = constant(2)
    b = constant(3)
    c = constant(4)
    expr = mul(add(a, b), c)
    return expr

def main():
    expr = compute_expression()
    print("Original Expression:", expr)
    
    # Compute gradients (trivial in this example)
    compute_gradients(expr)
    print("Gradient of first input (from add):", expr.inputs[0].grad)
    
    # Optimize the graph
    compiler = Compiler()
    optimized_expr = compiler.compile(expr)
    print("Optimized Expression:", optimized_expr)
    
    # Visualize the optimized graph (opens a viewer if configured)
    visualize(optimized_expr, filename="basic_usage_graph")
    
if __name__ == "__main__":
    main()