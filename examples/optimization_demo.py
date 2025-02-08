"""
Example: optimization_demo.py
Demonstrates the effect of graph optimizations, including common subexpression elimination.
"""

from src.core.tracer import constant, add, mul
from src.core.compiler import Compiler
from src.utils.visualization import visualize

def demo():
    # Create a computation graph with a redundant subexpression.
    a = constant(2)
    b = constant(3)
    sum1 = add(a, b)
    sum2 = add(a, b)  # Duplicate subexpression
    expr = mul(sum1, sum2)
    print("Before optimization:", expr)
    
    # Visualize the original graph.
    visualize(expr, filename="before_optimization")
    
    # Optimize the graph.
    compiler = Compiler()
    optimized_expr = compiler.compile(expr)
    print("After optimization:", optimized_expr)
    
    # Visualize the optimized graph.
    visualize(optimized_expr, filename="after_optimization")
    
if __name__ == "__main__":
    demo()