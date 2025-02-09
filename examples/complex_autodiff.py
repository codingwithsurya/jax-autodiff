"""
Example demonstrating autodifferentiation of complex expressions:
1. f(x) = x * x + 1
2. g(x) = x * x - 1/(x * x * x)
3. h(x) = f(x) * g(x)
"""

from src.core.tracer import constant, add, mul, div, trace, Node
from src.core.autodiff import compute_gradients
import graphviz

def create_visualization(node: Node, filename: str):
    """Create a graphviz visualization of the computation graph."""
    dot = graphviz.Digraph(comment='Computation Graph')
    dot.attr(rankdir='TB')
    
    visited = set()
    
    def add_nodes(n: Node):
        if n.id in visited:
            return
        visited.add(n.id)
        
        # Add node
        label = f"{n.op if isinstance(n.op, str) else n.op.__name__}"
        if hasattr(n, 'value') and n.value is not None:
            label += f"\nvalue={n.value}"
        if hasattr(n, 'grad'):
            label += f"\ngrad={n.grad}"
        
        dot.node(n.id, label)
        
        # Add edges from inputs
        for inp in n.inputs:
            add_nodes(inp)
            dot.edge(inp.id, n.id)
    
    add_nodes(node)
    dot.render(filename, view=True, format='png')

def main():
    # Create input x
    x = constant(2.0)  # Evaluate at x = 2

    # First expression: f(x) = x * x + 1
    x_squared = mul(x, x)
    f = add(x_squared, constant(1.0))

    # Second expression: g(x) = x * x - 1/(x * x * x)
    x_squared_2 = mul(x, x)
    x_cubed = mul(x, mul(x, x))
    one_over_x_cubed = div(constant(1.0), x_cubed)
    g = add(x_squared_2, mul(constant(-1.0), one_over_x_cubed))

    # Final expression: h(x) = f(x) * g(x)
    h = mul(f, g)

    # Compute gradients for f(x) separately
    x.grad = 0.0
    compute_gradients(f)
    print(f"Evaluating at x = 2.0:")
    print("f(x) = x * x + 1:")
    print(f"Value: {trace(f)}")
    print(f"Gradient: {x.grad}")  # Should be 4.0
    print()

    # Compute gradients for g(x)
    x.grad = 0.0
    compute_gradients(g)
    print("g(x) = x * x - 1/(x * x * x):")
    print(f"Value: {trace(g)}")
    print(f"Gradient: {x.grad}")  # Should be 4.1875
    print()

    # Compute gradients for h(x)
    x.grad = 0.0
    compute_gradients(h)
    print("h(x) = f(x) * g(x):")
    print(f"Value: {trace(h)}")
    print(f"Gradient: {x.grad}")  # Should be 36.4375

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
