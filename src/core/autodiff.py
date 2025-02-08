"""
Module: autodiff.py
Implements automatic differentiation for the computation graph.
"""

from typing import List
from .tracer import Node
import torch

def topological_sort(node: Node) -> List[Node]:
    """
    Return nodes in topologically sorted order.
    
    Args:
        node (Node): The output node.
    
    Returns:
        List[Node]: Nodes in topological order.
    """
    visited = set()
    order = []

    def visit(n: Node):
        if n.id in visited:
            return
        visited.add(n.id)
        for inp in n.inputs:
            visit(inp)
        order.append(n)

    visit(node)
    return order

def compute_gradients(node: Node, seed_grad: float = 1.0):
    """
    Compute gradients through the computation graph using reverse-mode autodiff.
    
    Args:
        node (Node): The output node to compute gradients from.
        seed_grad (float): Initial gradient value for the output node.
    """
    # Initialize gradients with proper type
    if isinstance(node.value, torch.Tensor):
        node.grad = torch.ones_like(node.value) * seed_grad
    else:
        node.grad = seed_grad
    
    # Topological sort for reverse-mode autodiff
    sorted_nodes = topological_sort(node)
    
    # Initialize gradients for all nodes
    for n in sorted_nodes:
        if n.grad is None:
            if isinstance(n.value, torch.Tensor):
                n.grad = torch.zeros_like(n.value)
            else:
                n.grad = 0.0
    
    # Reverse-mode autodiff
    for node in reversed(sorted_nodes):
        if node.op == "add":
            # d(a + b)/da = d(a + b)/db = 1
            for inp in node.inputs:
                if isinstance(inp.grad, torch.Tensor) or isinstance(node.grad, torch.Tensor):
                    inp.grad = inp.grad + node.grad if isinstance(inp.grad, torch.Tensor) else torch.tensor(inp.grad) + node.grad
                else:
                    inp.grad += node.grad
        elif node.op == "mul":
            # d(a * b)/da = b, d(a * b)/db = a
            a, b = node.inputs
            a_val = a.value if a.value is not None else evaluate(a)
            b_val = b.value if b.value is not None else evaluate(b)
            
            # Convert to tensors if needed
            if isinstance(node.grad, torch.Tensor) or isinstance(a_val, torch.Tensor) or isinstance(b_val, torch.Tensor):
                if not isinstance(a_val, torch.Tensor):
                    a_val = torch.tensor(a_val)
                if not isinstance(b_val, torch.Tensor):
                    b_val = torch.tensor(b_val)
                if not isinstance(node.grad, torch.Tensor):
                    node_grad = torch.tensor(node.grad)
                else:
                    node_grad = node.grad
                
                a.grad = a.grad + (node_grad * b_val) if isinstance(a.grad, torch.Tensor) else torch.tensor(a.grad) + (node_grad * b_val)
                b.grad = b.grad + (node_grad * a_val) if isinstance(b.grad, torch.Tensor) else torch.tensor(b.grad) + (node_grad * a_val)
            else:
                a.grad += float(node.grad) * b_val
                b.grad += float(node.grad) * a_val
        elif node.op == "const":
            pass  # Constants have zero gradient
        else:
            grad_fn = node.metadata.get("grad_fn", None)
            if grad_fn:
                grads = grad_fn(node)
                for inp, grad in zip(node.inputs, grads):
                    inp.grad += grad

def evaluate(node: Node):
    if node.op == "const":
        return node.value
    elif node.op == "add":
        return evaluate(node.inputs[0]) + evaluate(node.inputs[1])
    elif node.op == "mul":
        return evaluate(node.inputs[0]) * evaluate(node.inputs[1])
    elif node.op == "div":
        return evaluate(node.inputs[0]) / evaluate(node.inputs[1])
    else:
        raise ValueError(f"Unknown operation: {node.op}")
