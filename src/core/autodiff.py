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
    (Fixed: resets all node gradients and adds a branch for division.)
    
    Args:
        node (Node): The output node to compute gradients from.
        seed_grad (float): Initial gradient value for the output node.
    """
    # Get nodes in topological order.
    sorted_nodes = topological_sort(node)
    
    # Reset gradients for all nodes (this prevents accumulation from prior calls)
    for n in sorted_nodes:
        if isinstance(n.value, torch.Tensor):
            n.grad = torch.zeros_like(n.value)
        else:
            n.grad = 0.0

    # Set the output node's gradient to the seed value.
    if isinstance(node.value, torch.Tensor):
        node.grad = torch.ones_like(node.value) * seed_grad
    else:
        node.grad = seed_grad
    
    # Reverse-mode autodiff: process nodes in reverse topological order.
    for n in reversed(sorted_nodes):
        if n.op == "add":
            # For addition, d(a+b)/da = d(a+b)/db = 1.
            for inp in n.inputs:
                if isinstance(inp.grad, torch.Tensor) or isinstance(n.grad, torch.Tensor):
                    inp.grad = inp.grad + n.grad if isinstance(inp.grad, torch.Tensor) else torch.tensor(inp.grad) + n.grad
                else:
                    inp.grad += n.grad
        elif n.op == "mul":
            # For multiplication, if z = a * b then:
            # dz/da = b and dz/db = a.
            a, b = n.inputs
            a_val = a.value if a.value is not None else evaluate(a)
            b_val = b.value if b.value is not None else evaluate(b)
            if isinstance(n.grad, torch.Tensor) or isinstance(a_val, torch.Tensor) or isinstance(b_val, torch.Tensor):
                if not isinstance(a_val, torch.Tensor):
                    a_val = torch.tensor(a_val)
                if not isinstance(b_val, torch.Tensor):
                    b_val = torch.tensor(b_val)
                node_grad = n.grad if isinstance(n.grad, torch.Tensor) else torch.tensor(n.grad)
                a.grad = a.grad + (node_grad * b_val) if isinstance(a.grad, torch.Tensor) else torch.tensor(a.grad) + (node_grad * b_val)
                b.grad = b.grad + (node_grad * a_val) if isinstance(b.grad, torch.Tensor) else torch.tensor(b.grad) + (node_grad * a_val)
            else:
                a.grad += float(n.grad) * b_val
                b.grad += float(n.grad) * a_val
        elif n.op == "div":
            # For division, if z = a / b then:
            # dz/da = 1 / b  and  dz/db = -a / (b^2)
            a, b = n.inputs
            a_val = a.value if a.value is not None else evaluate(a)
            b_val = b.value if b.value is not None else evaluate(b)
            if isinstance(n.grad, torch.Tensor) or isinstance(a_val, torch.Tensor) or isinstance(b_val, torch.Tensor):
                if not isinstance(a_val, torch.Tensor):
                    a_val = torch.tensor(a_val)
                if not isinstance(b_val, torch.Tensor):
                    b_val = torch.tensor(b_val)
                node_grad = n.grad if isinstance(n.grad, torch.Tensor) else torch.tensor(n.grad)
                a.grad = a.grad + (node_grad / b_val) if isinstance(a.grad, torch.Tensor) else torch.tensor(a.grad) + (node_grad / b_val)
                b.grad = b.grad + (node_grad * (-a_val / (b_val * b_val))) if isinstance(b.grad, torch.Tensor) else torch.tensor(b.grad) + (node_grad * (-a_val / (b_val * b_val)))
            else:
                a.grad += n.grad / b_val
                b.grad += n.grad * (-a_val / (b_val * b_val))
        elif n.op == "const":
            # Constants do not contribute gradients.
            pass
        else:
            grad_fn = n.metadata.get("grad_fn", None)
            if grad_fn:
                grads = grad_fn(n)
                for inp, grad in zip(n.inputs, grads):
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
