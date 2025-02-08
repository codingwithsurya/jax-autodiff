"""
Module: tracer.py
Defines the computation graph, tracing functionality, and memory management.
"""

import uuid
from typing import List, Any, Callable, Dict
import torch
from ..metal.metal_ops import metal_add, metal_mul, metal_div, to_tensor, get_device

class Node:
    """A node in the computation graph."""
    
    def __init__(self, op: str = None, inputs: List[Any] = None, value: Any = None):
        """
        Create a new node in the computation graph.
        
        Args:
            op (str): The operation name.
            inputs (List[Any], optional): List of input nodes or values.
            value (Any, optional): The constant value (if any).
        """
        self.id = str(uuid.uuid4())
        self.op = op
        self.inputs = []
        if inputs:
            for inp in inputs:
                if isinstance(inp, Node):
                    self.inputs.append(inp)
                    inp.ref_count += 1
                else:
                    # Convert non-Node inputs to constant Nodes
                    const_node = constant(inp)
                    self.inputs.append(const_node)
                    const_node.ref_count += 1
        self.value = value
        self.grad = None  # For automatic differentiation
        
        # --- Memory Management Enhancements ---
        self.ref_count = 0  # Track references for memory management
        self.buffer = None  # For memory allocation planning
        
        self.metadata: Dict[str, Any] = {}

    def release(self):
        """Release memory and decrease ref counts of inputs."""
        # Release any allocated buffers
        self.buffer = None
        
        # Decrease reference counts of inputs
        for inp in self.inputs:
            inp.ref_count -= 1
            if inp.ref_count == 0:
                inp.release()
    
    def __add__(self, other: Any) -> 'Node':
        return add(self, other)
    
    def __mul__(self, other: Any) -> 'Node':
        return mul(self, other)
        
    def __truediv__(self, other: Any) -> 'Node':
        return div(self, other)
    
    def __repr__(self):
        if self.value is not None:
            return f"Const({self.value})"
        return f"Node(op={self.op}, id={self.id[:6]})"

def constant(value: Any) -> Node:
    """Create a constant node."""
    return Node(op="const", value=value)

def add(a: Any, b: Any) -> Node:
    """Add two values."""
    if not isinstance(a, Node):
        a = constant(a)
    if not isinstance(b, Node):
        b = constant(b)
    return Node(op="add", inputs=[a, b])

def mul(a: Any, b: Any) -> Node:
    """Multiply two values."""
    if not isinstance(a, Node):
        a = constant(a)
    if not isinstance(b, Node):
        b = constant(b)
    return Node(op="mul", inputs=[a, b])

def div(a: Any, b: Any) -> Node:
    """Divide two values."""
    if not isinstance(a, Node):
        a = constant(a)
    if not isinstance(b, Node):
        b = constant(b)
    return Node(op="div", inputs=[a, b])

def evaluate(node: Node) -> Any:
    """
    Evaluate a node in the computation graph using Metal acceleration when possible.
    
    Args:
        node (Node): The node to evaluate.
    
    Returns:
        Any: The computed value (scalar or tensor).
    """
    if node.value is not None:
        return node.value
        
    # Evaluate inputs
    inputs = [evaluate(inp) for inp in node.inputs]
    
    # Convert inputs to tensors if any input is a tensor
    has_tensor = any(isinstance(x, torch.Tensor) for x in inputs)
    if has_tensor:
        inputs = [torch.tensor(x, device='cpu') if not isinstance(x, torch.Tensor) else x.cpu() for x in inputs]
        
        # Use Metal operations for tensors
        if node.op == "add":
            return inputs[0] + inputs[1]
        elif node.op == "mul":
            return inputs[0] * inputs[1]
        elif node.op == "div":
            return inputs[0] / inputs[1]
        else:
            raise ValueError(f"Unknown operation: {node.op}")
    else:
        # Use regular Python operations for scalars
        if node.op == "add":
            return float(inputs[0] + inputs[1])
        elif node.op == "mul":
            return float(inputs[0] * inputs[1])
        elif node.op == "div":
            return float(inputs[0] / inputs[1])
        else:
            raise ValueError(f"Unknown operation: {node.op}")

def trace(node_or_value):
    """
    Evaluate a node or return a value.
    
    Args:
        node_or_value: A Node to evaluate or a value to return as is.
        
    Returns:
        The evaluated result.
    """
    if isinstance(node_or_value, Node):
        return evaluate(node_or_value)
    return node_or_value
