"""
Module: jit.py
Just-In-Time compilation transformation, similar to JAX's jit.
"""

from typing import Callable, Any, Dict, List
import torch
from .transform_base import Transform
from ..core.tracer import Node, constant, trace
from ..metal.metal_ops import get_device, to_tensor

class CachedGraph:
    """A cached computation graph that can be reused with different input values."""
    
    def __init__(self, output: Node, input_nodes: List[Node]):
        self.output = output
        self.input_nodes = input_nodes
        
    def __call__(self, *args):
        # Update input node values
        for node, arg in zip(self.input_nodes, args):
            if isinstance(arg, Node):
                node.value = arg.value
            else:
                node.value = arg
        return trace(self.output)

class jit(Transform):
    """
    Just-In-Time compilation transformation.
    
    Similar to JAX's jit, this transformation traces the function once to create
    a computation graph, then reuses that graph for future calls with new inputs.
    
    Example:
        @jit
        def f(x, y):
            return x * x + y
            
        result = f(2.0, 3.0)  # First call traces and caches
        result = f(4.0, 5.0)  # Reuses cached graph
    """
    
    def __init__(self, fn: Callable):
        super().__init__(fn)
        self.cache = {}
        
    def transform(self, fn: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            # Use a cache key that’s based on argument types (or ideally, shapes)
            cache_key = tuple(type(arg) for arg in args)
            if cache_key not in self.cache:
                traced_args = []
                input_nodes: List[Node] = []
                for arg in args:
                    if isinstance(arg, Node):
                        traced_args.append(arg)
                        input_nodes.append(arg)
                    else:
                        node = constant(arg)
                        traced_args.append(node)
                        input_nodes.append(node)
                output = fn(*traced_args, **kwargs)
                self.cache[cache_key] = CachedGraph(output, input_nodes)
            # The cached graph will update its input node values on each call.
            result = self.cache[cache_key](*args)
            return result
        return wrapped
