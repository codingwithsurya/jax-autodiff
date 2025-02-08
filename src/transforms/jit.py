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
    """Represents a cached computation graph for reuse."""
    def __init__(self, output_node: Node, input_nodes: List[Node]):
        self.output_node = output_node
        self.input_nodes = input_nodes
        self.compiled_fn = None
        
    def __call__(self, *args):
        # Update input values
        for node, arg in zip(self.input_nodes, args):
            node.value = arg
        return trace(self.output_node)

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
            # Create a cache key based on input types
            cache_key = tuple(type(arg) for arg in args)
            
            if cache_key not in self.cache:
                # First call: build and cache the computation graph
                traced_args = []
                for arg in args:
                    if isinstance(arg, Node):
                        traced_args.append(arg)
                    else:
                        traced_args.append(constant(arg))
                        
                output = fn(*traced_args, **kwargs)
                self.cache[cache_key] = lambda *args: output
                
            # Use the cached graph
            result = self.cache[cache_key](*args)
            
            # If the result is a Node, evaluate it
            if isinstance(result, Node):
                return trace(result)
            return result
            
        return wrapped

class CachedGraph:
    """Represents a cached computation graph for reuse."""
    def __init__(self, output_node: Node, input_nodes: List[Node]):
        self.output_node = output_node
        self.input_nodes = input_nodes
        self.compiled_fn = None
        
    def __call__(self, *args):
        # Update input values
        for node, arg in zip(self.input_nodes, args):
            node.value = arg
        return trace(self.output_node)

    def _make_cache_key(self, args, kwargs) -> str:
        """Create a cache key based on input shapes and types."""
        key_parts = []
        for arg in args:
            if isinstance(arg, (int, float)):
                key_parts.append(f"scalar_{type(arg).__name__}")
            elif isinstance(arg, torch.Tensor):
                key_parts.append(f"tensor_{list(arg.shape)}_{arg.dtype}")
            else:
                key_parts.append(f"other_{type(arg).__name__}")
        
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}_{type(v).__name__}")
            
        return "_".join(key_parts)
