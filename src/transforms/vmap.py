"""
Module: vmap.py
Vectorized mapping transformation, similar to JAX's vmap.
"""

from typing import Callable, Any, Tuple, List, Optional
import torch
from .transform_base import Transform
from ..core.tracer import Node, constant, trace
from ..metal.metal_ops import get_device, to_tensor

def vmap(fn: Callable = None):
    """
    Vectorizing map transformation.
    
    Similar to JAX's vmap, this transformation converts a function that operates on
    single elements into one that operates on vectors/batches.
    """
    if fn is None:
        return lambda f: vmap(f)
        
    class VmapTransform(Transform):
        def __init__(self, fn: Callable):
            super().__init__(fn)
            
        def transform(self, fn: Callable) -> Callable:
            def wrapped(*args, **kwargs):
                # Convert inputs to tensors and create batch dimension
                batched_args = []
                for arg in args:
                    if isinstance(arg, (list, tuple)):
                        arg = torch.tensor(arg)
                    if isinstance(arg, torch.Tensor):
                        # Move tensor to CPU for consistency
                        batched_args.append(arg.cpu())
                    else:
                        # Broadcast scalar to match batch size
                        batch_size = max(arg.shape[0] for arg in batched_args) if batched_args else 1
                        batched_args.append(torch.full((batch_size,), arg).cpu())
                
                # Apply function to each element
                results = []
                for i in range(len(batched_args[0])):
                    # Extract i-th element from each argument
                    single_args = [arg[i].reshape(()) for arg in batched_args]  # Convert to scalar tensors
                    # Apply function
                    result = fn(*single_args, **kwargs)
                    # Only trace if result is a Node
                    if isinstance(result, Node):
                        result = trace(result)
                    results.append(result)
                
                # Stack results along batch dimension and ensure CPU
                stacked = torch.stack(results).cpu()
                
                # Squeeze any extra dimensions
                if stacked.shape[-1] == 1:
                    stacked = stacked.squeeze(-1)
                
                # Return the stacked tensor directly
                return stacked
                
            return wrapped
            
    return VmapTransform(fn)
