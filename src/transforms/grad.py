"""
Module: grad.py
Gradient transformation, similar to JAX's grad.
"""

from typing import Callable, Any, Optional
from .transform_base import Transform
from ..core.tracer import Node, constant, trace
from ..core.autodiff import compute_gradients
import torch

def grad(fn: Callable = None, argnums: Any = 0, has_aux: bool = False):
    """
    Gradient transformation.
    
    Similar to JAX's grad, this transformation converts a function into one that
    computes its gradient with respect to specified inputs.
    
    Args:
        fn (Callable): Function to transform.
        argnums (int or tuple): Which argument(s) to take gradient with respect to.
        has_aux (bool): Whether the function returns auxiliary data.
    """
    if fn is None:
        return lambda f: grad(f, argnums=argnums, has_aux=has_aux)
        
    if isinstance(argnums, int):
        argnums = (argnums,)
        
    class GradTransform(Transform):
        def __init__(self, fn: Callable):
            super().__init__(fn)
            self.argnums = argnums
            self.has_aux = has_aux
            
        def transform(self, fn: Callable) -> Callable:
            def wrapped(*args, **kwargs):
                # Convert inputs to nodes
                traced_args = []
                for i, arg in enumerate(args):
                    if i in self.argnums:
                        if isinstance(arg, Node):
                            traced_args.append(arg)
                        else:
                            traced_args.append(constant(arg))
                    else:
                        traced_args.append(arg)
                        
                # Run the function to build the computation graph
                output = fn(*traced_args, **kwargs)
                
                if self.has_aux:
                    output, aux = output
                    aux_value = trace(aux) if isinstance(aux, Node) else aux
                
                # If output is a Tensor, wrap it in a Node
                if isinstance(output, torch.Tensor):
                    output = Node(op='grad_output', inputs=traced_args, value=output)
                
                # Compute gradients
                compute_gradients(output)
                
                # Extract gradients for the specified arguments
                if len(self.argnums) == 1:
                    result = traced_args[self.argnums[0]].grad
                else:
                    result = tuple(traced_args[i].grad for i in self.argnums)
                    
                if self.has_aux:
                    return result, aux_value
                return result
                
            return wrapped
            
    return GradTransform(fn).transform(fn)

def value_and_grad(fn: Callable, argnums: Any = 0, has_aux: bool = False):
    """
    Transform a function to return both its value and gradient.
    Similar to JAX's value_and_grad.
    
    Args:
        fn: The function to transform.
        argnums: Which argument(s) to take gradient with respect to.
        has_aux: Whether the function returns auxiliary data.
        
    Returns:
        A function that returns a tuple (value, gradient).
    """
    
    def wrapped(*args, **kwargs):
        # Convert inputs to Node objects
        traced_args = []
        for i, arg in enumerate(args):
            if i in ((argnums,) if isinstance(argnums, int) else argnums):
                traced_args.append(constant(arg))
            else:
                traced_args.append(arg)
                
        # Run the function to build the computation graph
        output = fn(*traced_args, **kwargs)
        
        if has_aux:
            output, aux = output
            aux_value = trace(aux) if isinstance(aux, Node) else aux
            
        # Get the value before computing gradients
        value = trace(output)
        
        # Compute gradients
        compute_gradients(output)
        
        # Extract gradients
        if isinstance(argnums, int):
            grad_output = traced_args[argnums].grad
        else:
            grad_output = tuple(traced_args[i].grad for i in argnums)
            
        if has_aux:
            return (value, grad_output), aux_value
        return value, grad_output
        
    return wrapped
