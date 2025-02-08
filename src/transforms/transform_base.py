"""
Module: transform_base.py
Base classes and utilities for JAX-style function transformations.
"""

from typing import Callable, Any, Tuple, List, Dict
from functools import wraps
import inspect
from ..core.tracer import Node, constant, trace

class Transform:
    """Base class for all function transformations."""
    
    def __init__(self, fn: Callable):
        self.fn = fn
        self.transformed_fn = None
        wraps(fn)(self)
        
    def __call__(self, *args, **kwargs):
        if self.transformed_fn is None:
            self.transformed_fn = self.transform(self.fn)
        return self.transformed_fn(*args, **kwargs)
    
    def transform(self, fn: Callable) -> Callable:
        """Transform the function. Must be implemented by subclasses."""
        raise NotImplementedError
