"""
Module: patterns.py
Defines a simple pattern matching system for algebraic simplifications.
"""

from typing import Callable
from src.core.tracer import Node, constant

class Pattern:
    def __init__(self, match_fn: Callable[[Node], bool], replace_fn: Callable[[Node], Node]):
        """
        Initialize a pattern with a matching function and a replacement function.
        
        Args:
            match_fn (Callable[[Node], bool]): Returns True if the node matches the pattern.
            replace_fn (Callable[[Node], Node]): Returns the replacement node.
        """
        self.match = match_fn
        self.replace = replace_fn

patterns = [
    # Simplify: x * 0 = 0
    Pattern(
        match_fn=lambda n: n.op == 'mul' and any(i.value == 0 for i in n.inputs if i.value is not None),
        replace_fn=lambda n: constant(0)
    ),
    # Simplify: x * 1 = x (assuming one non-one input)
    Pattern(
        match_fn=lambda n: n.op == 'mul' and any(i.value == 1 for i in n.inputs if i.value is not None),
        replace_fn=lambda n: next(i for i in n.inputs if i.value != 1)
    )
]