"""
Module: compiler.py
Implements the compilation and optimization passes for the computation graph.
"""

import logging
from typing import Callable, List
from .tracer import Node
from ..optimizations import cse, constant_folding, dead_code, fusion, patterns

logging.basicConfig(level=logging.INFO)

class Compiler:
    def __init__(self):
        """
        Initialize the compiler with a list of optimization passes.
        """
        self.passes: List[Callable[[Node], Node]] = [
            constant_folding.optimize,
            cse.optimize,
            dead_code.optimize,
            fusion.optimize,
            # Apply pattern-based simplifications pass.
            lambda graph: apply_patterns(graph),
        ]

    def compile(self, graph: Node) -> Node:
        """
        Apply optimization passes to the computation graph.
        
        Args:
            graph (Node): The root of the computation graph.
        
        Returns:
            Node: The optimized computation graph.
        """
        optimized_graph = graph
        for opt_pass in self.passes:
            pass_name = getattr(opt_pass, '__name__', str(opt_pass))
            logging.info(f"Applying optimization pass: {pass_name}")
            optimized_graph = opt_pass(optimized_graph)
        return optimized_graph

def apply_patterns(graph: Node) -> Node:
    """
    Apply pattern-based simplifications to the computation graph.
    
    Args:
        graph (Node): The root node.
    
    Returns:
        Node: The simplified graph.
    """
    from ..optimizations.patterns import patterns

    def helper(node: Node) -> Node:
        new_inputs = [helper(inp) for inp in node.inputs]
        node.inputs = new_inputs
        for pattern in patterns:
            if pattern.match(node):
                logging.info(f"Pattern matched for node {node.id}, applying replacement.")
                return pattern.replace(node)
        return node

    return helper(graph)
