"""
Module: cse.py
Implements Common Subexpression Elimination (CSE) optimization.
"""

import logging
from typing import Dict, Tuple
from ..core.tracer import Node

def optimize(graph: Node) -> Node:
    """
    Perform common subexpression elimination on the computation graph.
    
    Args:
        graph (Node): The root node of the computation graph.
    
    Returns:
        Node: The optimized computation graph.
    """
    subexpr_map: Dict[Tuple, Node] = {}

    def helper(node: Node) -> Node:
        if not node.inputs:
            return node
        new_inputs = [helper(inp) for inp in node.inputs]
        key = (node.op, tuple(inp.id for inp in new_inputs), node.value)
        if key in subexpr_map:
            logging.info(f"CSE: Merging node {node} with existing node {subexpr_map[key]}")
            return subexpr_map[key]
        else:
            node.inputs = new_inputs
            subexpr_map[key] = node
            return node

    return helper(graph)
