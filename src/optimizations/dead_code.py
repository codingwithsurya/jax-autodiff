"""
Module: dead_code.py
Implements Dead Code Elimination optimization.
"""

import logging
from ..core.tracer import Node

def optimize(graph: Node) -> Node:
    """
    Eliminate dead code from the computation graph by marking reachable nodes.
    
    Args:
        graph (Node): The root node.
    
    Returns:
        Node: The computation graph (unchanged in structure for this demo).
    """
    reachable = set()

    def mark(node: Node):
        if node.id in reachable:
            return
        reachable.add(node.id)
        for inp in node.inputs:
            mark(inp)

    mark(graph)
    logging.info("Dead code elimination: Completed marking reachable nodes.")
    return graph
