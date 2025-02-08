"""
Module: constant_folding.py
Implements constant folding and propagation optimization.
"""

import logging
import torch
from ..core.tracer import Node, constant

def optimize(graph: Node) -> Node:
    """
    Perform constant folding on the computation graph.
    
    Args:
        graph (Node): The root node.
    
    Returns:
        Node: The computation graph with constants folded.
    """
    def helper(node: Node) -> Node:
        # If already a constant, return as is
        if node.op == "const":
            return node

        # Recursively optimize inputs
        optimized_inputs = [helper(inp) for inp in node.inputs]
        node.inputs = optimized_inputs

        # Check if all inputs are constants
        if all(inp.op == "const" for inp in optimized_inputs):
            values = [inp.value for inp in optimized_inputs]
            try:
                if node.op == "add":
                    # Handle both scalar and tensor addition
                    if any(isinstance(v, torch.Tensor) for v in values):
                        folded_value = sum(torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in values)
                    else:
                        folded_value = sum(values)
                    logging.info(f"Constant folding: Replacing add node with constant {folded_value}")
                    return Node(op="const", value=folded_value)
                elif node.op == "mul":
                    # Handle both scalar and tensor multiplication
                    folded_value = values[0]
                    for v in values[1:]:
                        if isinstance(folded_value, torch.Tensor) or isinstance(v, torch.Tensor):
                            folded_value = torch.tensor(folded_value) * torch.tensor(v)
                        else:
                            folded_value *= v
                    logging.info(f"Constant folding: Replacing mul node with constant {folded_value}")
                    return Node(op="const", value=folded_value)
                else:
                    return node
            except Exception as e:
                logging.error(f"Error during constant folding: {e}")
                return node
        return node

    return helper(graph)
