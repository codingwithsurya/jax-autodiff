"""
Module: visualization.py
Provides utilities to visualize the computation graph using Graphviz.
"""

from graphviz import Digraph
from ..core.tracer import Node

def visualize(graph: Node, filename: str = "graph"):
    """
    Visualize the computation graph and save it to a file.
    
    Args:
        graph (Node): The root node of the computation graph.
        filename (str): The base filename for the output (without extension).
    """
    dot = Digraph(comment='Computation Graph')
    visited = set()

    def add_node(node: Node):
        if node.id in visited:
            return
        visited.add(node.id)
        label = node.op if node.value is None else f"{node.op}({node.value})"
        dot.node(node.id, label)
        for inp in node.inputs:
            dot.edge(inp.id, node.id)
            add_node(inp)

    add_node(graph)
    dot.render(filename, view=True)