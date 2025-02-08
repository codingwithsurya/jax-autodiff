"""
Module: fusion.py
Implements operation fusion optimization with hardware-specific patterns.
"""

import logging
from typing import List, Dict, Set, Tuple
from ..core.tracer import Node
from ..metal.metal_ops import get_device

# Fusion patterns that can be efficiently executed on Metal
METAL_FUSION_PATTERNS = {
    ('mul', 'add'): 'fma',  # Fused multiply-add
    ('mul', 'mul'): 'mul2',  # Consecutive multiplications
    ('add', 'add'): 'add2',  # Consecutive additions
    ('mul', 'div'): 'scale',  # Multiplication followed by division
}

def can_fuse(op1: Node, op2: Node) -> bool:
    """
    Determine if two operations can be fused based on operation type,
    hardware capabilities, and data dependencies.
    
    Args:
        op1 (Node): First operation.
        op2 (Node): Second operation.
    
    Returns:
        bool: True if the operations can be fused.
    """
    # Check if operations form a known fusion pattern
    pattern = (op1.op, op2.op)
    if pattern not in METAL_FUSION_PATTERNS:
        return False
        
    # Check data dependencies
    if len(op2.inputs) != 2 or op1 not in op2.inputs:
        return False
        
    # Check shapes if available
    if 'shape' in op1.metadata and 'shape' in op2.metadata:
        return op1.metadata['shape'] == op2.metadata['shape']
        
    return True

def create_fused_op(ops: List[Node]) -> Node:
    """
    Create a single fused operation node from multiple operations.
    The fused operation will be optimized for the current hardware.
    
    Args:
        ops (List[Node]): List of operations to fuse.
    
    Returns:
        Node: A new node representing the fused operation.
    """
    if len(ops) < 2:
        return ops[0]
        
    # Identify the fusion pattern
    pattern = tuple(op.op for op in ops[:2])
    fused_type = METAL_FUSION_PATTERNS.get(pattern)
    
    if not fused_type:
        return ops[0]
        
    # Create new node with fused operation
    fused = Node(op=fused_type, inputs=ops[0].inputs + [inp for op in ops[1:] for inp in op.inputs if inp != ops[0]])
    
    # Copy relevant metadata
    fused.metadata = {
        'fused_ops': [op.op for op in ops],
        'original_nodes': ops,
        'shape': ops[0].metadata.get('shape'),
    }
    
    return fused

def find_fusion_candidates(graph: Node) -> List[List[Node]]:
    """
    Find sequences of operations that can be fused together.
    
    Args:
        graph (Node): The root node of the computation graph.
        
    Returns:
        List[List[Node]]: Lists of nodes that can be fused together.
    """
    candidates = []
    visited = set()
    
    def visit(node: Node):
        if node.id in visited:
            return
        visited.add(node.id)
        
        # Check inputs for fusion opportunities
        for inp in node.inputs:
            visit(inp)
            
        # Try to build fusion groups starting at this node
        if node.inputs:
            group = [node]
            current = node
            while current.inputs and len(current.inputs) == 2:
                prev = current.inputs[0]  # Assume first input is the main operation
                if can_fuse(prev, current):
                    group.insert(0, prev)
                    current = prev
                else:
                    break
            
            if len(group) > 1:
                candidates.append(group)
    
    visit(graph)
    return candidates

def optimize(graph: Node) -> Node:
    """
    Fuse consecutive operations in the computation graph based on hardware-specific patterns.
    
    Args:
        graph (Node): The root node.
    
    Returns:
        Node: The optimized computation graph.
    """
    logging.info("Starting operation fusion optimization...")
    
    # Find fusion candidates
    fusion_groups = find_fusion_candidates(graph)
    
    if not fusion_groups:
        return graph
        
    # Create a mapping from old nodes to their fused replacements
    replacements = {}
    for group in fusion_groups:
        fused = create_fused_op(group)
        for old_node in group:
            replacements[old_node.id] = fused
            
    # Create a new graph with fused operations
    def replace_node(node: Node, visited: Set[str]) -> Node:
        if node.id in visited:
            return replacements.get(node.id, node)
            
        visited.add(node.id)
        
        if node.id in replacements:
            return replacements[node.id]
            
        # Recursively replace inputs
        new_inputs = [replace_node(inp, visited) for inp in node.inputs]
        if new_inputs != node.inputs:
            new_node = Node(op=node.op, inputs=new_inputs)
            new_node.metadata = node.metadata.copy()
            return new_node
            
        return node
        
    optimized = replace_node(graph, set())
    
    logging.info(f"Operation fusion complete. Found {len(fusion_groups)} fusion opportunities.")
    return optimized