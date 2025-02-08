"""
Module: metal_ops.py
Provides optimized Metal Performance Shaders (MPS) operations for Apple Silicon GPUs.
"""

import torch
import numpy as np

def get_device():
    """
    Get the appropriate device (MPS if available, else CPU).
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_tensor(x, device=None):
    """
    Convert input to tensor and move to appropriate device.
    
    Args:
        x: Input value (scalar, numpy array, or tensor)
        device: Target device (if None, uses default from get_device())
    
    Returns:
        torch.Tensor: Tensor on the specified device
    """
    if device is None:
        device = get_device()
    
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    else:
        return torch.tensor(x, device=device)

def metal_add(a, b):
    """
    Perform element-wise addition using Metal acceleration.
    
    Args:
        a: First operand (scalar, numpy array, or tensor)
        b: Second operand (scalar, numpy array, or tensor)
    
    Returns:
        torch.Tensor: Result of a + b
    """
    device = get_device()
    a_tensor = to_tensor(a, device)
    b_tensor = to_tensor(b, device)
    return a_tensor + b_tensor

def metal_mul(a, b):
    """
    Perform element-wise multiplication using Metal acceleration.
    
    Args:
        a: First operand (scalar, numpy array, or tensor)
        b: Second operand (scalar, numpy array, or tensor)
    
    Returns:
        torch.Tensor: Result of a * b
    """
    device = get_device()
    a_tensor = to_tensor(a, device)
    b_tensor = to_tensor(b, device)
    return a_tensor * b_tensor

def metal_div(a, b):
    """
    Perform element-wise division using Metal acceleration.
    
    Args:
        a: First operand (scalar, numpy array, or tensor)
        b: Second operand (scalar, numpy array, or tensor)
    
    Returns:
        torch.Tensor: Result of a / b
    """
    device = get_device()
    a_tensor = to_tensor(a, device)
    b_tensor = to_tensor(b, device)
    return a_tensor / b_tensor
