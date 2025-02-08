"""
Tests for JAX-style transformations (jit, vmap, grad).
"""

import unittest
import torch
import numpy as np
from src.transforms.jit import jit
from src.transforms.vmap import vmap
from src.transforms.grad import grad, value_and_grad
from src.core.tracer import constant, add, mul, div, trace

class TestTransformations(unittest.TestCase):
    def test_jit(self):
        """Test just-in-time compilation transformation."""
        
        @jit
        def f(x):
            return add(mul(x, x), constant(1.0))
            
        # Test scalar input
        result = f(2.0)
        self.assertAlmostEqual(result, 5.0)
        
        # Test repeated calls (should use cached graph)
        result1 = f(3.0)
        result2 = f(3.0)
        self.assertAlmostEqual(result1, result2)
        
    def test_vmap(self):
        """Test vectorized mapping transformation."""
        
        @vmap
        def f(x):
            return add(mul(x, x), constant(1.0))
            
        # Test batch input
        x = torch.tensor([1.0, 2.0, 3.0])
        result = f(x)
        expected = torch.tensor([2.0, 5.0, 10.0])  # x^2 + 1 for each input
        torch.testing.assert_close(result, expected)
        
    def test_grad(self):
        """Test gradient transformation."""
        
        def f(x, y):
            return add(mul(x, x), y)
            
        # Test gradient with respect to first argument
        df_dx = grad(f)
        grad_x = df_dx(2.0, 3.0)
        self.assertAlmostEqual(grad_x, 4.0)  # d/dx(x^2 + y) = 2x
        
        # Test gradient with respect to second argument
        df_dy = grad(f, argnums=1)
        grad_y = df_dy(2.0, 3.0)
        self.assertAlmostEqual(grad_y, 1.0)  # d/dy(x^2 + y) = 1
        
        # Test multiple argument gradients
        df_dxy = grad(f, argnums=(0, 1))
        grad_x, grad_y = df_dxy(2.0, 3.0)
        self.assertAlmostEqual(grad_x, 4.0)
        self.assertAlmostEqual(grad_y, 1.0)
        
    def test_value_and_grad(self):
        """Test value_and_grad transformation."""
        
        def f(x):
            return add(mul(x, x), constant(1.0))
            
        f_with_grad = value_and_grad(f)
        value, gradient = f_with_grad(2.0)
        
        self.assertAlmostEqual(value, 5.0)  # x^2 + 1 = 4 + 1 = 5
        self.assertAlmostEqual(gradient, 4.0)  # d/dx(x^2 + 1) = 2x = 4
        
    def test_transform_composition(self):
        """Test composition of multiple transformations."""
        
        # Function that computes sum(x^2 + 1) for a batch of x
        @jit
        @vmap
        def f(x):
            return add(mul(x, x), constant(1.0))
            
        # Compute gradient of the sum with respect to all inputs
        def sum_f(x):
            return torch.sum(f(x))
            
        grad_sum_f = grad(sum_f)
        
        # Test with batch input
        x = torch.tensor([1.0, 2.0, 3.0])
        result = grad_sum_f(x)
        
        # Expected gradient is 2x for each input
        expected = torch.tensor([2.0, 4.0, 6.0])
        torch.testing.assert_close(result, expected)
        
    def test_aux_outputs(self):
        """Test transformations with auxiliary outputs."""
        
        def f(x):
            y = add(mul(x, x), constant(1.0))
            return y, x  # Return value and auxiliary data
            
        # Test grad with auxiliary outputs
        grad_f = grad(f, has_aux=True)
        gradient, aux = grad_f(2.0)
        self.assertAlmostEqual(gradient, 4.0)
        self.assertAlmostEqual(aux, 2.0)
        
        # Test value_and_grad with auxiliary outputs
        f_with_grad = value_and_grad(f, has_aux=True)
        (value, gradient), aux = f_with_grad(2.0)
        self.assertAlmostEqual(value, 5.0)
        self.assertAlmostEqual(gradient, 4.0)
        self.assertAlmostEqual(aux, 2.0)

if __name__ == '__main__':
    unittest.main()
