"""
Tests: test_autodiff.py
Unit tests for the automatic differentiation functionality.
"""

import unittest
from src.core.tracer import constant, add, mul
from src.core.autodiff import compute_gradients

class TestAutoDiff(unittest.TestCase):
    def test_addition_grad(self):
        a = constant(2)
        b = constant(3)
        expr = add(a, b)
        compute_gradients(expr)
        # For addition, the derivative with respect to each input is 1.
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, 1)

    def test_multiplication_grad(self):
        a = constant(2)
        b = constant(3)
        expr = mul(a, b)
        compute_gradients(expr)
        # For multiplication, gradients should be nonzero.
        self.assertNotEqual(a.grad, 0)
        self.assertNotEqual(b.grad, 0)

if __name__ == "__main__":
    unittest.main()