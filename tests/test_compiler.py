"""
Tests: test_compiler.py
Unit tests for the compiler and optimization passes.
"""

import unittest
from src.core.tracer import constant, add
from src.core.compiler import Compiler

class TestCompiler(unittest.TestCase):
    def test_constant_folding(self):
        a = constant(2)
        b = constant(3)
        expr = add(a, b)
        compiler = Compiler()
        optimized_expr = compiler.compile(expr)
        self.assertEqual(optimized_expr.op, "const")
        self.assertEqual(optimized_expr.value, 5)

if __name__ == "__main__":
    unittest.main()