
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import unittest
import numpy as np

from core.variable import  Variable
from core.arithmetic.pointwise import square

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(2.0)
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

