import unittest
import numpy as np
from LR4 import logist_next, logist_map


class TestChaos(unittest.TestCase):
    def test_next(self):
        #r=2.0, x_curr=0.5: 2.0 * 0.5 * (1 - 0.5) = 0.5
        self.assertAlmostEqual(logist_next(x_curr=0.5, r_val=2.0), 0.5, places=7)
        #r=3.0, x_curr=0.1: 3.0 * 0.1 * (1 - 0.1) = 0.27
        self.assertAlmostEqual(logist_next(x_curr=0.1, r_val=3.0), 0.27, places=7)
        #r=4.0, x_curr=0.25: 4.0 * 0.25 * (1 - 0.25) = 0.75
        self.assertAlmostEqual(logist_next(x_curr=0.25, r_val=4.0), 0.75, places=7)

    def test_logist_edge(self):
        #x_curr = 0, результат = 0
        self.assertAlmostEqual(logist_next(x_curr=0.0, r_val=3.5), 0.0, places=7)
        #x_curr = 1, результат = 0
        self.assertAlmostEqual(logist_next(x_curr=1.0, r_val=3.5), 0.0, places=7)

    def test_logist_map_zero(self):
        x0 = 0.5
        n_iter = 0
        r_val = 3.0
        expected = np.array([0.5])
        actual = logist_map(x0, n_iter, r_val)
        self.assertEqual(len(actual), n_iter + 1)
        self.assertTrue(np.allclose(actual, expected, atol=1e-7))

    def test_logist_map_one(self):
        x0 = 0.2
        n_iter = 1
        r_val = 3.0
        #logist_next(0.2, 3.0) = 3.0 * 0.2 * (1 - 0.2) = 0.6 * 0.8 = 0.48
        expected = np.array([0.2, 0.48])
        actual = logist_map(x0, n_iter, r_val)
        self.assertEqual(len(actual), n_iter + 1) #масив має буть n + 1
        self.assertTrue(np.allclose(actual, expected, atol=1e-7))

    def test_logist_map_two(self):
        x0 = 0.2
        n_iter = 2
        r_val = 3.0
        #x1 = 3.0 * 0.2 * (1 - 0.2) = 0.48
        #x2 = 3.0 * 0.48 * (1 - 0.48) = 0.7488
        expected = np.array([0.2, 0.48, 0.7488])
        actual = logist_map(x0, n_iter, r_val)
        self.assertEqual(len(actual), n_iter + 1)
        self.assertTrue(np.allclose(actual, expected, atol=1e-7))


if __name__ == '__main__':
    unittest.main()