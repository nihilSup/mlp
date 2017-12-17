"tests for neyron class"
import unittest
import mlp

class TestNeyron(unittest.TestCase):
    "test case"
    def test_inputs_sigmoid(self):
        "test inputs"
        neyron = mlp.Neyron(mlp.sigmoid, 5)
        x_values = [1] * 5
        self.assertEqual(neyron.process_input(x_values), 3)
        self.assertAlmostEqual(neyron(x_values), 0.9525741268, places=10)

if __name__ == '__main__':
    unittest.main()
