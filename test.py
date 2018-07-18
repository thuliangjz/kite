#!../bin/python3.6

import const
import unittest

class TestSampleFunction(unittest.TestCase):
    lst = [(0, 2), (1, 5), (2, 9)]
    f = const.SampleFunction(lst)
    def test_start_point(self):
        self.assertEqual(self.f.compute(0), 2)
    def test_end_point(self):
        self.assertEqual(self.f.compute(2), 9)
    def test_mid(self):
        self.assertEqual(self.f.compute(0.5), 3.5)
    def test_serial(self):
        self.assertEqual(self.f.compute(0.6), 3.8)
        self.assertEqual(self.f.compute(0.4), 3.2)

unittest.main()