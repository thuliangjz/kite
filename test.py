#!../bin/python3.6

import const
import unittest
import math
import util

class TestSampleFunction(unittest.TestCase):
    lst = [(0, 2), (1, 5), (2, 9)]
    f = util.SampleFunction(lst)
    def test_start_point(self):
        self.assertEqual(self.f.compute(0), 2)
    def test_end_point(self):
        self.assertEqual(self.f.compute(2), 9)
    def test_mid(self):
        self.assertEqual(self.f.compute(0.5), 3.5)
    def test_serial(self):
        self.assertEqual(self.f.compute(0.6), 3.8)
        self.assertEqual(self.f.compute(0.4), 3.2)

class TestConstantSolverCDLL(unittest.TestCase):
    'CDLL:constant-drag-linear-lift，drag的系数为常量而lift的系数为线性,以下所有样例的测试均遵循这个范式'
    def test_right_solution(self):
        '求解器得到的结果调用compute方法应该误差在delta内'
        solver = const.ConstantSolver()
        solver.set_constants(0.2, 1, 1.225, 5)
        solver.set_c_d(util.SampleFunction([(0, 0.5), (math.pi / 2, 0.5)]))
        solver.set_c_l(util.SampleFunction([(0, 0), (math.pi / 2, 5 * math.pi / 2)]))
        delta = 0.0001
        phi = solver.solve(delta=delta)
        self.assertLessEqual(solver.compute(phi), delta, "solution's function value too big")

    def test_strong_wind(self):
        '当风力非常大的时候phi应当与风速无关，只要c_l / c_d 近似等于 tan(phi)即可'
        solver1 = const.ConstantSolver()
        solver1.set_constants(0.2, 1, 1.225, 10000)
        solver1.set_c_d(util.SampleFunction([(0, 0.5), (math.pi / 2, 0.5)]))
        solver1.set_c_l(util.SampleFunction([(0, 0), (math.pi / 2, 5 * math.pi / 2)]))
        delta = 0.001
        phi1 = solver1.solve(delta=delta)

        solver2 = const.ConstantSolver()
        solver2.set_constants(0.2, 1, 1.225, 60000)
        solver2.set_c_d(util.SampleFunction([(0, 0.5), (math.pi / 2, 0.5)]))
        solver2.set_c_l(util.SampleFunction([(0, 0), (math.pi / 2, 5 * math.pi / 2)]))
        delta = 0.001
        phi2 = solver2.solve(delta=delta)

        self.assertLessEqual(abs(math.tan(phi1) -  math.tan(phi2)),  3 * delta, "not constant under strong wind")

    def test_greater_cl(self):
        '增大c_l的线性系数，应当使得phi增大'
        solver1 = const.ConstantSolver()
        solver1.set_constants(0.2, 1, 1.225, 5)
        solver1.set_c_d(util.SampleFunction([(0, 0.5), (math.pi / 2, 0.5)]))
        solver1.set_c_l(util.SampleFunction([(0, 0), (math.pi / 2, 4 * math.pi / 2)]))
        delta = 0.001
        phi1 = solver1.solve(delta=delta)

        solver2 = const.ConstantSolver()
        solver2.set_constants(0.2, 1, 1.225, 5)
        solver2.set_c_d(util.SampleFunction([(0, 0.5), (math.pi / 2, 0.5)]))
        solver2.set_c_l(util.SampleFunction([(0, 0), (math.pi / 2, 5 * math.pi / 2)]))
        delta = 0.001
        phi2 = solver2.solve(delta=delta)

        self.assertGreater(phi2, phi1, "phi should be greater when c_l increases")

    def test_greater_cd(self):
        '增大c_d的线性系数，应当使得phi减小'
        solver1 = const.ConstantSolver()
        solver1.set_constants(0.2, 1, 1.225, 5)
        solver1.set_c_d(util.SampleFunction([(0, 0.5), (math.pi / 2, 0.5)]))
        solver1.set_c_l(util.SampleFunction([(0, 0), (math.pi / 2, 4 * math.pi / 2)]))
        delta = 0.001
        phi1 = solver1.solve(delta=delta)

        solver2 = const.ConstantSolver()
        solver2.set_constants(0.2, 1, 1.225, 5)
        solver2.set_c_d(util.SampleFunction([(0, 1), (math.pi / 2, 1)])) #增大了c_d常量
        solver2.set_c_l(util.SampleFunction([(0, 0), (math.pi / 2, 4 * math.pi / 2)]))
        delta = 0.001
        phi2 = solver2.solve(delta=delta)
        
        self.assertLess(phi2, phi1, "phi should be less when c_d increases")
       

unittest.main()
