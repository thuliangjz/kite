"""
功能测试：
version: 0.1
基本思路：给定一系列的外界条件和风筝结构，观察一段时间内风筝的位矢和速度
"""
import unittest
import sys
import os
import numpy as np
import dynamic_2_5d
import reader_2_5d

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
#pylint: disable=wrong-import-position
import util
#pylint: enable=wrong-import-position

class FunTestReaderForceConst(reader_2_5d.ReaderBase):
    def read(self, time):
        return 50
    def get_type(self):
        return self.TYPE_FORCE

class FunTestReaderAccelerationStable(reader_2_5d.ReaderBase):
    def read(self, time):
        return 0
    def get_type(self):
        return self.TYPE_ACCELERATION

class FunTestCLConst(util.KiteFunction):
    def compute(self, val):
        return 0.5

class FunTestCDConst(util.KiteFunction):
    def compute(self, val):
        return 0.2



def create_panel(plane_basis, leading_edge, ref_pt, area, c_l, c_d):
    return {
        "plane_basis": np.array(plane_basis),
        "leading_edge": np.array(leading_edge),
        "ref_pt": np.array(ref_pt),
        "area": area,
        "c_l": c_l,
        "c_d": c_d
    }

def rotate_panels(lst_panel, transformation):
    for panel in lst_panel:
        panel["plane_basis"] = transformation.dot(panel["plane_basis"].transpose()).transpose()
        panel["leading_edge"] = transformation.dot(panel["leading_edge"])
        panel["ref_pt"] = transformation.dot(panel["ref_pt"])

def create_launch_solver(is_acc_reader):
    panels = [
        create_panel([(1, 0, 0), (0, 1, 0)], (1, 0, 0), (0, 0, 0), 2, \
            FunTestCLConst(), FunTestCDConst()),
        create_panel([(1, 0, 1), (0, 1, 0)], (1, 0, 1), (1.5, 0, 0.5), 2 ** 0.5,\
            FunTestCLConst(), FunTestCDConst()),
        create_panel([(1, 0, -1), (0, 1, 0)], (1, 0, -1), (-1.5, 0, 0.5), 2 ** 0.5,\
            FunTestCLConst(), FunTestCDConst()),
    ]
    #zOx平面沿y轴负方向旋转90度
    rotation = np.array([
        (0, 0, -1),
        (0, 1, 0),
        (1, 0, 0),
    ])
    rotate_panels(panels, rotation)
    reader = FunTestReaderAccelerationStable() if is_acc_reader\
        else FunTestReaderForceConst()
    solver = dynamic_2_5d.Dynamic(reader, 0.01)
    for panel in panels:
        solver.add_panel(panel)
    solver.set_consts({
        "v_wind": np.array([5, 0, 0]),
        "mass": 0.5,
        "density": 1.225
    })
    solver.init({
        "r0":np.array((10, 0, 0)),
        "v0":np.array((0, 2, 0))
    })
    return solver

class FunctionTest2DLaunch(unittest.TestCase):
    """
    使用2维中发射风筝的场景进行3维情形检测
    """
    def test_v_direction(self):
        """
        刚开始上升时风筝的速度应该有向上（沿着+y方向）的分量，同时z方向不应该有分量
        """
        solver = create_launch_solver(False)
        step_count = 10
        for i in range(step_count):
            solver.step()
            status = solver.get_state()
            self.assertGreater(status[1][1], 0, "迭代%d:风筝速度不是向上的"%i)
            self.assertEqual(status[1][2], 0, "迭代%d:风筝z方向有速度"%i)

    def test_v_smooth(self):
        """
        如果风筝向上
        """
        solver = create_launch_solver(False)
        step_count = 10
        status0 = solver.get_state()
        for i in range(step_count):
            solver.step()
            status1 = solver.get_state()
            delta = status1[1] - status0[1]
            for idx, axis in enumerate(delta):
                self.assertLess(abs(axis), 1, "迭代%d: 第%d速度分量增量过大"%(i, idx))
            status0 = status1

    def test_stable_length(self):
        """
        如果使用的是绳子加速度的Reader，那么在若干次迭代之后绳子的长度不应该发生较大的变化
        """
        solver = create_launch_solver(True)
        step_count = 100
        status = solver.get_state()
        length_init = np.linalg.norm(status[0])
        for i in range(step_count):
            length = np.linalg.norm(solver.get_state()[0])
            self.assertLess(abs(length - length_init) / length_init, 0.01, "迭代至%d时绳子长度已经有较大变化"%i)

if __name__ == "__main__":
    unittest.main()
