"""
功能测试：
version: 1.0
基本思路：给定一系列的外界条件和风筝结构，观察一段时间内风筝的位矢和速度
"""
import unittest
import sys
import os
import math
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

class FunTestCLDoublePeak(util.KiteFunction):
    def __init__(self):
        self.__compute_theta = util.SampleFunction([
            (0, 0.2),
            (math.pi / 3, 0.7),
            (math.pi / 2, 0.4),
            (math.pi * 2 / 3, 0.7),
            (math.pi, 0.2)
        ])
        self.__compute_phi = util.SampleFunction([
            (-math.pi, 0.2),
            (-math.pi / 2, 0.7),
            (0, 0.2),
            (math.pi / 2, 0.7),
            (math.pi, 0.2)
        ])
    def compute(self, val):
        return self.__compute_phi.compute(val[0])\
            * self.__compute_theta.compute(val[1])

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

def create_simple_solver(**args):
    """
    可指定参数：
        rotation: 风筝的旋转矩阵
        reader: 读取器类型
        r0: 起始位矢
        v0: 起始速度
    """
    panels = [
        create_panel([(1, 0, 0), (0, 1, 0)], (1, 0, 0), (0, 0, 0), 2, \
            FunTestCLConst(), FunTestCDConst()),
        create_panel([(1, 0, 1), (0, 1, 0)], (1, 0, 1), (1.5, 0, 0.5), 2 ** 0.5,\
            FunTestCLConst(), FunTestCDConst()),
        create_panel([(1, 0, -1), (0, 1, 0)], (1, 0, -1), (-1.5, 0, 0.5), 2 ** 0.5,\
            FunTestCLConst(), FunTestCDConst()),
    ]
    #zOx平面沿y轴负方向旋转90度
    rotation = args.get("rotation")
    rotate_panels(panels, rotation)
    reader = args.get("reader")
    solver = dynamic_2_5d.Dynamic(reader, 0.01)
    for panel in panels:
        solver.add_panel(panel)
    solver.set_consts({
        "v_wind": np.array([5, 0, 0]),
        "mass": 0.5,
        "density": 1.225
    })
    solver.init({
        "r0":np.array(args.get("r0")),
        "v0":np.array(args.get("v0"))
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
        solver = create_simple_solver(rotation=np.array([
            (0, 0, -1),
            (0, 1, 0),
            (1, 0, 0),
        ]), reader=FunTestReaderForceConst(), r0=(10, 0, 0), v0=(0, 2, 0))
        step_count = 10
        for i in range(step_count):
            solver.step()
            status = solver.get_state()
            self.assertGreater(status[1][1], 0, "迭代%d:风筝速度不是向上的"%i)
            self.assertEqual(status[1][2], 0, "迭代%d:风筝z方向有速度"%i)

    def test_v_smooth(self):
        """
        在参数选取合适的情况下，速度的变化量不应过大（这里取1作为阈值）
        """
        solver = create_simple_solver(rotation=np.array([
            (0, 0, -1),
            (0, 1, 0),
            (1, 0, 0),
        ]), reader=FunTestReaderForceConst(), r0=(10, 0, 0), v0=(0, 2, 0))
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
        solver = create_simple_solver(rotation=np.array([
            (0, 0, -1),
            (0, 1, 0),
            (1, 0, 0),
        ]), reader=FunTestReaderAccelerationStable(), r0=(10, 0, 0), v0=(0, 2, 0))
        step_count = 100
        status = solver.get_state()
        length_init = np.linalg.norm(status[0])
        for i in range(step_count):
            length = np.linalg.norm(solver.get_state()[0])
            self.assertLess(abs(length - length_init) / length_init, 0.01, "迭代至%d时绳子长度已经有较大变化"%i)

class FunctionTestSideSliding(unittest.TestCase):
    def test_wf_z(self):
        """
        风筝侧着飞行，leading_edge位于xOy平面内，具有+z方向的加速度
        此时受到的风力在z方向应该有分量
        """
        solver = create_simple_solver(rotation=np.array([
            (0, 0, -1),
            (-1, 0, 0),
            (0, 1, 0),
        ]), reader=FunTestReaderAccelerationStable(), r0=(10, 10, 0), v0=(0, 0, 2))
        step_count = 10
        for _ in range(step_count):
            solver.step()
            force_wind = solver.get_force()["force_wind"]
            self.assertGreater(force_wind[2], 0, "侧向飞行受到风力z分量异常")

    def test_wf_y(self):
        """
        风筝侧着飞行，leading_edge位于xOy平面内，具有+z方向的加速度
        """
        solver = create_simple_solver(rotation=np.array([
            (0, 0, -1),
            (-1, 0, 0),
            (0, 1, 0),
        ]), reader=FunTestReaderAccelerationStable(), r0=(10, 10, 0), v0=(0, 0, 5))
        step_count = 10
        for _ in range(step_count):
            solver.step()
            force_wind = solver.get_force()["force_wind"]
            self.assertGreater(force_wind[1], 0)

    def test_inverse_direction(self):
        """
        风筝仍然侧着飞行，但是leading edge的朝向和前面相反,
        此时应该受到的风力也反向
        """
        solver = create_simple_solver(rotation=np.array([
            (0, 0, -1),
            (1, 0, 0),
            (0, -1, 0),
        ]), reader=FunTestReaderAccelerationStable(), r0=(10, 10, 0), v0=(0, 0, -2))
        step_count = 10
        for _ in range(step_count):
            solver.step()
            force_wind = solver.get_force()["force_wind"]
            self.assertLess(force_wind[2], 0, "侧向飞行受到风力z分量异常")
            self.assertGreater(force_wind[1], 0, "侧向飞行受到风力应该有向上分量")

class FunctionTestExtremeInput(unittest.TestCase):
    def test_zero_fl(self):
        """
        有一个panel的leading_edge平行于风速并且风筝速度为0时检查是否正常
        这个样例检测的情形类似与launch但是风筝不是正对着风向的
        """
        tmp = 2 ** 0.5 / 2
        solver = create_simple_solver(rotation=np.array([
            (tmp, 0, -tmp),
            (0, 1, 0),
            (tmp, 0, tmp),
        ]), reader=FunTestReaderAccelerationStable(), r0=(10, 0, 0), v0=(0, 0, 0))
        step_count = 10
        for _ in range(step_count):
            solver.step()
            force_wind = solver.get_force()["force_wind"]
            self.assertGreater(force_wind[1], 0)

def create_single_panel(**args):
    """
    可定义参数：
    rotation
    reader
    r0
    v0
    """
    panel = create_panel([(1, 0, 0), (0, 1, 0)], (1, 0, 0), (0, 0, 0), 2, \
        FunTestCLDoublePeak(), FunTestCDConst())
    rotation = args["rotation"]
    panel["plane_basis"] = rotation.dot(panel["plane_basis"].transpose()).transpose()
    panel["leading_edge"] = rotation.dot(panel["leading_edge"])
    panel["ref_pt"] = rotation.dot(panel["ref_pt"])

    reader = args["reader"]
    solver = dynamic_2_5d.Dynamic(reader, 0.01)
    solver.add_panel(panel)
    solver.set_consts({
        "v_wind": np.array([5, 0, 0]),
        "mass": 0.5,
        "density": 1.225
    })
    solver.init({
        "r0": args["r0"],
        "v0": args["v0"]
    })
    return solver


class FunctionTestPlaneAngle(unittest.TestCase):
    """
    测试风与平面所加的角度不同时风力的区别
    """
    def test_vari_theta(self):
        """
        比较位于地面（正面风）和在空中（有倾角）时受到的f_l
        """
        solver = create_single_panel(rotation=np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ]), reader=FunTestReaderForceConst(), r0=np.array((10, 0, 0)), v0=np.array((0, 0, 0)))
        solver.step()
        force0 = solver.get_force()["force_wind"]

        solver = create_single_panel(rotation=np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ]), reader=FunTestReaderForceConst(), r0=np.array((10, 10, 0)), v0=np.array((0, 0, 0)))
        solver.step()
        force1 = solver.get_force()["force_wind"]
        self.assertGreater(force1[1], force0[1], "有倾角时c_l更大，f_l应该更大")

    def test_vari_phi(self):
        """
        控制theta相同而改变phi角进行比较，
        第一种情形是正面风但是沿着y轴有转动
        第二种情形是第一种情形沿着plane_basis决定平面法向量转动90度得到
        """
        tmp = 2 ** 0.5 / 2
        solver = create_single_panel(rotation=np.array([
            [-tmp, 0, -tmp],
            [0, 1, 0],
            [tmp, 0, -tmp]
        ]), reader=FunTestReaderForceConst(), r0=np.array((10, 0, 0)), v0=np.array((0, 0, 0)))
        solver.step()
        force1 = solver.get_force()["force_fl"]

        #绕着上面的面板的法向量再转90度
        solver = create_single_panel(rotation=np.array([
            [0, -tmp, -tmp],
            [-1, 0, 0],
            [0, tmp, -tmp]
        ]), reader=FunTestReaderForceConst(), r0=np.array((10, 0, 0)), v0=np.array((0, 0, 0)))
        solver.step()
        force2 = solver.get_force()["force_fl"]
        self.assertLess(np.linalg.norm(force1),
                        np.linalg.norm(force2),
                        "正向面对风时phi最小")
if __name__ == "__main__":
    unittest.main()
