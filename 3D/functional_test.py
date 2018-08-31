"""
3D模型的自动化功能测试
"""
import sys
import os
import unittest
import numpy as np
import dynamic_3d
import reader_3d

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
#pylint: disable=wrong-import-position
import util
#pylint: enable=wrong-import-position

class CL3DConst(util.KiteFunction):
    def compute(self, val):
        return 0.5

class CD3DConst(util.KiteFunction):
    def compute(self, val):
        return 0.1

class Reader3DStableLength(reader_3d.Reader3DBase):
    def read(self, time):
        return [0, 0]
    def get_type(self):
        return self.TYPE_ACCELERATION
    def get_attach_pts(self):
        return SIMPLE_KITE["attach_pts"].copy()

class Reader3DConstF(reader_3d.Reader3DBase):
    def read(self, time):
        return [2.5, 2.5]
    def get_type(self):
        return self.TYPE_FORCE
    def get_attach_pts(self):
        return SIMPLE_KITE["attach_pts"].copy()

#pylint: disable=W0105
"""
三块夹角为120度的板子，
高度0.5，
左右长度各0.5，中间长度1
原点在等腰梯形底边中点:
    ^ y   _______
    |   /         \
    |  /_ _ _._ _ _\
        ---> x

风筝被z平面上下各切一半
"""
#pylint: enable=W0105
SIMPLE_KITE = {
    "panels":[
        {
            "c_l": CL3DConst(),
            "c_d": CD3DConst(),
            "plane_basis": np.array([
                (3 ** 0.5 / 2, 0.5, 0),
                (0, 0, 1),
            ]),
            "leading_edge": np.array([3 ** 0.5 / 2, 0.5, 0]),
            "ref_pt": np.array([-0.5 - 3 ** 0.5 / 8, 1 / 8, 0]),
            "area": 0.25,
        },
        {
            "c_l": CL3DConst(),
            "c_d": CD3DConst(),
            "plane_basis": np.array([
                (1, 0, 0),
                (0, 0, 1),
            ]),
            "leading_edge": np.array([1, 0, 0]),
            "ref_pt": np.array([0, 1 / 4, 0]),
            "area": 0.5,
        },
        {
            "c_l": CL3DConst(),
            "c_d": CD3DConst(),
            "plane_basis": np.array([
                (3 ** 0.5 / 2, -0.5, 0),
                (0, 0, 1),
            ]),
            "leading_edge": np.array([3 ** 0.5 / 2, -0.5, 0]),
            "ref_pt": np.array([0.5 + 3 ** 0.5 / 8, 1 / 8, 0]),
            "area": 0.25,
        },
    ],
    "mass_pts":[
        #总重0.2kg
        {
            "m": 0.05,
            "r": np.array([-0.5 - 3 ** 0.5 / 8, 1 / 8, 0]),
        },
        {
            "m": 0.1,
            "r": np.array([0, 1 / 4, 0]),
        },
        {
            "m": 0.05,
            "r": np.array([0.5 + 3 ** 0.5 / 8, 1 / 8, 0])
        },
    ],
    #两根绳子，连接点不直接在风筝上
    "attach_pts":[
        np.array([-0.5, -1 / 4, 0]),
        np.array([0.5, -1 / 4, 0]),
    ],
    "v_wind": np.array([0, 5, 0]),
    "density": 1.225
}

def create_solver(data, reader, step):
    """
    创建未调用init的solver
    """
    solver = dynamic_3d.Dynamic3D(reader, step)
    for panel in data["panels"]:
        solver.add_panel(panel)
    for mass_pt in data["mass_pts"]:
        solver.add_mass_point(mass_pt)
    return solver

class TestSimpleKite(unittest.TestCase):
    def test_launch_v(self):
        '''
        风筝从y轴上起飞，起飞时z方向的速度应该增加
        '''
        reader = Reader3DConstF()
        reader.set_attach_pts(SIMPLE_KITE["attach_pts"])
        solver = create_solver(SIMPLE_KITE, reader, 0.01)
        solver.init({
            "T0": np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
            "r0": np.array([0, 10, 0]),
            "v0": np.array([0, 0, 2]),
            "L0": np.array([0, 0, 0]),
            "v_wind": np.array([0, 5, 0]),
            "density": SIMPLE_KITE["density"]
        })
        solver.step()
        state = solver.get_state()
        self.assertGreater(state["vc"][2], 2)
    def test_angular_v_delta(self):
        '''
        刚开始时角动量变化不应过大
        '''
        reader = Reader3DConstF()
        reader.set_attach_pts(SIMPLE_KITE["attach_pts"])
        solver = create_solver(SIMPLE_KITE, reader, 0.01)
        solver.init({
            "T0": np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
            "r0": np.array([0, 10, 0]),
            "v0": np.array([0, 0, 2]),
            "L0": np.array([0, 0, 0]),
            "v_wind": np.array([0, 5, 0]),
            "density": SIMPLE_KITE["density"]
        })
        step_count = 10
        v_angular_last = np.array([0, 0, 0])
        for _ in range(step_count):
            solver.step()
            self.assertLess(np.linalg.norm(solver.get_state()["v_angular"] - v_angular_last), 1)
            v_angular_last = solver.get_state()["v_angular"]
    def test_stable_len_rope(self):
        '''
        在使用定长的绳子的读取器时，至少在很短的一段时间内线的长度不应该发生太大的变化
        '''
        reader = Reader3DStableLength()
        reader.set_attach_pts(SIMPLE_KITE["attach_pts"])
        solver = create_solver(SIMPLE_KITE, reader, 0.01)
        solver.init({
            "T0": np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]),
            "r0": np.array([0, 10, 0]),
            "v0": np.array([0, 0, 2]),
            "L0": np.array([0, 0, 0]),
            "v_wind": np.array([0, 5, 0]),
            "density": SIMPLE_KITE["density"]
        })
        rope_length0 = solver.get_rope_length()
        step_count = 10     #当前模型只在迭代次数非常少的时候拉力才比较“合理”
        for _ in range(step_count):
            solver.step()
            rope_length1 = solver.get_rope_length()
            for tpl in zip(rope_length0, rope_length1):
                self.assertLess(abs(tpl[0] - tpl[1]), tpl[0]*0.01)
            rope_length0 = rope_length1

unittest.main()
