"""
“2.5维”风筝求解器
风筝拥有多块可以自行设置位置的面板,风筝的质量仍然集中在一个点
风筝受到的力是3维的，但是风筝在运动的同时原点（基站）也在运动，且两者沿着z轴方向的速度相同
只有一根绳子拉着风筝，作用点即在这个点上
观察风筝运动的坐标系如下：
        y
        |
        |
        |
        o---------x
       /
      /
     z
注意重力始终沿着-y方向
"""

import sys
import os
import numpy as np
import reader_2_5d
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
#pylint: disable=wrong-import-position
import util
#pylint: enable=wrong-import-position


class Dynamic:
    """
    2.5维动态求解器
    所有的向量（包括点）都是以numpy中的1维向量来存储的
    多个向量通过2维array来存储
    每个向量对应矩阵的各个行(而不是通常的列向量)
    """
    PANEL_KEYS = ["c_l", "c_d", "plane_basis", "leading_edge", "ref_pt", "area"]
    BORDER_COND_KEYS = ["r0", "v0"]
    CONST_KEYS = ["v_wind", "mass", "density"]
    def __init__(self, reader, step_interval):
        self.__panels = []
        util.check_instance(reader, reader_2_5d.ReaderBase)
        self.__reader = reader
        self.__step_interval = step_interval
        self.__force_last = {}

    def add_panel(self, panel):
        """
        所有的key参见Dynamic.PANEL_KEYS
        暂定ref_point是f_d和f_l相同的作用点坐标，也是计算v_rel时风筝速度的参考点坐标,
        传入的panel中所有向量的基向量和计算风筝轨迹的坐标系的基向量是相同的,表示相应向量（点）在风筝位于x轴上时的方向（位置），
        绳子一端附着在观察坐标系中的原点，另一端附着在panel使用的坐标系的原点(这个点和风筝抽象的质点位置重合)
        plane_basis是两个彼此正交的单位向量,由于numpy的qr分解会导致正方向变化，故不对这组向量进行归一化
        leading_edge 是单位方向向量
        """
        util.check_keys(panel, self.PANEL_KEYS)
        util.check_instance(panel["c_l"], util.KiteFunction)
        util.check_instance(panel["c_d"], util.KiteFunction)
        util.check_vector(panel["leading_edge"], 3)
        util.check_vector(panel["ref_pt"], 3)
        util.check_vector(panel["plane_basis"], 2)
        for i in range(2):
            util.check_vector(panel["plane_basis"][i], 3)
        #正规化
        leading_edge = panel["leading_edge"]
        panel["leading_edge"] = leading_edge / np.linalg.norm(leading_edge)
        self.__panels.append(panel)

    def set_consts(self, consts):
        util.check_keys(consts, self.CONST_KEYS)
        #pylint: disable=w0201
        self.__v_wind = consts["v_wind"]
        util.check_vector(self.__v_wind, 3)
        self.__mass = consts["mass"]
        self.__density = consts["density"]
        #pylint: enable=w0201

    def init(self, border_cond):
        """
        设置边界条件
        r虽然是3维向量，但是其z分量始终为0
        """
        util.check_keys(border_cond, self.BORDER_COND_KEYS)
        util.check_vector(border_cond["r0"], 3)
        util.check_vector(border_cond["v0"], 3)
        self.__r = border_cond["r0"]
        self.__v = border_cond["v0"]
        self.__logic_timer = 0

    def step(self):
        val = self.__reader.read(self.__logic_timer)
        tp_reader = self.__reader.get_type()
        if tp_reader == reader_2_5d.ReaderBase.TYPE_FORCE:
            self.__step_read_force(val)
        else:
            self.__step_read_acceleration(val)

    def get_state(self):
        return (self.__r, self.__v)

    def get_force(self):
        return dict(**self.__force_last)

    def __step_read_force(self, force, **args):
        """
        根据拉力测试仪读取的拉力更新风筝的状态，
        也会被__step_read_acceleration进行调用,
        force只是一个值，如果args中有other键，则使用这个作为其他力
        """
        proj_xy = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        r_unit = self.__r / np.linalg.norm(self.__r)
        force_other = args.get("other", self.__compute_force_no_pull())
        force_pull = -force * r_unit    #拉力以径向的负方向为正方向
        acceleration = (force_other + force_pull) / self.__mass
        #仅在xy分量更新r
        self.__r = self.__r + proj_xy.dot(self.__v) * self.__step_interval
        self.__v = self.__v + acceleration * self.__step_interval
        self.__logic_timer += self.__step_interval
        self.__force_last["pull"] = force_pull

    def __step_read_acceleration(self, acceleration):
        """
        根据a和拉力T之间的仿射关系求出T，然后调用__step_read_acceleration
        """
        f_other = self.__compute_force_no_pull()
        pull = (f_other.dot(self.__r) + self.__mass * (
            self.__v.dot(self.__v) - \
            self.__r.dot(self.__v)**2 / self.__r.dot(self.__r) - \
            acceleration
        )) / np.linalg.norm(self.__r)
        self.__step_read_force(pull, other=f_other)

    def __compute_force_no_pull(self):
        rot_90_xy = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]])
        norm_r = np.linalg.norm(self.__r)
        r_unit = self.__r / norm_r
        proj_xy = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
        #位置转换矩阵
        transformation = np.array([
            r_unit, rot_90_xy.dot(r_unit), [0, 0, 1]
        ]).transpose()
        #用于计算附加速度的位置转换矩阵导数
        tmp = -self.__r.dot(self.__v) / (norm_r ** 3) * self.__r + self.__v / norm_r
        deri_transformation = proj_xy.dot(np.array([
            tmp, rot_90_xy.dot(tmp), [0, 0, 0]
        ]).transpose())

        lst_fl = []
        lst_fd = []
        for panel in self.__panels:
            v_rel = self.__v_wind - (self.__v + deri_transformation.dot(panel["ref_pt"]))
            dir_fl = np.cross(transformation.dot(panel["leading_edge"]), v_rel)
            dir_fl = dir_fl / np.linalg.norm(dir_fl) if np.linalg.norm(dir_fl) != 0 \
                else np.array([0, 0, 0])
            f_l = 0.5 * self.__density * panel["area"] * \
                panel["c_l"].compute(util.vec_plane_angle(v_rel,\
                    panel["plane_basis"].dot(transformation.transpose())))\
                * np.linalg.norm(v_rel) ** 2 * dir_fl
            f_d = 0.5 * self.__density * panel["area"] * \
                panel["c_d"].compute(util.vec_plane_angle(v_rel,\
                    panel["plane_basis"].dot(transformation.transpose())))\
                * np.linalg.norm(v_rel) * v_rel
            lst_fd.append(f_d)
            lst_fl.append(f_l)


        #重力沿着y轴方向
        force = sum(lst_fd + lst_fl) + np.array([0, self.__mass * -9.8, 0])
        self.__force_last["force_wind"] = sum(lst_fd + lst_fl)
        self.__force_last["force_fl"] = lst_fl.copy()
        self.__force_last["force_fd"] = lst_fd.copy()
        return force
