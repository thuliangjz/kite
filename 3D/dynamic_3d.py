"""
实现3维风筝模拟求解器
"""
import numpy as np
import util
import reader_3d
class Dynamic3D:
    """
    三维风筝模拟器
    调用模式:
    dynamic = Dynamic3D(reader, step_interval)
    dynamic.add_pannel(...)
    dynamic.add_mass_pt(...)
    dynamic.init(...)
    dynamic.step()
    dynamic.get_state()
    """
    PANEL_KEYS = ["c_l", "c_d", "plane_basis", "leading_edge", "ref_pt", "area"]
    MASS_PT_KEYS = ["m", "r"]
    INIT_COND_KEYS = ["T0", "r0", "v0", "L0", "v_wind", "density"]
    def __init__(self, reader, step_interval):
        #除了self.__transformation，剩下的所有向量都是1Darray(行向量), 矩阵是通常写法的转置
        if not isinstance(reader, reader_3d.Reader3DBase):
            raise ValueError("reader must be subclass of Reader3DBase")
        self.__reader = reader
        self.__panels = []
        self.__mass_pts = []
        self.__step_interval = step_interval
    def add_pannel(self, pannel):
        """
        所有的key参见Dynamic3D.PANNEL_KEYS
        暂定ref_point是f_d和f_l相同的作用点坐标，也是计算v_rel时风筝速度的参考点坐标,
            注意这个坐标不是质心系的坐标，参考坐标系的中心是在输入时任意选定的
        plane_basis是两个彼此正交的单位向量
        leading_edge 是单位方向向量
        """
        util.check_keys(pannel, self.PANEL_KEYS)
        util.check_instance(pannel["c_l"], util.KiteFunction)
        util.check_instance(pannel["c_d"], util.KiteFunction)
        util.check_vector(pannel["leading_edge"], 3)
        util.check_vector(pannel["ref_pt"], 3)
        util.check_vector(pannel["plane_basis"], 2)
        for i in range(2):
            util.check_vector(pannel["plane_basis"][i], 3)
        #正规化
        pannel["leading_edge"] = pannel["leading_edge"] / np.linalg.norm(pannel["leading_edge"])
        #reduce模式得到的应该是一个3*2的矩阵q s.t q.T.dot(q) = 0,采用转置保证向量位于行上
        pannel["plane_basis"] = np.linalg.qr(pannel["plane_basis"])[0].transpose()

        self.__panels.append(pannel)
    def add_mass_point(self, mass_pt):
        """
        r是质点在初始参考坐标系中的位矢
            注意这个坐标不是质心系的坐标，参考坐标系的中心是在输入时任意选定的
        """
        util.check_keys(mass_pt, self.MASS_PT_KEYS)
        util.check_vector(mass_pt['r'], 3)

        self.__mass_pts.append(mass_pt)
    def init(self, init_cond):
        """
        设置边界条件
        参数键值参见Dynamic3D.INIT_COND_KEYS
        注意T0传入的时候应该是一个3*3的正交矩阵
        r0是参考坐标系原点（不是质心）的初始位置
        v0是整个风筝质心的初始速度
        L0是初始状态下质心坐标系中的角动量
        """
        util.check_keys(init_cond, self.INIT_COND_KEYS)
        util.check_vector(init_cond["T0"], 3)
        for i in range(3):
            util.check_vector(init_cond["T0"][i], 3)
        util.check_vector(init_cond["r0"], 3)
        util.check_vector(init_cond["v0"], 3)
        util.check_vector(init_cond["L0"], 3)
        util.check_vector(init_cond["v_wind"], 3)

        self.__vc = init_cond["v0"]
        self.__l = init_cond["L0"]
        self.__transformation = init_cond["T0"]
        self.__v_wind = init_cond["v_wind"]
        self.__density = init_cond["density"]

        #辅助变量计算
        self.__mass_total = sum([pt["m"] for pt in self.__mass_pts])
        mass_center = sum([pt["r"] for pt in self.__mass_pts]) / self.__mass_total

        #计算质心在空间中的初始位置
        self.__rc = init_cond["r0"] + mass_center

        #将位矢全部转换为质心坐标系下的位矢
        for panel in self.__panels:
            panel["ref_pt"] = panel["ref_pt"] - mass_center
        for pt in self.__mass_pts:
            pt["r"] = pt["r"] - mass_center

    def __step_read_force(self, pull_forces, attach_pts):
        """
        pull_forces: 各个点上拉力的大小，正方向沿着作用点指向原点
        attach_pts: 各个拉力作用点(绳子附着的点)在质心参考坐标系中的位置
        在该函数的计算中只存在两个坐标系之间的转换：质心坐标系和绝对坐标系
        """
        angular_mass = self.__angular_mass()
        angular_velocity = self.__l / angular_mass

        #compute list of f_l and f_d
        lst_f_l = []
        lst_f_d = []
        for panel in self.__panels:
            v_rotate = np.cross(angular_velocity, \
                self.__transformation.dot(panel["ref_pt"]))
            v_rel = self.__v_wind - (v_rotate + self.__vc)

            if np.linalg.norm(v_rel) == 0:
                lst_f_l.append(np.array([0, 0, 0]))
                continue
            #注意plane_basis是一个2*3的矩阵，dot在矩阵情况下做正常的乘法，需要先转置，转置完成再调用transpose将两个向量放到行上
            angle_tpl = util.vec_plane_angle(self.__v_wind, \
                self.__transformation.dot(panel["plane_basis"].transpose()).transpose())

            f_l_unit = np.cross(self.__transformation.dot(panel["leading_edge"]), v_rel)\
                / np.linalg.norm(f_l_unit)

            f_l = 0.5 * self.__density * panel["area"] *\
                np.linalg.norm(v_rel) ** 2 *\
                panel["c_l"].compute(angle_tpl) *\
                f_l_unit

            f_d = 0.5 * self.__density * panel["area"] *\
                np.linalg.norm(v_rel) * v_rel *\
                panel["c_d"].compute(angle_tpl)

            lst_f_l.append(f_l)
            lst_f_d.append(f_d)

        #计算拉力向量
        pull_forces_vec = []
        for force, attach_pt in zip(pull_forces, attach_pts):
            attach_pt_abs = self.__transformation.dot(attach_pt) + self.__rc
            pull_forces_vec.append(force * attach_pt_abs / np.linalg.norm(attach_pt_abs))

        a_centroid = sum(lst_f_d + lst_f_l + \
            [np.array([0, 0, -self.__mass_total * 9.8]),] + \
            pull_forces_vec) / self.__mass_total

        torque_forces = lst_f_d + lst_f_l + pull_forces_vec
        reference_pts = [panel["ref_pt"] for panel in self.__panels]
        torque_postions = reference_pts + reference_pts + attach_pts
        torque_postions = [self.__transformation.dot(pt) for pt in torque_postions]
        torque = sum([np.cross(position, force) for \
            position, force in zip(torque_postions, torque_forces)])

        self.__l = self.__l + torque * self.__step_interval
        #由于transformation是在列上保留向量的，在调用cross的时候需要使用转置
        self.__transformation = self.__transformation + \
            np.cross(angular_velocity, self.__transformation.transpose()).transpose() \
            * self.__step_interval
        self.__rc = self.__rc + self.__vc * self.__step_interval
        self.__vc = self.__vc + a_centroid * self.__step_interval

    def __angular_mass(self):
        norm_l = np.linalg.norm(self.__l)
        if norm_l == 0:
            l_unit = np.array([0, 0, 1])    #在角动量为0的情况下选取固定方向求解
        else:
            l_unit = self.__l / norm_l
        angular_mass = 0
        for pt in self.__mass_pts:
            r_vertical = np.cross(self.__transformation.dot(pt["r"]), l_unit)
            angular_mass += pt["m"] * np.linalg.norm(r_vertical) ** 2
        return angular_mass
