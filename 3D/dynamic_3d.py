"""
实现3维风筝模拟求解器
"""
import sys
import os
import numpy as np
import reader_3d
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
#pylint: disable=wrong-import-position
import util
#pylint: enable=wrong-import-position

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
    由于numpy和matlab的区别，所有向量，都是1Darray(行向量), 矩阵是通常写法的转置
    """
    PANEL_KEYS = ["c_l", "c_d", "plane_basis", "leading_edge", "ref_pt", "area"]
    MASS_PT_KEYS = ["m", "r"]
    INIT_COND_KEYS = ["T0", "r0", "v0", "L0", "v_wind", "density"]
    def __init__(self, reader, step_interval):
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
        self.__timer = 0

        #辅助变量计算
        self.__mass_total = sum([pt["m"] for pt in self.__mass_pts])
        mass_center = sum([pt["r"] for pt in self.__mass_pts]) / self.__mass_total

        #计算质心在空间中的初始位置
        self.__rc = init_cond["r0"] + mass_center
        self.__attach_pts = []

        #将位矢全部转换为质心坐标系下的位矢
        for panel in self.__panels:
            panel["ref_pt"] = panel["ref_pt"] - mass_center
        for pt in self.__mass_pts:
            pt["r"] = pt["r"] - mass_center
        for pt in self.__reader.get_attach_pts():
            self.__attach_pts.append(pt - mass_center)

    def __step(self):
        val = self.__reader.read()
        type_value = self.__reader.get_type()
        if type_value == reader_3d.Reader3DBase.TYPE_FORCE:
            self.__step_read_force(val, self.__attach_pts)
        else:
            self.__step_read_acceleration(val, self.__attach_pts)

    def __step_read_force(self, pull_forces, attach_pts, **args):
        """
        pull_forces: 各个点上拉力的大小，正方向沿着作用点指向原点
        attach_pts: 各个拉力作用点(绳子附着的点)在质心参考坐标系中的位置
        可以在args中通过other键来传入除了拉力之外的合力和合力矩
        在该函数的计算中只存在两个坐标系之间的转换：质心坐标系和绝对坐标系
        """
        angular_mass = self.__angular_mass()
        angular_velocity = self.__l / angular_mass
        transformation = self.__transformation.transpose()

        f_other, tor_other = args.get("other", self.__get_f_tor_no_pull())
        #计算拉力向量
        pull_forces_vec = []
        for force, attach_pt in zip(pull_forces, attach_pts):
            attach_pt_abs = transformation.dot(attach_pt) + self.__rc
            pull_forces_vec.append(force * attach_pt_abs / np.linalg.norm(attach_pt_abs))

        a_centroid = sum(pull_forces_vec.append(f_other)) / self.__mass_total

        torque_postions = [transformation.dot(pt) for pt in attach_pts]
        torque = sum([np.cross(position, force) for \
            position, force in zip(torque_postions, pull_forces_vec)]) + tor_other

        self.__l = self.__l + torque * self.__step_interval
        #由于transformation的列向量实际上是__transformation的行向量，直接用cross就可以，这里有矩阵和向量叉乘
        self.__transformation = self.__transformation + \
            np.cross(angular_velocity, self.__transformation) \
            * self.__step_interval
        self.__rc = self.__rc + self.__vc * self.__step_interval
        self.__vc = self.__vc + a_centroid * self.__step_interval

        self.__timer += self.__step_interval

    def __step_read_acceleration(self, accelerations, attach_pts):
        """
        根据传入的加速度和关联点(在质心坐标系下的坐标)更新风筝状态
        """
        angular_mass = self.__angular_mass()
        angular_velocity = self.__l / angular_mass
        transformation = self.__transformation.transpose()
        #有矩阵和向量叉乘，deriv_transformation希望得到正常的矩阵
        deriv_transformation = np.cross(angular_velocity, self.__transformation).transpose()
        f_other, tor_other = self.__get_f_tor_no_pull()
        lst_f = []
        lst_tor = []
        for pt in attach_pts:
            r_rel = transformation.dot(pt)
            r_abs = self.__rc + r_rel
            r_abs_unit = r_abs / np.linalg.norm(r_abs)
            lst_f.append(r_abs_unit)
            lst_tor.append(np.cross(r_rel, r_abs_unit))
        matrix_f = np.array(lst_f + [f_other,]).transpose()
        matrix_tor = np.array(lst_tor + [tor_other,]).transpose()
        dimension = len(accelerations)
        #计算角动量和质心加速度关于拉力的仿射变换矩阵
        op_deri_l = util.affinization(matrix_tor, False, dimension) + \
            util.affinization(tor_other, True, dimension)
        op_a_c = util.affinization(matrix_f, False, dimension) + \
            util.affinization(f_other, True, dimension)
        #计算角动量微分关于拉力的仿射变换矩阵
        norm_l = np.linalg.norm(self.__l)
        if norm_l != 0:
            l_unit = self.__l / norm_l
            #l' / |l| - l * (l `dot` l') / |l|^3，这部分会被在下面反复使用，避免重复计算
            op_deri_ang_v_0 = op_deri_l / norm_l - \
                self.__l[None].transpose().dot(self.__l[None].dot(op_deri_l))
            lst_op_deri_ang_v_1 = []
            for pt in self.__mass_pts:
                #(T' * r_m[i]) `cross` l_unit
                op_deri_ang_v_1 = util.affinization(\
                    np.cross(deriv_transformation.dot(pt["r"]), l_unit), True, dimension)
                #tmp = tmp + (T * r_m[i]) `cross` tmp0
                #注意有矩阵和向量的叉乘
                op_deri_ang_v_1 = op_deri_ang_v_1 + \
                    np.cross(transformation.dot(pt["r"]), op_deri_ang_v_0.transpose())
                #tmp = 2 * m * ((T * r_m[i]) `cross` l_unit) `dot` tmp
                op_deri_ang_v_1 = 2 * pt["m"] * \
                    np.cross(transformation.dot(pt["r"]), l_unit)[None].dot(op_deri_ang_v_1)
                lst_op_deri_ang_v_1.append(op_deri_ang_v_1)
            op_deri_ang_v = sum(lst_op_deri_ang_v_1)
            op_deri_ang_v = op_deri_l / angular_mass - \
                angular_mass ** -2 * self.__l[None].transpose().dot(op_deri_ang_v)
        else:
            #注意这种情况下转动惯量的方向不是正确的，但是这样处理的好处是使得方程转变为线性方程，而且这种情况几乎不出现
            op_deri_ang_v = op_deri_l / angular_mass

        #生成方程组
        rows = []
        target = []
        for a_attach_pt, r_attach_rel in zip(accelerations, attach_pts):
            v_attach_pt = np.cross(angular_velocity, transformation.dot(r_attach_rel)) + self.__vc
            r_attach_abs = transformation.dot(r_attach_rel) + self.__rc
            op_a_attach = np.cross(\
                            op_deri_ang_v.transpose(), \
                            transformation.dot(r_attach_rel))\
                        .transpose() + \
                        util.affinization(\
                            np.cross(angular_velocity, \
                                deriv_transformation.dot(r_attach_rel)), \
                            True, dimension) + \
                        op_a_c
            norm_r = np.linalg.norm(r_attach_abs)
            op_a_rope = \
                util.affinization(
                    (v_attach_pt / norm_r - \
                        r_attach_abs * r_attach_abs.dot(v_attach_pt) \
                        / norm_r ** 3).dot(v_attach_pt), \
                    True, dimension) + \
                (r_attach_abs / norm_r)[None].dot(op_a_attach)
            op_a_rope = op_a_rope[0]
            rows.append(op_a_rope[:dimension])
            target.append(a_attach_pt - op_a_rope[dimension])
        #计算拉力向量，对矩阵奇异情形做处理
        matrix_t2a = np.array(rows)
        try:
            inv = np.linalg.inv(matrix_t2a)
        except np.linalg.linalg.LinAlgError:
            inv = np.linalg.pinv(matrix_t2a)
        pull_forces = inv.dot(np.array(target))
        self.__step_read_force(pull_forces, attach_pts, other=(f_other, tor_other))

    def __get_f_tor_no_pull(self):
        angular_mass = self.__angular_mass()
        angular_velocity = self.__l / angular_mass
        transformation = self.__transformation.transpose()
        #compute list of f_l and f_d
        lst_f_l = []
        lst_f_d = []
        for panel in self.__panels:
            v_rotate = np.cross(angular_velocity, \
                transformation.dot(panel["ref_pt"]))
            v_rel = self.__v_wind - (v_rotate + self.__vc)

            if np.linalg.norm(v_rel) == 0:
                lst_f_l.append(np.array([0, 0, 0]))
                continue
            #注意plane_basis是一个2*3的矩阵，dot在矩阵情况下做正常的乘法，需要先转置，转置完成再调用transpose将两个向量放到行上
            angle_tpl = util.vec_plane_angle(self.__v_wind, \
                transformation.dot(panel["plane_basis"].transpose()).transpose())

            f_l_unit = np.cross(transformation.dot(panel["leading_edge"]), v_rel)\
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
        lst_f_tor_other = lst_f_l + lst_f_d
        f_other = sum(lst_f_tor_other + [np.array([0, 0, -self.__mass_total * 9.8]),])
        reference_pts = [panel["ref_pt"] for panel in self.__panels]
        torque_postions = reference_pts + reference_pts
        torque_postions = [transformation.dot(pt) for pt in torque_postions]
        torque_other = sum([np.cross(position, force) for \
                    position, force in zip(torque_postions, lst_f_tor_other)])
        return (f_other, torque_other)


    def __angular_mass(self):
        transformation = self.__transformation.transpose()
        norm_l = np.linalg.norm(self.__l)
        if norm_l == 0:
            l_unit = np.array([0, 0, 1])    #在角动量为0的情况下选取固定方向求解
        else:
            l_unit = self.__l / norm_l
        angular_mass = 0
        for pt in self.__mass_pts:
            r_vertical = np.cross(transformation.dot(pt["r"]), l_unit)
            angular_mass += pt["m"] * np.linalg.norm(r_vertical) ** 2
        return angular_mass
