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
    dynamic.add_panel(...)
    dynamic.add_mass_pt(...)
    dynamic.init(...)
    dynamic.step()
    dynamic.get_state()
    由于numpy和matlab的区别，所有向量，都是1Darray(行向量), 矩阵是通常写法的转置
    Dynamic3D在实现时有3个坐标系：
    构建坐标系
        用户在这个坐标系中构建风筝,用于设置add_panel和add_mass_point中的所有向量
    运动坐标系
        用于描述风筝质心的运动
    质心坐标系
        用于描述风筝姿态的变化
    三个坐标系拥有相同的基矢量:
        z
        |
        |
        |
        o------y
       /
      /
     x
    """
    PANEL_KEYS = ["c_l", "c_d", "plane_basis", "leading_edge", "ref_pt", "area"]
    MASS_PT_KEYS = ["m", "r"]
    INIT_COND_KEYS = ["T0", "r0", "v0", "L0", "v_wind", "density"]
    def __init__(self, reader, step_interval):
        util.check_instance(reader, reader_3d.Reader3DBase)
        self.__reader = reader
        self.__panels = []
        self.__mass_pts = []
        self.__step_interval = step_interval
        self.__states = {}
    def add_panel(self, pannel):
        """
        所有的key参见Dynamic3D.PANNEL_KEYS
        暂定ref_point是f_d和f_l相同的作用点坐标，也是计算v_rel时风筝速度的参考点坐标,
            注意这个坐标不是质心系的坐标，参考坐标系的中心是在输入时任意选定的
        plane_basis是两个彼此正交的单位向量(不适用np.linalg.qr是考虑到方向可能发生变化)
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
        注意T0传入的时候应该是一个3*3的正交矩阵，用于设置初始状态下风筝朝向
        r0是参考坐标系原点（不是质心风筝质心）的初始位置
        v0是整个风筝质心的初始速度,L0是初始状态下质心坐标系中的角动量
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
        mass_center = sum([pt["r"] * pt["m"] for pt in self.__mass_pts]) / self.__mass_total

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

    def get_interval(self):
        return self.__step_interval

    def set_interval(self, interval):
        self.__step_interval = float(interval)

    def get_state(self):
        return self.__states

    def get_rope_length(self):
        transformation = self.__transformation.transpose()
        rope_lengths = []
        for pt in self.__attach_pts:
            rope_lengths.append(np.linalg.norm(transformation.dot(pt) + self.__rc))
        return rope_lengths

    def step(self):
        val = self.__reader.read(self.__timer)
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
            pull_forces_vec.append(-force * attach_pt_abs / np.linalg.norm(attach_pt_abs))

        a_centroid = sum(pull_forces_vec + [f_other, ]) / self.__mass_total
        torque = sum([np.cross(position, force) for position, force in \
            zip(
                [transformation.dot(pt) for pt in attach_pts],
                pull_forces_vec
            )]) + tor_other
        #pylint: disable=c0301, w0105
        '''
        #对求解结果进行反向的验证
        lst_a_rope = []
        deriv_angular_mass = sum([
            2 * pt["m"] * np.cross(transformation.dot(pt["r"]), self.__l / np.linalg.norm(self.__l)).dot(\
            (np.cross(np.cross(angular_velocity, transformation.dot(pt["r"])), self.__l / np.linalg.norm(self.__l)) + \
            np.cross(transformation.dot(pt["r"]), (torque / np.linalg.norm(self.__l) - self.__l * self.__l.dot(torque) / np.linalg.norm(self.__l) ** 3))))\
            for pt in self.__mass_pts
        ])
        deriv_angular_v = (torque * angular_mass - deriv_angular_mass * self.__l) / angular_mass ** 2
        for r_a in attach_pts:
            r_ta = transformation.dot(r_a)
            r_taa = self.__rc + r_ta
            v_taa = np.cross(angular_velocity, r_ta) + self.__vc
            a_taa = np.cross(deriv_angular_v, transformation.dot(r_a)) + np.cross(angular_velocity, np.cross(angular_velocity, self.__transformation).transpose().dot(r_a)) + a_centroid
            a_rope = (v_taa.dot(v_taa) + r_taa.dot(a_taa)) / np.linalg.norm(r_taa) - r_taa.dot(v_taa) ** 2 / np.linalg.norm(r_taa) ** 3
            lst_a_rope.append(a_rope)
        '''
        #pylint: enable=c0301, w0105

        self.__l = self.__l + torque * self.__step_interval
        #numpy的叉乘规则，矩阵和1d array相乘，是矩阵的各个行和该数组进行叉乘得到新的行
        #由于transformation的列向量实际上是__transformation的行向量，直接用cross就可以，这里有矩阵和向量叉乘
        self.__transformation = self.__transformation + \
            np.cross(angular_velocity, self.__transformation) \
            * self.__step_interval
        #transformation归一化(保持方向的Schmidt正交化)
        self.__transformation[0] = self.__transformation[0] / \
            np.linalg.norm(self.__transformation[0])

        self.__transformation[1] = self.__transformation[1] - \
            self.__transformation[1].dot(self.__transformation[0]) * \
            self.__transformation[0]
        self.__transformation[1] = self.__transformation[1] / \
            np.linalg.norm(self.__transformation[1])

        self.__transformation[2] = self.__transformation[2] - \
            self.__transformation[2].dot(self.__transformation[0]) * self.__transformation[0] - \
            self.__transformation[2].dot(self.__transformation[1]) * self.__transformation[1]
        self.__transformation[2] = self.__transformation[2] / \
            np.linalg.norm(self.__transformation[2])

        self.__rc = self.__rc + self.__vc * self.__step_interval
        self.__vc = self.__vc + a_centroid * self.__step_interval

        self.__timer += self.__step_interval

        self.__states = {}
        self.__states["v_angular"] = angular_velocity
        self.__states["rc"] = self.__rc
        self.__states["vc"] = self.__vc
        self.__states["l"] = self.__l
        self.__states["transformation"] = transformation
        self.__states["pull_forces"] = pull_forces



    def __step_read_acceleration(self, accelerations, attach_pts):
        """
        根据传入的加速度和关联点(在质心坐标系下的坐标)更新风筝状态
        """
        angular_mass = self.__angular_mass()
        angular_velocity = self.__l / angular_mass
        transformation = self.__transformation.transpose()
        #deriv_transformation = omiga `cross` transformation
        deriv_transformation = np.cross(angular_velocity, self.__transformation).transpose()

        #计算角动量和质心加速度关于拉力的仿射变换矩阵(op_a_c和op_deri_l)
        f_other, tor_other = self.__get_f_tor_no_pull()
        lst_f = []
        lst_tor = []
        for pt in attach_pts:
            r_rel = transformation.dot(pt)
            r_abs = self.__rc + r_rel
            r_abs_unit = r_abs / np.linalg.norm(r_abs)
            lst_f.append(-r_abs_unit)
            lst_tor.append(np.cross(r_rel, -r_abs_unit))
        dimension = len(accelerations)
        op_deri_l = (util.affinization(np.array(lst_tor).transpose(), False, dimension) + \
            util.affinization(tor_other, True, dimension))
        op_a_c = (util.affinization(np.array(lst_f).transpose(), False, dimension) + \
            util.affinization(f_other, True, dimension)) / self.__mass_total
        #计算角动量微分关于拉力的仿射变换矩阵
        norm_l = np.linalg.norm(self.__l)
        if norm_l != 0:
            l_unit = self.__l / norm_l
            #l' / |l| - l * (l `dot` l') / |l|^3，这部分会被在下面反复使用，避免重复计算
            op_deri_ang_v_0 = op_deri_l / norm_l - \
                self.__l[None].transpose().dot(self.__l[None].dot(op_deri_l)) / norm_l ** 3
            lst_op_deri_ang_v_1 = []
            for pt in self.__mass_pts:
                #循环计算：
                #2 * m * ((T * r_m[i]) `cross` l_unit) `dot`
                #(
                #   T' * r_m[i]) `cross` l_unit + (T * r_m[i]) `cross`
                #   (
                #       l' / |l| - l * (l `dot` l') / |l|^3
                #   )
                #)
                lst_op_deri_ang_v_1.append(
                    2 * pt["m"] * \
                    np.cross(transformation.dot(pt["r"]), l_unit)[None].dot(
                        util.affinization(\
                            np.cross(deriv_transformation.dot(pt["r"]), l_unit),\
                            True, dimension) + \
                        np.cross(\
                            transformation.dot(pt["r"]), \
                            op_deri_ang_v_0.transpose()\
                        ).transpose()\
                    )
                )
            #求和这一步实际上是计算出了转动惯量导数的仿射矩阵
            op_deri_ang_mass = sum(lst_op_deri_ang_v_1)
            op_deri_ang_v = op_deri_l / angular_mass - \
                angular_mass ** -2 * self.__l[None].transpose().dot(op_deri_ang_mass)
        else:
            #注意这种情况下转动惯量不是正确的，但是这样处理的好处是使得方程转变为线性方程，而且这种情况几乎不出现
            op_deri_ang_v = op_deri_l / angular_mass

        #生成方程组
        rows = []
        target = []
        for a_attach_pt, r_attach_rel in zip(accelerations, attach_pts):
            v_attach_pt = np.cross(angular_velocity, transformation.dot(r_attach_rel)) + self.__vc
            r_attach_abs = transformation.dot(r_attach_rel) + self.__rc
            op_a_attach = \
                np.cross(\
                    op_deri_ang_v.transpose(), \
                    transformation.dot(r_attach_rel)\
                ).transpose() + \
                util.affinization(\
                    np.cross(\
                        angular_velocity, \
                        deriv_transformation.dot(r_attach_rel)\
                    ), True, dimension\
                ) + op_a_c
            norm_r = np.linalg.norm(r_attach_abs)
            op_a_rope = \
                util.affinization(
                    [(v_attach_pt / norm_r - \
                        r_attach_abs * r_attach_abs.dot(v_attach_pt) \
                        / norm_r ** 3).dot(v_attach_pt), ], \
                    True, dimension) + \
                (r_attach_abs / norm_r)[None].dot(op_a_attach)
            #转化成1D array
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

            #注意plane_basis是一个2*3的矩阵，dot在矩阵情况下做正常的乘法，需要先转置，转置完成再调用transpose将两个向量放到行上
            angle_tpl = util.vec_plane_angle(v_rel, \
                transformation.dot(panel["plane_basis"].transpose()).transpose())

            f_l_dir = np.cross(transformation.dot(panel["leading_edge"]), v_rel)
            if np.array_equal(f_l_dir, np.array([0, 0, 0])):
                #应对v_rel == 0或者v_rel与该块panel当时的leading_edge平行
                f_l = np.array([0, 0, 0])
            else:
                f_l = 0.5 * self.__density * panel["area"] *\
                    np.linalg.norm(v_rel) ** 2 *\
                    panel["c_l"].compute(angle_tpl) *\
                    f_l_dir / np.linalg.norm(f_l_dir)

            f_d = 0.5 * self.__density * panel["area"] *\
                np.linalg.norm(v_rel) * v_rel *\
                panel["c_d"].compute(angle_tpl)

            lst_f_l.append(f_l)
            lst_f_d.append(f_d)

        lst_f_tor_other = lst_f_l + lst_f_d
        f_other = sum(lst_f_tor_other + [np.array([0, 0, -self.__mass_total * 9.8]),])
        torque_positions = [transformation.dot(panel["ref_pt"]) \
            for panel in self.__panels + self.__panels]
        torque_other = sum([np.cross(position, force) for \
                    position, force in zip(torque_positions, lst_f_tor_other)])
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
