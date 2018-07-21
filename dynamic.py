#!../bin/python3.6

import util
import numpy as np

class DynamicSolver:
    def __init__(self):
        pass
    def init(self, mass, density, area, c_d, c_l, v_wind, v_vertical_0, r_0, reader ,step_interval = 0.01):
        '系统初始化函数,reader使用的应当是VReaderBase的子类'
        #系统参数
        self.__mass = mass; self.__density = density; self.__area = area
        self.__c_d = c_d; self.__c_l = c_l      #c_l和c_d都是函数对象，需要使用compute接口 
        self.__v_wind = np.array(v_wind)
        self.__v_vertical = np.array(v_vertical_0)
        #边界条件
        self.__r = np.array(r_0)
        self.__step_interval = step_interval
        #求解时钟初始化
        self.__logic_timer = 0
        #设置读取器
        self.__reader = reader
    def step(self):
        '根据设定的求解步长用数值方法求解下一个小时间间隔的v_vertical和位矢r'

        v_parallel = self.get_v_parallel()          #由于get_v_parallel可能具有副作用，所以在整个step的过程中只调用1次
        
        g = 9.8 #重力加速度常数
        cntr_clk_90 = np.array([[0, 1], [-1, 1]])    #逆时针旋转90度的常数矩阵
        clk_90 = np.array([[0, -1], [1, 0]])        #顺时针旋转90度的常数矩阵
        v_kite = self.__r.dot * v_parallel + cntr_clk_90.dot(self.__r) * self.__v_vertical #根据两个分量计算v矢量
        v_rel = self.__v_wind - v_kite
        attack_angle = util.get_angle_2d(clk_90.dot(self.__r), v_rel)   #计算风吹到风筝上的角度

        #下面计算除去拉力的合力f_total
        f_l = 0.5 * self.__density * self.__area * self.__c_l.compute(attack_angle) \
            * cntr_clk_90.dot(v_rel) * np.linalg.norm(v_rel)     #计算提升力
        f_d = 0.5 * self.__density * self.__area * self.__c_d.compute(attack_angle) \
            * np.linalg.norm(v_rel) * v_rel     #计算拉力
        gravity = np.array([0, -self.__mass * g])
        f_total = f_l + f_d + gravity

        #计算v_vertical的导数
        deriv_v_vertical = (np.linalg.det([f_total, self.__r]) - self.__v_vertical * v_parallel) / \
            np.linalg.norm(self.__r)
        
        #更新状态
        self.__r += self.__v_vertical * self.__step_interval
        self.__v_vertical += deriv_v_vertical * self.__step_interval

        #更新逻辑时钟滴答
        self.__logic_timer += self.__step_interval

    def get_state(self):
        return (self.__logic_timer, np.array(self.__r))

    def get_v_parallel(self):
        return self.__reader.read(self.__logic_timer)