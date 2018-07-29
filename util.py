"""
utility functions and classes for kite project
"""
import math
import numpy as np
class KiteFunctionException(Exception):
    pass

class KiteFunction:
    def compute(self, val):
        pass
        #raise KiteFunctionException("function-specific compute method should be implemented")

    def get_range(self):
        pass
        #raise KiteFunctionException("function-specific get_range method should be implemented")


class SampleFunction(KiteFunction):
    'sample_lst的格式为[(sample_point, sample_value)], 无sample_point相同,函数内部会对列表按照sample_point排序'
    def __init__(self, sample_lst):
        if len(sample_lst) < 2:
            raise ValueError("sample list too short (at least 2)")
        sample_lst.sort(key=lambda x: x[0])
        self.__domain = (sample_lst[0][0], sample_lst[-1][0])
        self.__sample_lst = sample_lst
        self.__idx_start_last = 0

    def compute(self, val):
        '在查找给定值区间的时候借助了上一次记录下来的结果，更好地利用访问的局部性'
        if not (val >= self.__domain[0] and val <= self.__domain[1]):
            raise ValueError("value out of domain range")
        if val > self.__sample_lst[self.__idx_start_last][0]:
            delta = 1
        else:
            delta = -1
        i_start = self.__idx_start_last
        while True:
            if val >= self.__sample_lst[i_start][0] and \
                val <= self.__sample_lst[i_start + 1][0]:
                break
            i_start += delta
        self.__idx_start_last = i_start
        start_point, start_val = self.__sample_lst[i_start]
        end_point, end_val = self.__sample_lst[i_start + 1]
        return start_val + (end_val - start_val) / (end_point - start_point) * (val - start_point)

    def get_range(self):
        return (self.__domain[0], self.__domain[1])


def sgn(x_val):
    return -1 if x_val < 0 else 0 if x_val == 0 else 1

def get_angle_2d(v_1, v_2):
    '获取二维向量v_1转到v_2的夹角，范围-pi~pi'
    norm_v1 = np.linalg.norm(v_1)
    norm_v2 = np.linalg.norm(v_2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    u_v1 = v_1 / norm_v1
    u_v2 = v_2 / norm_v2
    theta = math.acos(u_v1.dot(u_v2))
    cross = sgn(v_1[0] * v_2[1] - v_1[1] * v_2[0])
    if cross != 0:
        theta *= cross
    return theta

def check_vector(vec, length):
    try:
        len_v = len(vec)
    except TypeError:
        raise ValueError("vector checked have no length")
    if len_v != length:
        raise ValueError("vector should be of length %d"%(length))

def check_keys(dic, keys):
    if not isinstance(dic, dict):
        raise ValueError("object checked not key")
    for key in keys:
        if not key in dic.keys():
            raise KeyError("%s not in dictionary"%(key))

def check_instance(obj, cls):
    if not isinstance(obj, cls):
        raise ValueError("object not of type %s"%(cls.__name__))

def vec_plane_angle(vec, plane):
    """
    v: 3维向量
    plane：两个正交单位3维向量
    返回:(phi, theta)
    其中phi是vec在plane上投影的幅角
    theta是vec在plane法向量与在plane上投影张成平面上的幅角
    """
    proj_xy = (vec.dot(plane[0]), vec.dot(plane[1]))
    phi = get_angle_2d(np.array(proj_xy), np.array([1, 0]))
    proj_z = vec.dot(np.cross(plane[0], plane[1]))
    theta = get_angle_2d(
        np.array([proj_z, np.linalg.norm(proj_xy)]),
        np.array([0, 1])
    )
    return (phi, theta)
