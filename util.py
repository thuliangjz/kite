import numpy as np
import math
class KiteFunctionException(Exception):
    pass

class KiteFunction:
    def compute(self, val):
        raise KiteFunctionException("function-specific compute method should be implemented")

    def get_range(self):
        raise KiteFunctionException("function-specific get_range method should be implemented")


class SampleFunction(KiteFunction):
    'sample_lst的格式为[(sample_point, sample_value)], 无sample_point相同,函数内部会对列表按照sample_point排序'
    def __init__(self, sample_lst):
        if len(sample_lst) < 2:
            raise ValueError("sample list too short (at least 2)")
        sample_lst.sort(key= lambda x:x[0])
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


def sgn(x):
    return -1 if x < 0 else 0 if x == 0 else 1

def get_angle_2d(v1, v2):
    '获取二维向量v1转到v2的夹角，范围-pi~pi'
    u_v1 = v1 / np.linalg.norm(v1)
    u_v2 = v2 / np.linalg.norm(v2)
    theta = math.acos(u_v1.dot(u_v2))
    cross = sgn(v1[0] * v2[1] - v1[1] * v2[0])
    if cross != 0:
        theta *= cross
    return theta

def vector_check(v, length):
    try:
        l = len(v)
    except TypeError:
        return False
    return l == length