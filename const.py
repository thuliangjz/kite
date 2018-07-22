#!../bin/python3.6

import util
import math
import pdb


class ConstantSolverError(Exception):
    ERR_EXPLORE_FAILED = "exploration failed"
    ERR_DELTA_TOO_SMALL = "delta too small"
    pass


class ConstantSolver:
    '定常问题二分求解器'
    def __init__(self):
        pass

    def set_constants(self, mass, area, density, v_wind):
        self.__mass = mass
        self.__area = area
        self.__density = density
        self.__v_wind = v_wind
        self.__g = 9.8
    
    def set_c_l(self, c_l):
        '角度都通过弧度给出'
        self.__c_l = c_l

    def set_c_d(self, c_d):
        '角度都通过弧度给出'
        self.__c_d = c_d

    def __get_valid_range(self):
        '计算有效的求解phi的范围'
        r1s, r1e = self.__c_d.get_range()
        r2s, r2e = self.__c_l.get_range()
        s = max(r1s, r2s); e = min(r1e, r2e)
        return (math.pi / 2 - e, math.pi / 2 - s)

    def compute(self, phi):
        '计算要求解零点的函数在给定的phi处的值'
        c_d = self.__c_d.compute(math.pi / 2 - phi)
        return self.__c_l.compute(math.pi / 2 - phi) / c_d - \
            self.__g * self.__mass / \
            (0.5 * self.__density * self.__area * self.__v_wind ** 2 * c_d) \
            - math.tan(phi)

    def solve(self, explore_step_ratio = 1/200, delta = 0.01):
        '外部调用的时候应当指定首次探索时的步长，以及最终数值解可以接受的精度'
        valid_start, valid_end = self.__get_valid_range()
        res_start = self.compute(valid_start)
        if abs(res_start) < delta:
            return valid_start
        sgn_start = util.sgn(res_start)

        zero_found = False
        explore_step = explore_step_ratio * (valid_end - valid_start)
        sample_point = valid_start + explore_step
        while sample_point < valid_end:
            sgn_tmp = util.sgn(self.compute(sample_point))
            if sgn_tmp == 0:
                return sample_point
            if sgn_tmp != sgn_start:
                zero_found = True
                break
            sample_point += explore_step
        if not zero_found:
            raise ConstantSolverError(ConstantSolverError.ERR_EXPLORE_FAILED)
        
        #二分法求解
        start = valid_start
        end = sample_point; sgn_end = sgn_tmp
        while start != end:
            mid = (start + end) / 2
            res_mid = self.compute(mid)
            if abs(res_mid) < delta:
                return mid
            sgn_mid = util.sgn(res_mid)
            if sgn_mid == sgn_end:
                end = mid; sgn_end = sgn_mid
            else:
                start = mid
        raise ConstantSolverError(ConstantSolverError.ERR_DELTA_TOO_SMALL)
