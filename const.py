#!../bin/python3.6

class SampleFunction:
    def __init__(self, sample_lst):
        'sample_lst的格式为[(sample_point, sample_value)], 无sample_point相同,函数内部会对列表按照sample_point排序'
        if len(sample_lst) < 2:
            raise ValueError("sample list too short (at least 2)")
        sample_lst.sort(key= lambda x:x[0])
        self.domain = (sample_lst[0][0], sample_lst[-1][0])
        self.sample_lst = sample_lst
        self.idx_start_last = 0

    def compute(self, val):
        '在查找给定值区间的时候借助了上一次记录下来的结果，更好地利用访问的局部性'
        if not (val >= self.domain[0] and val <= self.domain[1]):
            raise ValueError("value out of domain range")
        if val > self.sample_lst[self.idx_start_last][0]:
            delta = 1
        else:
            delta = -1
        i_start = self.idx_start_last
        while True:
            if val >= self.sample_lst[i_start][0] and \
                val <= self.sample_lst[i_start + 1][0]:
                break
            i_start += delta
        self.idx_start_last = i_start
        start_point, start_val = self.sample_lst[i_start]
        end_point, end_val = self.sample_lst[i_start + 1]
        return start_val + (end_val - start_val) / (end_point - start_point) * (val - start_point)
        
class ConstantSolver:
    def __init__(self):
        pass

    def set_constants(self, mass, area, density, v_wind):
        self.mass = mass
        self.area = area
        self.density = density
        self.v_wind = v_wind

