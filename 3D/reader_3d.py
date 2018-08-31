"""
3维模拟器的reader类
负责指定读取器的类型（读取当前时刻的加速度还是拉力）
实现自定义读取方法
"""
class Reader3DBase:
    TYPE_FORCE = 0
    TYPE_ACCELERATION = 1
    def __init__(self):
        self.__attach_pts__ = []
    def read(self, time):
        raise NotImplementedError("implement your own read method")
    def get_type(self):
        raise NotImplementedError("implement your own read method")
    def set_attach_pts(self, lst_pts):
        self.__attach_pts__ = lst_pts
    def get_attach_pts(self):
        return self.__attach_pts__

class Reader3DStableLength(Reader3DBase):
    def read(self, time):
        return [0 for _ in self.__attach_pts__]
    def get_type(self):
        return self.TYPE_ACCELERATION

class Reader3DConstF(Reader3DBase):
    def __init__(self, force):
        self.__force = force
        super(Reader3DConstF, self).__init__()
    def read(self, time):
        return [self.__force for _ in self.__attach_pts__]
    def get_type(self):
        return self.TYPE_FORCE
 