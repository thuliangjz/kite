"""
2.5维模拟器的reader类
负责指定读取器的类型（读取当前时刻的加速度还是拉力）
实现自定义读取方法
"""
class ReaderBase:
    TYPE_FORCE = 0
    TYPE_ACCELERATION = 1
    def __init__(self):
        pass
    def read(self, time):
        raise NotImplementedError("implement your own read method")
    def get_type(self):
        raise NotImplementedError("implement your own read method")
