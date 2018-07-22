#!../bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import dynamic
import reader
import util
import math
import pdb
#定义演示用的c_d和c_l函数对象
class DemoDragFunction(util.KiteFunction):
    def compute(self, val):
        return 0.1
        '''
        return 0.01 + 0.75 * math.sin(val) if val > 0 else \
            0.01 + 0.5 * math.sin(val)
        '''
    def get_range(self):
        return (-math.pi, math.pi)

class DemoLiftFunction(util.KiteFunction):
    def compute(self, val):
        return 8 / math.pi * abs(val) * (abs(val) - math.pi)
    def get_range(self):
        return (-math.pi, math.pi)

#动画更新函数
def update_line(frame, xdata, ydata, solver, line, ax):
    solver.step()
    pos = solver.get_state()[1]
    print(pos)
    lim_x = ax.get_xlim()
    lim_y = ax.get_ylim()
    xmax, ymax = (abs(pos[0]), abs(pos[1]))
    if xmax > lim_x[1]:
        ax.set_xlim(-1.5 * xmax, 1.5 * xmax)
    if ymax > lim_y[1]:
        ax.set_ylim(-1.5 * ymax, 1.5 * ymax)
    xdata.append(pos[0])
    ydata.append(pos[1])
    line.set_data(xdata, ydata)
    return line,

fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(0, 10)
xdata = []
ydata = []

line, = ax.plot(xdata, ydata, 'r-')
solver = dynamic.DynamicSolver()
solver.init(
    mass=0.2, density=1.225, area=1, 
    c_d=DemoDragFunction(),c_l=DemoLiftFunction(),
    v_wind=(-5, 0), v_vertical_0=2, r_0=(-10, 0), 
    reader=reader.VReaderSteady()
    )
line_ani = animation.FuncAnimation(fig, update_line, 1000, fargs=(xdata, ydata, solver, line, ax), interval=50)
plt.show()