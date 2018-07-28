#!../bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys, os
sys.path.append(os.path.abspath("."))
import util
import dynamic
import reader
import math
import pdb
import const
#定义演示用的c_d和c_l函数对象
class DemoDragFunction(util.KiteFunction):
    def compute(self, val):
        return 0.01 + 0.75 * math.sin(val) if val > 0 else \
            0.01 + 0.5 * math.sin(val)
    def get_range(self):
        return (-math.pi, math.pi)

class DemoLiftFunction(util.KiteFunction):
    def compute(self, val):
        return 8 / math.pi * abs(val) * (math.pi - abs(val))
    def get_range(self):
        return (-math.pi, math.pi)

#动画更新函数
def update_line(frame, xdata, ydata, solver, line, ax):
    solver.step()
    state = solver.get_state()
    pos = state[1]
    print("pos:", pos)
    print("v_vertical", state[2])
    print("+"*32)
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
alpha = 0
dic_para = {
    'mass':0.05,
    'density':1.225,
    'area':1,
    'c_d':DemoDragFunction(),
    'c_l':DemoLiftFunction(),
    'v_wind':(-5, 0),
}
solver.init(
    mass=dic_para['mass'], density=dic_para['density'], area=dic_para['area'], 
    c_d=dic_para['c_d'],c_l=dic_para['c_l'],
    v_wind=dic_para['v_wind'], v_vertical_0=2, r_0=(-10* math.cos(alpha), 10 * math.sin(alpha)), 
    reader=reader.VReaderSteady()
    )

const_solver = const.ConstantSolver()
const_solver.set_constants(mass=dic_para['mass'], 
    area=dic_para['area'], 
    density=dic_para['density'], 
    v_wind=abs(dic_para['v_wind'][0]))
const_solver.set_c_d(dic_para['c_d'])
const_solver.set_c_l(dic_para['c_l'])
res = const_solver.solve()
print(math.tan(res))
line_ani = animation.FuncAnimation(fig, update_line, 1000, fargs=(xdata, ydata, solver, line, ax), interval=50)
plt.show()