"""
3D风筝Demo
"""
import sys
import os
import math
import re
import pdb
from collections import deque
import numpy as np
import pygame
import OpenGL.GL as gl
import OpenGL.GLU as glu

import reader_3d
import dynamic_3d
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
))
#pylint: disable=wrong-import-position
import util
#pylint: enable=wrong-import-position

class CL3DTest(util.KiteFunction):
    def __init__(self, m_cl):
        self.__max_cl = m_cl
    def compute(self, val):
        #最大0.5,反向之后仍然向上,随着theta接近pi/2而减小
        return (math.pi / 2 - val[1])  / math.pi * 2 * self.__max_cl

class CD3DConst(util.KiteFunction):
    def __init__(self, c):
        self.__c = c
    def compute(self, val):
        return self.__c

class Reader3DDynamicLength(reader_3d.Reader3DBase):
    """
    根据输入控制读取加速度的读取器
    内部保留了一个象征加速度的tuple的队列
    tpl:(开始时间,结束时间,加速度大小)
    每按一次对应添加一个类型的加速度事件
    """
    def __init__(self, a_pull, time_pull, ratio_pull_rel):
        self.__acceleratoin_events = []
        self.__a_pull = float(a_pull)
        self.__time_pull = float(time_pull)
        self.__a_rel = -a_pull / float(ratio_pull_rel)
        self.__time_rel = float(time_pull) * ratio_pull_rel
        #元组含义:起始时间到当前时间距离,长度,加速度
        self.__events = {
            "pull": [(0, time_pull / 2, a_pull), (time_pull / 2, time_pull / 2, -a_pull)],
            "rel": [(0, self.__time_rel / 2, self.__a_rel), \
                (self.__time_rel / 2, self.__time_rel / 2, -self.__a_rel)]
        }
        self.__time = 0
        self.dic_key_event = {
            pygame.K_1: (0, "pull"),
            pygame.K_2: (0, "rel"),
            pygame.K_3: (1, "pull"),
            pygame.K_4: (1, "rel")
        }
        super(Reader3DDynamicLength, self).__init__()
    def get_key_event_map(self):
        return self.dic_key_event
    def new_a_event(self, tp_event):
        """
        有按键按下时加入加速度事件
        """
        idx, key = tp_event
        events_new = [(self.__time + tp[0], self.__time + tp[0] + tp[1], tp[2])
                      for tp in self.__events[key]]
        queue = list(self.__acceleratoin_events[idx])
        i = 0
        for n_e in events_new:
            while i < len(queue):
                overlap = (max(n_e[0], queue[i][0]), min(n_e[1], queue[i][1]))
                if overlap[0] < overlap[1]:
                    queue.insert(i + 1, (overlap[0], overlap[1], n_e[2] + queue[i][2]))
                    queue[i] = (queue[i][0], overlap[0], queue[i][2])
                    n_e = (overlap[1], n_e[1], n_e[2])
                    i += 1
                if n_e[0] >= n_e[1]:
                    break
                i += 1
            if n_e[0] < n_e[1]:
                queue.append(tuple(n_e))
                i += 1
        self.__acceleratoin_events[idx] = deque(queue)


    def read(self, time):
        self.__time = time
        lst_result = []
        for i in range(len(self.__acceleratoin_events)):
            while self.__acceleratoin_events[i] and time >= self.__acceleratoin_events[i][0][1]:
                self.__acceleratoin_events[i].popleft()
            if self.__acceleratoin_events[i] and time >= self.__acceleratoin_events[i][0][0]:
                #位于起始时间之后才读取该事件的加速度
                lst_result.append(-self.__acceleratoin_events[i][0][2])
            else:
                lst_result.append(0)
        return lst_result


    def get_type(self):
        return self.TYPE_ACCELERATION
    def set_attach_pts(self, lst_pts):
        super(Reader3DDynamicLength, self).set_attach_pts(lst_pts)
        self.__acceleratoin_events = [deque() for _ in self.__attach_pts__]
    def reset(self):
        for queue in self.__acceleratoin_events:
            queue.clear()
        self.__time = 0

#上下侧翼参数
LEN_SIDE_WING = 0.25
THETA_SIDE_WING = math.pi / 4
BASIS_UP = np.array([
    (1, 0, 0),
    (0, -math.cos(THETA_SIDE_WING), math.sin(THETA_SIDE_WING)),
    ])
BASIS_DOWN = np.array([
    (1, 0, 0),
    (0, math.cos(THETA_SIDE_WING), math.sin(THETA_SIDE_WING)),
    ])
REF_UP = np.array([0, 0.25, 0.25]) + 0.5 * LEN_SIDE_WING * BASIS_UP[1]
REF_DOWN = np.array([0, REF_UP[1], -REF_UP[2]])
C_D_LR = 0.3
C_D_UD = 0.5
C_L_LR = 0.7
C_L_UD = 0.5
#leading_edge始终是plane_basis的第一个
SIMPLE_KITE = {
    "panels":[
        {
            "c_l": CL3DTest(C_L_LR),
            "c_d": CD3DConst(C_D_LR),
            "plane_basis": np.array([
                (3 ** 0.5 / 2, 0.5, 0),
                (0, 0, 1),
            ]),
            "leading_edge": np.array([3 ** 0.5 / 2, 0.5, 0]),
            "ref_pt": np.array([-0.5 - 3 ** 0.5 / 8, 1 / 8, 0]),
            "area": 0.25,
            "width": 0.5,
            "length": 0.5,
            "color": (1, 0, 0)
        },
        {
            "c_l": CL3DTest(C_L_UD),
            "c_d": CD3DConst(C_D_UD),
            "plane_basis": np.array([
                (1, 0, 0),
                (0, 0, 1),
            ]),
            "leading_edge": np.array([1, 0, 0]),
            "ref_pt": np.array([0, 1 / 4, 0]),
            "area": 0.5,
            "width": 0.5,
            "length": 1.0,
            "color": (0, 1, 0)
        },
        {
            "c_l": CL3DTest(C_L_LR),
            "c_d": CD3DConst(C_D_LR),
            "plane_basis": np.array([
                (3 ** 0.5 / 2, -0.5, 0),
                (0, 0, 1),
            ]),
            "leading_edge": np.array([3 ** 0.5 / 2, -0.5, 0]),
            "ref_pt": np.array([0.5 + 3 ** 0.5 / 8, 1 / 8, 0]),
            "area": 0.25,
            "width": 0.5,
            "length": 0.5,
            "color": (0, 0, 1)
        },
        {
            "c_l": CL3DTest(C_L_UD),
            "c_d": CD3DConst(C_D_UD),
            "plane_basis": BASIS_UP,
            "leading_edge": (1, 0, 0),
            "ref_pt": REF_UP,
            "area": LEN_SIDE_WING,
            "width": LEN_SIDE_WING,
            "length": 1,
            "color": (0, 1, 1)
        },
        {
            "c_l": CL3DTest(C_L_UD),
            "c_d": CD3DConst(C_D_UD),
            "plane_basis": BASIS_DOWN,
            "leading_edge": (1, 0, 0),
            "ref_pt": REF_DOWN,
            "area": LEN_SIDE_WING,
            "width": LEN_SIDE_WING,
            "length": 1,
            "color": (0, 1, 1)
        },
    ],
    "mass_pts":[
        #总重0.2kg
        {
            "m": 0.05,
            "r": np.array([-0.5 - 3 ** 0.5 / 8, 1 / 8, 0]),
        },
        {
            "m": 0.1,
            "r": np.array([0, 1 / 4, 0]),
        },
        {
            "m": 0.05,
            "r": np.array([0.5 + 3 ** 0.5 / 8, 1 / 8, 0])
        },
    ],
    #两根绳子，连接点不直接在风筝上
    "attach_pts":[
        np.array([-0.5, -1 / 4, 0]),
        np.array([0.5, -1 / 4, 0]),
    ],
}

def normalize_kite(kite_data):
    """
    将构建风筝模型的坐标系转换为风筝的质心坐标系
    """
    r_center = sum([pt["m"] * pt["r"] for pt in kite_data["mass_pts"]])
    for panel in kite_data["panels"]:
        panel["ref_pt"] = panel["ref_pt"] - r_center
    for pt in kite_data["mass_pts"]:
        pt["r"] = pt["r"] - r_center
    kite_data["attach_pts"] = [pt - r_center for pt in kite_data["attach_pts"]]

def set_quad_vert(panel):
    """
    为panel添加各个顶点和边在质心坐标系中的初始位置
    """
    basis = panel["plane_basis"]
    length = panel["length"]
    width = panel["width"]
    ref_pt = panel["ref_pt"]
    panel["quad_verts"] = [
        ref_pt - length / 2 * basis[0] - width / 2 * basis[1],
        ref_pt + length / 2 * basis[0] - width / 2 * basis[1],
        ref_pt + length / 2 * basis[0] + width / 2 * basis[1],
        ref_pt - length / 2 * basis[0] + width / 2 * basis[1],
    ]
    quad_verts_2 = panel["quad_verts"].copy()
    quad_verts_2.pop(0)
    quad_verts_2.append(panel["quad_verts"][0])
    panel["quad_edges"] = list(zip(panel["quad_verts"], quad_verts_2))

def lst_matrix(mat):
    """
    将矩阵转换成为列表
    """
    lst = list(mat)
    lst = [list(v) for v in lst]
    res = []
    for vec in lst:
        res.extend(vec)
    return res

def draw_kite(kite_data, kite_status, view_parameter):
    """
    依据风筝数据，状态，以及给定的观察角度绘制风筝
    kite_status: r_c, transformation
    kite_data: 参考SIMPLE_KITE
    view_parameter: look_at{eye, center, up}(都是np.array), perspective:用于构造投影矩阵的tuple
    注意look_at的各个坐标是模拟器中的世界坐标
    """
    #风筝模拟器的世界坐标系和opengl的世界坐标系的转换矩阵
    world_transformation = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])
    #清空画面&变换阵复原
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    #加载投影矩阵
    gl.glMatrixMode(gl.GL_PROJECTION)
    glu.gluPerspective(*view_parameter["perspective"])

    #加载观察阵
    gl.glMatrixMode(gl.GL_MODELVIEW)
    arg_eye = world_transformation.dot(view_parameter["look_at"]["eye"])
    arg_center = world_transformation.dot(view_parameter["look_at"]["center"])
    arg_up = world_transformation.dot(view_parameter["look_at"]["up"])
    glu.gluLookAt(*(tuple(arg_eye) + tuple(arg_center) + tuple(arg_up)))

   #在世界坐标系中绘制风筝，注意模拟器世界坐标系和风筝的世界坐标系不一致
    gl.glBegin(gl.GL_QUADS)
    for panel in kite_data["panels"]:
        gl.glColor3fv(panel["color"])
        for pt in panel["quad_verts"]:
            gl.glVertex3fv(world_transformation.dot(
                kite_status["rc"] + kite_status["transformation"].dot(pt)
            ))
    gl.glEnd()

    gl.glLineWidth(1)
    gl.glBegin(gl.GL_LINES)
    gl.glColor3fv((1, 1, 1))
    for panel in kite_data["panels"]:
        for edge in panel["quad_edges"]:
            for pt in edge:
                gl.glVertex3fv(world_transformation.dot(
                    kite_status["rc"] + kite_status["transformation"].dot(pt)
                ))
    gl.glEnd()

    #绘制风筝线
    gl.glLineWidth(3)
    gl.glBegin(gl.GL_LINES)
    for pt in kite_data["attach_pts"]:
        gl.glVertex3fv(world_transformation.dot(
            kite_status["rc"] + kite_status["transformation"].dot(pt)
        ))
        gl.glVertex3fv((0, 0, 0))
    gl.glEnd()


#pygame事件响应函数, event是通过pygame.event.poll()获得的

def animation_input_function(event):
    """
    空格播放暂停，方向左键和右键单步前进和后退,
    按U回到初始状态
    ESC回到上一级
    """
    handler = GLOBAL_OBJECTS["ani_input_handler"]
    solver = GLOBAL_OBJECTS["solver"]
    reader = solver.get_reader()
    if event.type == pygame.KEYDOWN and event.key in reader.get_key_event_map():
        reader.new_a_event(reader.get_key_event_map()[event.key])
    if handler.is_playing:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            handler.is_playing = False
        else:
            GLOBAL_OBJECTS["state_list"].append(solver.get_state())
            solver.step()
            state = solver.get_state()
            GLOBAL_OBJECTS["current_state"] = state
            interval = solver.get_interval()
            handler.time_played += interval
            if handler.time_played > handler.max_play_time and handler.max_play_time > 0:
                handler.is_playing = False
                handler.time_played = 0
    elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_SPACE:
            handler.is_playing = True
        elif event.key == pygame.K_RIGHT:
            GLOBAL_OBJECTS["state_list"].append(solver.get_state())
            solver.step()
            state = solver.get_state()
            GLOBAL_OBJECTS["current_state"] = state
        elif event.key == pygame.K_LEFT:
            #检查列表为空的一种手段
            if GLOBAL_OBJECTS["state_list"]:
                state = GLOBAL_OBJECTS["state_list"].pop()
                GLOBAL_OBJECTS["solver"].init({
                    "T0": state["transformation"],
                    "r0": state["rc"],
                    "v0": state["vc"],
                    "L0": state["l"],
                    "v_wind": GLOBAL_OBJECTS["v_wind"],
                    "density": GLOBAL_OBJECTS["density"],
                })
                GLOBAL_OBJECTS["current_state"] = state
            else:
                return ("animation", "已经位于初始状态")
        elif event.key == pygame.K_u and GLOBAL_OBJECTS["state_list"]:
            state = GLOBAL_OBJECTS["state_list"][0]
            solver.init({
                "T0": state["transformation"],
                "r0": state["rc"],
                "v0": state["vc"],
                "L0": state["l"],
                "v_wind": GLOBAL_OBJECTS["v_wind"],
                "density": GLOBAL_OBJECTS["density"],
            })
            reader.reset()
            GLOBAL_OBJECTS["current_state"] = state
            GLOBAL_OBJECTS["state_list"].clear()

        elif event.key == pygame.K_ESCAPE:
            return ("control_unset", None)
    return ("animation", None)

def view_input_function(event):
    """
    ASDW控制相机的方向:
        W和S控制theta角（和z轴的夹角）
        A和D控制phi角（在xOy平面内和x轴正方向的夹角）
    方向键+JK控制相机位置:
        方向键控制在xOy平面内的运动
        z轴方向：J向下，K向上
    按U回复初始状态（包括位置和方向）
    """
    phi = GLOBAL_OBJECTS["current_view_angle"]["phi"]

    dir_front = np.array([math.cos(phi), math.sin(phi), 0])
    dir_right = np.array([math.sin(phi), -math.cos(phi), 0])
    dir_up = np.array([0, 0, 1])
    dict_move = {
        pygame.K_RIGHT: dir_right,
        pygame.K_LEFT: -1 * dir_right,
        pygame.K_UP: dir_front,
        pygame.K_DOWN: -1 * dir_front,
        pygame.K_k: dir_up,
        pygame.K_j: -1 * dir_up
    }
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_a:
            GLOBAL_OBJECTS["current_view_angle"]["phi"] += GLOBAL_OBJECTS["angle_delta"]
        elif event.key == pygame.K_d:
            GLOBAL_OBJECTS["current_view_angle"]["phi"] -= GLOBAL_OBJECTS["angle_delta"]
        elif event.key == pygame.K_w:
            GLOBAL_OBJECTS["current_view_angle"]["theta"] += GLOBAL_OBJECTS["angle_delta"]
        elif event.key == pygame.K_s:
            GLOBAL_OBJECTS["current_view_angle"]["theta"] -= GLOBAL_OBJECTS["angle_delta"]

        elif event.key in dict_move.keys():
            GLOBAL_OBJECTS["current_view_pos"] = GLOBAL_OBJECTS["current_view_pos"] + \
                GLOBAL_OBJECTS["pos_delta"] * dict_move[event.key]

        elif event.key == pygame.K_u:
            GLOBAL_OBJECTS["current_view_pos"] = np.array(GLOBAL_OBJECTS["init_view_pos"])
            GLOBAL_OBJECTS["current_view_angle"] = dict(GLOBAL_OBJECTS["init_view_angle"])

        elif event.key == pygame.K_ESCAPE:
            return ("control_unset", None)

    return ("view", None)

#*_input_function:其中*是状态的名称，传入的参数str_in是接受的输入，
#返回一个新的状态和消息（没有则置位None）的元组
def get_input_func_by_dict(dic, state):
    def input_function(str_in):
        if str_in in dic.keys():
            return (dic[str_in], None)
        if str_in == "help":
            return (state, STATES[state]["greet"])
        return (state, "未识别的指令")
    return input_function

def angle_input_converter(str_in):
    deg = float(str_in)
    return math.pi * deg / 180

def set_input_func(str_in):
    """
    设置变量的处理函数
    """
    if str_in == "q":
        return ("control_unset", None)
    rules = [
        (r"d_angle[\s]*(.*)", "angle_delta", angle_input_converter),
        (r"d_pos[\s]*(.*)", "pos_delta", float),
        (r"play_max[\s]*(.*)", "ani_input_handler", "max_play_time", float, "attr"),
        (r"step[\s]*(.*)", "solver", "set_interval", float, "function")
    ]
    for rule in rules:
        match = re.match(rule[0], str_in)
        if match is not None:
            try:
                if len(rule) == 3:
                    GLOBAL_OBJECTS[rule[1]] = rule[2](match.groups()[-1])
                elif len(rule) == 5 and rule[4] == "attr":
                    setattr(GLOBAL_OBJECTS[rule[1]], rule[2], rule[3](match.groups()[-1]))
                elif len(rule) == 5 and rule[4] == "function":
                    getattr(GLOBAL_OBJECTS[rule[1]], rule[2])(rule[3](match.groups()[-1]))
                return ("set", "设置成功")
            except ValueError:
                return ("set", "数值错误")
    return ("set", "未找到设置项")

class AniInputHandler:
    def __init__(self):
        self.time_played = 0
        self.max_play_time = -1
        self.is_playing = False
        self.key_rope_map = [pygame.K_1, pygame.K_2]

GLOBAL_OBJECTS = {
    "init_state": {
        "rc": np.array([0, 5, 0]),
        "transformation": np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        "vc": np.array([0, 0, 0]),
        "l": np.array([0, 0, 0]),
    },
    "init_view_angle": {"theta": math.pi / 2, "phi": math.pi / 2},
    "init_view_pos": np.array([0, -5, 2]),
    "v_wind": np.array([0, 5, 0]),
    "density": 1.225,
    "is_playing": False,
    "state_list": [],
    "angle_delta": math.pi / 36,
    "pos_delta": 1,
    "ani_input_handler": AniInputHandler()
#pylint: disable = C0330
#等待在main中进行初始化的键
# "current_view_angle"
# "current_view_pos"
# "current_state"
# "solver"
#pylint: enable = C0330
}


STATES = {
    "control_unset":{
        "is_final": False,
        "input_func": get_input_func_by_dict({
            "ani": "animation",
            "view": "view",
            "q": "quit",
            "set": "set",
        }, "control_unset"),
        "greet": "输入ani或view进入控制动画或视频选项,输入set设置变量,输入q退出"
    },
    "animation":{
        "is_final": True,
        "input_func": animation_input_function,
        "greet": "按空格开始或停止模拟，右方向键单步模拟，左方向键单步后退,按esc返回上一级"
    },
    "view":{
        "is_final": True,
        "input_func": view_input_function,
        "greet": "ASDW控制相机在水平（xOy平面）和竖直平面内转动，" + \
            "方向键(xOy平面)和JK（z轴）控制相机位置, 按U回复到初始视角"
    },
    "set":{
        "is_final": False,
        "input_func": set_input_func,
        "greet": "设置移动相机时的角度增量（d_angle *）,位置增量(d_pos *)" + \
            "最大动画播放时间(play_max *)"
    }
}

def get_look_at_parameter(angles, pos):
    """
    从球面角和位置推算出传入gluLookAt的参数
    """
    direction = np.array([
        math.sin(angles["theta"]) * math.cos(angles["phi"]),
        math.sin(angles["theta"]) * math.sin(angles["phi"]),
        math.cos(angles["theta"])
    ])
    dir_up = np.array([
        math.sin(abs(math.pi / 2 - angles["theta"])) * math.cos(angles["phi"] + math.pi),
        math.sin(abs(math.pi / 2 - angles["theta"])) * math.sin(angles["phi"] + math.pi),
        math.cos(abs(math.pi / 2 - angles["theta"]))
    ])
    return {
        "eye": pos,
        "center": pos + direction,
        "up": dir_up
    }

def main():
    """
    当前为测试入口
    """
    pdb.set_trace()
    display = (800, 600)
    pygame.init()
    pygame.display.set_mode(display, pygame.OPENGL | pygame.DOUBLEBUF)

    normalize_kite(SIMPLE_KITE)
    gl.glEnable(gl.GL_DEPTH_TEST)
    for panel in SIMPLE_KITE["panels"]:
        set_quad_vert(panel)
    GLOBAL_OBJECTS["current_state"] = GLOBAL_OBJECTS["init_state"]
    GLOBAL_OBJECTS["current_view_angle"] = dict(GLOBAL_OBJECTS["init_view_angle"])
    GLOBAL_OBJECTS["current_view_pos"] = np.array(GLOBAL_OBJECTS["init_view_pos"])
    view_parameter = {
        "look_at": get_look_at_parameter(GLOBAL_OBJECTS["current_view_angle"], \
            GLOBAL_OBJECTS["current_view_pos"]),
        "perspective": (45, display[0] / display[1], 0.1, 50)
    }
    #设置求解器
    #reader = reader_3d.Reader3DConstF(0)
    #reader = reader_3d.Reader3DStableLength()
    reader = Reader3DDynamicLength(2, 0.2, 1)
    reader.set_attach_pts(SIMPLE_KITE["attach_pts"])
    GLOBAL_OBJECTS["solver"] = dynamic_3d.Dynamic3D(reader, 0.005)
    for panel in SIMPLE_KITE["panels"]:
        GLOBAL_OBJECTS["solver"].add_panel(panel)
    for pt in SIMPLE_KITE["mass_pts"]:
        GLOBAL_OBJECTS["solver"].add_mass_point(pt)
    GLOBAL_OBJECTS["state_list"].append(GLOBAL_OBJECTS["init_state"])
    GLOBAL_OBJECTS["solver"].init({
        "T0": GLOBAL_OBJECTS["init_state"]["transformation"],
        "r0": GLOBAL_OBJECTS["init_state"]["rc"],
        "v0": GLOBAL_OBJECTS["init_state"]["vc"],
        "L0": GLOBAL_OBJECTS["init_state"]["l"],
        "v_wind": np.array([0, 5, 0]),
        "density": 1.225
    })
    #首次绘制风筝
    draw_kite(SIMPLE_KITE, GLOBAL_OBJECTS["current_state"], view_parameter)
    pygame.display.flip()

    current_state = "control_unset"
    print(STATES[current_state]["greet"])
    while True:
        #先进入命令行交互
        while not STATES[current_state]["is_final"]:
            str_in = input()
            new_state, msg = STATES[current_state]["input_func"](str_in)
            if msg is not None:
                print(msg)
            if new_state == "quit":
                quit(0)
            if new_state != current_state:
                print(STATES[new_state]["greet"])
                current_state = new_state
#                pdb.set_trace()
            #考虑到可能存在参数改变，需要重绘
            view_parameter["look_at"] = get_look_at_parameter(GLOBAL_OBJECTS["current_view_angle"],\
                GLOBAL_OBJECTS["current_view_pos"])
            draw_kite(SIMPLE_KITE, GLOBAL_OBJECTS["current_state"], view_parameter)
            pygame.display.flip()
        #进入pygame交互
        while True:
            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                quit()
            new_state, msg = STATES[current_state]["input_func"](event)
            if msg is not None:
                print(msg)
            if new_state == "quit":
                quit(0)
            if new_state != current_state:
                print(STATES[new_state]["greet"])
                current_state = new_state
                break
            view_parameter["look_at"] = get_look_at_parameter(GLOBAL_OBJECTS["current_view_angle"],\
                GLOBAL_OBJECTS["current_view_pos"])
            draw_kite(SIMPLE_KITE, GLOBAL_OBJECTS["current_state"], view_parameter)
            pygame.display.flip()
            pygame.time.wait(10)



main()
