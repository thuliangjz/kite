"""
3D风筝Demo
"""
import sys
import os
import math
import re
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

class CL3DConst(util.KiteFunction):
    def compute(self, val):
        return 0.5

class CD3DConst(util.KiteFunction):
    def compute(self, val):
        return 0.1

#leading_edge始终是plane_basis的第一个
SIMPLE_KITE = {
    "panels":[
        {
            "c_l": CL3DConst(),
            "c_d": CD3DConst(),
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
            "c_l": CL3DConst(),
            "c_d": CD3DConst(),
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
            "c_l": CL3DConst(),
            "c_d": CD3DConst(),
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
    if GLOBAL_OBJECTS["is_playing"]:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            GLOBAL_OBJECTS["is_playing"] = False
        else:
            GLOBAL_OBJECTS["solver"].step()
            state = GLOBAL_OBJECTS["solver"].get_state()
            GLOBAL_OBJECTS["current_state"] = state
            GLOBAL_OBJECTS["state_list"].append(state)
    elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_SPACE:
            GLOBAL_OBJECTS["is_playing"] = True
        elif event.key == pygame.K_RIGHT:
            GLOBAL_OBJECTS["solver"].step()
            state = GLOBAL_OBJECTS["solver"].get_state()
            GLOBAL_OBJECTS["current_state"] = state
            GLOBAL_OBJECTS["state_list"].append(state)
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
        elif event.key == pygame.K_u:
            state = GLOBAL_OBJECTS["state_list"][0]
            GLOBAL_OBJECTS["solver"].init({
                "T0": state["transformation"],
                "r0": state["rc"],
                "v0": state["vc"],
                "L0": state["l"],
                "v_wind": GLOBAL_OBJECTS["v_wind"],
                "density": GLOBAL_OBJECTS["density"],
            })
            GLOBAL_OBJECTS["current_state"] = state
            GLOBAL_OBJECTS["state_list"] = GLOBAL_OBJECTS["state_list"][:1]
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
    if str_in == "q":
        return ("control_unset", None)
    rules = [
        (r"d_angle[\s]*(.*)", "angle_delta", angle_input_converter),
        (r"d_pos[\s]*(.*)", "pos_delta", float),
    ]
    for rule in rules:
        match = re.match(rule[0], str_in)
        if match is not None:
            try:
                GLOBAL_OBJECTS[rule[1]] = rule[2](match.groups()[-1])
                return ("set", "设置成功")
            except ValueError:
                return ("set", "数值错误")
    return ("set", "未找到设置项")

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
    #等待在main中进行初始化的键
    # "current_view_angle"
    # "current_view_pos"
    # "current_state"
    # "solver"
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
        "greet": "设置移动相机时的角度增量（d_angle *）和位置增量(d_pos *)"
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
    display = (800, 600)
    pygame.init()
    pygame.display.set_mode(display, pygame.OPENGL | pygame.DOUBLEBUF)

    normalize_kite(SIMPLE_KITE)
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
    #reader = reader_3d.Reader3DConstF(2.5)
    reader = reader_3d.Reader3DStableLength()
    reader.set_attach_pts(SIMPLE_KITE["attach_pts"])
    GLOBAL_OBJECTS["solver"] = dynamic_3d.Dynamic3D(reader, 0.01)
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
