"""
3D风筝Demo
"""

import numpy as np
import pygame
import OpenGL.GL as gl
import OpenGL.GLU as glu
#pylint: disable = W0105
"""
三块夹角为120度的板子，
高度0.5，
左右长度各0.5，中间长度1
原点在等腰梯形底边中点:
    ^ y   _______
    |   /         \
    |  /_ _ _._ _ _\
        ---> x

风筝被z平面上下各切一半
"""
#pylint: enable = W0105

SIMPLE_KITE = {
    "panels":[
        {
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
                kite_status["r_c"] + kite_status["transformation"].dot(pt)
            ))
    gl.glEnd()

    gl.glLineWidth(1)
    gl.glBegin(gl.GL_LINES)
    gl.glColor3fv((1, 1, 1))
    for panel in kite_data["panels"]:
        for edge in panel["quad_edges"]:
            for pt in edge:
                gl.glVertex3fv(world_transformation.dot(
                    kite_status["r_c"] + kite_status["transformation"].dot(pt)
                ))
    gl.glEnd()

    #绘制风筝线
    gl.glLineWidth(3)
    gl.glBegin(gl.GL_LINES)
    for pt in kite_data["attach_pts"]:
        gl.glVertex3fv(world_transformation.dot(
            kite_status["r_c"] + kite_status["transformation"].dot(pt)
        ))
        gl.glVertex3fv((0, 0, 0))
    gl.glEnd()

def main():
    """
    当前为测试入口
    """
    display = (800, 600)
    normalize_kite(SIMPLE_KITE)
    for panel in SIMPLE_KITE["panels"]:
        set_quad_vert(panel)
    kite_status = {
        "r_c": np.array([0, 5, 0]),
        "transformation": np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    }
    view_parameter = {
        "look_at":{
            "eye": np.array([0, 0, 20]),
            "center": np.array([0, 0, 0]),
            "up": np.array([0, 1, 0])
        },
        "perspective":(45, display[0] / display[1], 0.1, 50),
    }

    pygame.init()
    pygame.display.set_mode(display, pygame.OPENGL | pygame.DOUBLEBUF)
    draw_kite(SIMPLE_KITE, kite_status, view_parameter)
    pygame.display.flip()
    input()
#    gl.glEnable(gl.GL_DEPTH_TEST)
#    while True:
#        event = pygame.event.poll()
#        angle_delta = 5
#        if event.type == pygame.QUIT:
#            break
#        elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
#            view_parameter["angle"] += angle_delta
#        elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
#            view_parameter["angle"] -= angle_delta
#        draw_kite(SIMPLE_KITE, kite_status, view_parameter)
#        pygame.display.flip()
#        pygame.time.wait(10)


main()
