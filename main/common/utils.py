# General utility functions for common objects 
import numpy as np

def raise_exception(type):
    if type == 'tree':
        print("Tree is invalid")
        raise Exception()

def normalize(vector):
    return vector / (np.linalg.norm(vector) + 1e-9)

def point_triangle_test(point_3d, triangle_3d):
    # Convert point
    point = [point_3d[0], point_3d[2]]
    triangle = [[vertex[0], vertex[2]] for vertex in triangle_3d]
    def sign(point, v1, v2):
        return (point[0] - v1[0]) * (v1[1] - v2[1]) - (v1[0] - v2[0]) * (point[1] - v1[1])

    d1 = sign(point, triangle[0], triangle[1])
    d2 = sign(point, triangle[1], triangle[2])
    d3 = sign(point, triangle[2], triangle[0])

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def get_rot_matrix(theta):
    costheta = float(np.cos(theta))
    sintheta = float(np.sin(theta))

    rotation_m = np.asarray([
            [costheta,0,sintheta],
            [0,1,0],
            [-sintheta,0,costheta],
            ])
    return rotation_m
    