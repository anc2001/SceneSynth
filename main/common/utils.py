from main.config import grid_size, bin_width, num_angles

import numpy as np
from itertools import chain, combinations
import os, shutil
import pickle 
from numba import jit

def write_data(data, filepath):
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_data(filepath):
    with open(filepath, 'rb') as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data

def vectorize_scene(scene, query_object):
    # Convert scene to object list and program to structures + constraints
    query_object_vector = query_object.vectorize(scene.objects[0])
    query_object_vector[0, 4] = 0
    query_object_vector[0, 5] = 0
    scene_vector = np.append(
        scene.vectorize(),
        query_object_vector,
        axis = 0
    )
    return scene_vector

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
def raise_exception(type):
    if type == 'tree':
        print("Tree is invalid")
        raise Exception()
    if type == 'front_facing':
        print("Called Face on non front facing object")
        raise Exception()

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

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

def get_grid_bounds(min_bound, max_bound, scene):
    grid_min_bound = (min_bound - scene.corner_pos) / scene.cell_size
    grid_min_bound = np.minimum(
        np.maximum(grid_min_bound, [0, 0, 0]), 
        [grid_size - 1, 0, grid_size - 1]
    ).astype(int)
    
    grid_max_bound = (max_bound - scene.corner_pos) / scene.cell_size
    grid_max_bound = np.minimum(
        np.maximum(grid_max_bound, [0, 0, 0]), 
        [grid_size - 1, 0, grid_size - 1]
    ).astype(int)

    return grid_min_bound, grid_max_bound

@jit(nopython=True)
def write_triangle_to_image(triangle, scene, image, color):
    min_bound = np.amin(triangle, axis = 0)
    max_bound = np.amax(triangle, axis = 0)
    grid_min_bound, grid_max_bound = get_grid_bounds(min_bound, max_bound, scene)
    for i in range(grid_min_bound[0], grid_max_bound[0] + 1):
        for j in range(grid_min_bound[2], grid_max_bound[2] + 1):
            cell_center = scene.corner_pos + np.array([i + 0.5, 0, j + 0.5]) * scene.cell_size
            if point_triangle_test(cell_center, triangle):
                image[i, j, :] = color

@jit(nopython=True)
def write_triangle_to_mask(triangle, scene, mask):
    min_bound = np.amin(triangle, axis = 0)
    max_bound = np.amax(triangle, axis = 0)
    grid_min_bound, grid_max_bound = get_grid_bounds(min_bound, max_bound, scene)
    for i in range(grid_min_bound[0], grid_max_bound[0] + 1):
        for j in range(grid_min_bound[2], grid_max_bound[2] + 1):
            cell_center = scene.corner_pos + np.array([i + 0.5, 0, j + 0.5]) * scene.cell_size
            if point_triangle_test(cell_center, triangle):
                mask[i, j] = 1  

def angle_to_index(angle):
    angle = 2 * np.pi + angle if angle < 0 else angle
    angle_idx = np.around(angle / bin_width).astype(int) % num_angles
    return angle_idx

def vector_angle_index(start, end):
    """
    start - numpy array of shape (3,)
    end - numpy array of shape (3,)
    Calculates the angle needed to rotate start by CCW so that it equals end
    Returns this angle binned as an index (for NUM_ANGLES = 4, 0 -> 0, pi/2 -> 1, ...)
    """
    angle = np.arccos(np.dot(start, end))
    plane_normal = np.cross(start, end)
    if plane_normal[1] > 0:
        angle = 2 * np.pi - angle
    angle_idx = angle_to_index(angle)
    return angle_idx