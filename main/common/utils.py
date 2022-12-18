from main.config import grid_size, bin_width, num_angles
from main.common.mesh_to_mask import render, get_triangles

import numpy as np
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
    query_object_vector[0, 4] = 0 # Position x
    query_object_vector[0, 5] = 0 # Position y
    query_object_vector[0, 6] = 0 # Rotation 
    scene_vector = np.append(
        scene.vectorize(),
        query_object_vector,
        axis = 0
    )
    return scene_vector
    
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

def get_rot_matrix(theta):
    costheta = float(np.cos(theta))
    sintheta = float(np.sin(theta))

    rotation_m = np.asarray([
            [costheta,0,sintheta],
            [0,1,0],
            [-sintheta,0,costheta],
            ])
    return rotation_m

def render_orthographic(verts, faces, corner_pos, cell_size):
    new_verts = np.clip(
        (verts - corner_pos) / cell_size, 
        [0, 0, 0], 
        [grid_size, grid_size, grid_size]
    )
    new_verts = new_verts[:,[0,2,1]]
    triangles = list(get_triangles(new_verts, faces))
    triangles = np.asarray(triangles, dtype=np.float32)
    img = render(triangles, grid_size, flat = True)
    return img

def get_grid_bounds(min_bound, max_bound, scene_corner_pos, scene_cell_size):
    grid_min_bound = (min_bound - scene_corner_pos) / scene_cell_size
    grid_min_bound = np.minimum(
        np.maximum(grid_min_bound, [0, 0, 0]), 
        [grid_size - 1, 0, grid_size - 1]
    ).astype(int)
    
    grid_max_bound = (max_bound - scene_corner_pos) / scene_cell_size
    grid_max_bound = np.minimum(
        np.maximum(grid_max_bound, [0, 0, 0]), 
        [grid_size - 1, 0, grid_size - 1]
    ).astype(int)

    return grid_min_bound, grid_max_bound

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