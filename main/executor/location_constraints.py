from main.config import grid_size, num_angles, direction_types

import numpy as np

def attach(query_object, reference_object, direction, scene):
    if direction_types[direction] == '<pad>':
        direction = np.arange(num_angles)
    mask = np.zeros((num_angles, num_angles, grid_size, grid_size))
    line_segs = reference_object.line_segs_in_direction(direction)

def reachable_by_arm(query_object, reference_object, scene):
    mask = np.zeros((num_angles, num_angles, grid_size, grid_size))