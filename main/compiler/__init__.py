from main.compiler.location_constraints import attach, reachable_by_arm
from main.compiler.orientation_constraints import align, face
from main.config import grid_size, num_angles, constraint_types

import numpy as np

# For directions -> all is a temporary token that means nothing, for location constraints involving the wall
# all is always the argument, if any other direction is given 
# For orientation constraints 

def solve_constraint(constraint, scene, query_object):
    mask = np.zeros((num_angles, num_angles, grid_size, grid_size))
    constraint_type = constraint[0]
    if constraint_types[constraint_type] == 'attach':
        pass
    elif constraint_types[constraint_type] == 'reachable_by_arm':
        pass
    elif constraint_types[constraint_type] == 'align':
        pass
    elif constraint_types[constraint_type] == 'face':
        pass

def ensure_placement_validity(centroid_mask, scene, query_object):
    pass