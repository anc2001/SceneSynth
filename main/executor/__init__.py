from main.executor.location_constraints import attach, reachable_by_arm
from main.executor.orientation_constraints import align, face
from main.config import constraint_types

import numpy as np

# For directions -> all is a temporary token that means nothing, for location constraints involving the wall
# all is always the argument, if any other direction is given 
# For orientation constraints 

def solve_constraint(constraint, scene, query_object):
    constraint_type = constraint[0]
    reference_object = scene.objects[constraint[2]]
    if constraint_types[constraint_type] == 'attach':
        return attach(query_object, reference_object, constraint[3], scene)
    elif constraint_types[constraint_type] == 'reachable_by_arm':
        return reachable_by_arm(query_object, reference_object, constraint[3], scene)
    elif constraint_types[constraint_type] == 'align':
        return align(query_object, reference_object, scene)
    elif constraint_types[constraint_type] == 'face':
        return face(query_object, reference_object, scene)

def ensure_placement_validity(centroid_mask, scene, query_object):
    pass