from main.compiler.constraints import align, face, \
    location_constraint
from main.compiler.mask_operations import \
    convert_mask_to_image, ensure_placement_validity
from main.config import constraint_types

import numpy as np

def solve_constraint(constraint, scene, query_object):
    constraint_type = constraint[0]
    reference_object = scene.objects[constraint[2]]
    if constraint_types[constraint_type] == 'attach':
        return location_constraint(
            query_object, 
            reference_object, 
            constraint[3], 
            scene, 
            attach=True
        )
    elif constraint_types[constraint_type] == 'reachable_by_arm':
        return location_constraint(
            query_object, 
            reference_object, 
            constraint[3], 
            scene, 
            attach=False
        )
    elif constraint_types[constraint_type] == 'align':
        return align(query_object, reference_object, scene)
    elif constraint_types[constraint_type] == 'face':
        return face(query_object, reference_object, scene)


