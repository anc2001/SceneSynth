from main.compiler.location_constraints import location_constraint
from main.compiler.orientation_constraints import align, face
from main.compiler.mask_operations import \
    collapse_mask, convert_mask_to_image
from main.common.object import BBox
from main.config import constraint_types, num_angles
from main.common.utils import get_grid_bounds

import numpy as np

# For directions -> all is a temporary token that means nothing, for location constraints involving the wall
# all is always the argument, if any other direction is given 
# For orientation constraints 

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

def ensure_placement_validity(centroid_mask, scene, query_object):
    scene_mask = scene.convert_to_mask()
    for possible_orientation in range(num_angles):
        bbox = BBox(query_object.extent / 2)
        bbox.rotate(np.array([possible_orientation * (2 * np.pi / num_angles)]))
        bbox.translate(scene.corner_pos)
        location_slice = centroid_mask[possible_orientation]
        possible_placements = np.argwhere(location_slice == 1)
        for x_and_y in possible_placements:
            i = x_and_y[0]
            j = x_and_y[1]
            translation = np.array([i + 0.5, 0, j + 0.5]) * scene.cell_size
            bbox.translate(translation)
            min_bound = np.amin(bbox.vertices, axis = 0)
            max_bound = np.amax(bbox.vertices, axis = 0)
            grid_min_bound, grid_max_bound = get_grid_bounds(min_bound, max_bound, scene)
            object_slice = scene_mask[
                grid_min_bound[0] : grid_max_bound[0] + 1,
                grid_min_bound[2] : grid_max_bound[2] + 1
            ]

            if np.any(object_slice):
                centroid_mask[possible_orientation, i, j] = 0
            
            # if len(object_slice):
            #     total_count = (object_slice.shape[0] * object_slice.shape[1])
            #     if np.sum(object_slice) / total_count > 0.1:

            bbox.translate(-translation)
