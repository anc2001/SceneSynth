from main.config import grid_size, num_angles, \
    max_allowed_sideways_reach, max_attach_distance, \
    direction_types
from main.common.utils import write_triangle_to_mask

import numpy as np

extent_reference_guide = np.array([
    [0, 2, 0, 2],
    [2, 0, 2, 0],
    [0, 2, 0, 2],
    [2, 0, 2, 0]
])

def location_constraint(query_object, reference_object, direction, scene, attach=True):
    mask = np.zeros((num_angles, num_angles, grid_size, grid_size))
    forgiveness = max_attach_distance if attach else max_allowed_sideways_reach - max_attach_distance
    world_directions = []
    if direction_types[direction] == '<pad>':
        world_directions = range(num_angles)
    else:
        world_directions = [reference_object.local_direction_to_world(direction)]
    
    for world_direction in world_directions:
        for possible_orientation in range(num_angles):
            line_segs = reference_object.line_segs_in_direction(world_direction)
            idx = extent_reference_guide[world_direction][possible_orientation]
            distance = query_object.extent[idx] / 2
            distance = distance if attach else distance + max_attach_distance 
            for line_seg in line_segs:
                p1 = line_seg.p1 + line_seg.normal * distance
                p2 = line_seg.p2 + line_seg.normal * distance
                p3 = p1 + line_seg.normal * forgiveness
                p4 = p2 + line_seg.normal * forgiveness
                triangles = np.array([
                    [p1, p2, p3],
                    [p2, p4, p3]
                ])
                for triangle in triangles:
                    write_triangle_to_mask(
                        triangle, 
                        scene, 
                        mask[world_direction, possible_orientation]
                    )
    return mask