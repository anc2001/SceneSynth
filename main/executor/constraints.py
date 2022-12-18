from main.config import grid_size, num_angles, \
    max_allowed_sideways_reach, max_attach_distance, \
    direction_types, direction_types_map
from main.common.utils import normalize, render_orthographic

import numpy as np

extent_reference_guide = np.array([
    [0, 2, 0, 2],
    [2, 0, 2, 0],
    [0, 2, 0, 2],
    [2, 0, 2, 0]
])

def location_constraint(query_object, reference_object, direction, scene, attach=True):
    mask = np.zeros((num_angles, grid_size, grid_size))
    forgiveness = max_attach_distance if attach else max_allowed_sideways_reach - max_attach_distance

    side = reference_object.local_direction_to_world(direction)
    possible_orientations = []
    possible_orientations = range(num_angles)

    for possible_orientation in possible_orientations:
        line_segs = reference_object.line_segs_in_direction(side)
        
        idx = extent_reference_guide[side][possible_orientation]

        object_extent = query_object.extent[idx] / 2
        distance = object_extent if attach else object_extent + max_attach_distance 
        padding = object_extent * 0.15
        for line_seg in line_segs:
            # extend out from lineseg 
            p1 = line_seg.p1 + line_seg.normal * (distance - padding)
            p2 = line_seg.p2 + line_seg.normal * (distance - padding)

            # extend along lineseg 
            extension = query_object.extent[0 if idx else 2] - line_seg.length()
            if extension > 0:
                u = normalize(p1 - p2)
                p1 = p1 + u * extension
                p2 = p2 - u * extension
                
            p3 = p1 + line_seg.normal * (forgiveness + 2 * padding)
            p4 = p2 + line_seg.normal * (forgiveness + 2 * padding)

            verts = [p1, p2, p3, p4]
            faces = [[0, 1, 2], [1, 3, 2]]
            img = render_orthographic(verts, faces, scene.corner_pos, scene.cell_size)
            mask[possible_orientation] += img
    
    return mask

def align(query_object, reference_object, direction):
    mask = np.zeros((num_angles, grid_size, grid_size))
    reference_fronts = reference_object.world_semantic_fronts()
    query_fronts = query_object.semantic_fronts

    if reference_object.id == 0 and not direction == direction_types_map['null']:
        for query_direction in query_fronts:
            valid_orientation = (direction - query_direction) % num_angles
            mask[valid_orientation, :, : ] = 1
    else:
        for world_direction in reference_fronts:
            for query_direction in query_fronts:
                valid_orientation = (world_direction - query_direction) % num_angles
                mask[valid_orientation, :, :] = 1
        
    return mask

def face(query_object, reference_object, scene):
    # Does not matter what the semantic fronts of the reference object are         
    mask = np.zeros((num_angles, grid_size, grid_size))
    query_front = list(query_object.semantic_fronts)[0]
    for world_direction in range(num_angles):
        line_segs = reference_object.line_segs_in_direction(world_direction)

        opposite_direction = int((world_direction - (num_angles / 2)) % num_angles)
        valid_orientation = (opposite_direction - query_front) % num_angles
        for line_seg in line_segs:
            p1 = line_seg.p1
            p2 = line_seg.p2
                               
            p3 = p1 + line_seg.normal * grid_size * scene.cell_size
            p4 = p2 + line_seg.normal * grid_size * scene.cell_size
            verts = [p1, p2, p3, p4]
            faces = [[0, 1, 2], [1, 3, 2]]
            img = render_orthographic(verts, faces, scene.corner_pos, scene.cell_size)
            mask[valid_orientation] += img

    return mask