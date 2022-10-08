from main.config import grid_size, num_angles, \
    max_allowed_sideways_reach, max_attach_distance, \
    direction_types
from main.common.utils import write_triangle_to_mask, \
    normalize, raise_exception

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

    reference_line_seg_direction = reference_object.local_direction_to_world(direction)
    pad_flag = direction_types[direction] == '<pad>'
    #  Also should reconsider whether to use the reference object line seg length as the indicator for how wide the mask will be 
    for world_direction in range(num_angles):
        for possible_orientation in range(num_angles):
            side = world_direction if pad_flag else reference_line_seg_direction
            line_segs = reference_object.line_segs_in_direction(side)
            idx = extent_reference_guide[side][possible_orientation]

            distance1 = 0 if attach else (query_object.extent[idx] / 2) + max_attach_distance 
            distance2 = (query_object.extent[idx] / 2) + forgiveness if attach else forgiveness
            for line_seg in line_segs:
                p1 = line_seg.p1 + line_seg.normal * distance1
                p2 = line_seg.p2 + line_seg.normal * distance1

                extension = query_object.extent[0 if idx else 2] - line_seg.length()
                if extension > 0:
                    u = normalize(p1 - p2)
                    p1 = p1 + u * extension
                    p2 = p2 - u * extension
                    
                p3 = p1 + line_seg.normal * distance2
                p4 = p2 + line_seg.normal * distance2
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

def align(query_object, reference_object, scene):
    mask = np.zeros((num_angles, num_angles, grid_size, grid_size))
    reference_fronts = reference_object.world_semantic_fronts()
    query_fronts = query_object.semantic_fronts
    for world_direction in range(num_angles):
        if world_direction in reference_fronts:
            for direction in query_fronts:
                valid_orientation = (world_direction - direction) % num_angles
                mask[world_direction, valid_orientation, :, :] = 1
        elif reference_object.front_facing:
            for direction in query_fronts:
                valid_orientation = (list(reference_fronts)[0] - direction) % num_angles
                mask[world_direction, valid_orientation, :, :] = 1
    return mask

def face(query_object, reference_object, scene):
    if not query_object.front_facing:
        raise_exception("front_facing")
        
    mask = np.zeros((num_angles, num_angles, grid_size, grid_size))
    query_front = list(query_object.semantic_fronts)[0]
    for world_direction in range(num_angles):
        line_segs = reference_object.line_segs_in_direction(world_direction)
        opposite_direction = int((world_direction - (num_angles / 2)) % num_angles)
        valid_orientation = (opposite_direction - query_front) % num_angles
        for line_seg in line_segs:
            p1 = line_seg.p1
            p2 = line_seg.p2
            
            idx = extent_reference_guide[world_direction][valid_orientation]
            extension = query_object.extent[0 if idx else 2] - line_seg.length()
            if extension > 0:
                u = normalize(p1 - p2)
                p1 = p1 + u * extension
                p2 = p2 - u * extension
                   
            p3 = p1 + line_seg.normal * grid_size * scene.cell_size
            p4 = p2 + line_seg.normal * grid_size * scene.cell_size
            triangles = np.array([
                [p1, p2, p3],
                [p2, p4, p3]
            ])

            for triangle in triangles:
                write_triangle_to_mask(
                    triangle, 
                    scene, 
                    mask[world_direction, valid_orientation]
                )
            for i in np.delete(range(num_angles), world_direction):
                mask[i, valid_orientation] = np.array(mask[world_direction, valid_orientation])
            
    return mask