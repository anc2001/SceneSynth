from matplotlib.pyplot import grid
from main.config import grid_size, num_angles
from main.common.utils import raise_exception, write_triangle_to_mask

import numpy as np

def align(query_object, reference_object, scene):
    mask = np.zeros((num_angles, num_angles, grid_size, grid_size))
    reference_fronts = reference_object.world_semantic_fronts()
    query_fronts = query_object.semantic_fronts
    for world_space_direction in range(num_angles):
        if world_space_direction in reference_fronts:
            for direction in query_fronts:
                valid_orientation = world_space_direction - direction % num_angles
                mask[world_space_direction, valid_orientation, :, :] = 1
        elif reference_object.front_facing:
            for direction in query_fronts:
                valid_orientation = list(reference_fronts)[0] - direction % num_angles
                mask[world_space_direction, valid_orientation, :, :] = 1
    return mask

def face(query_object, reference_object, scene):
    if not query_object.front_facing:
        raise_exception("front_facing")
        
    mask = np.zeros((num_angles, num_angles, grid_size, grid_size))
    query_fronts = query_object.semantic_fronts
    for world_space_direction in range(num_angles):
        line_segs = reference_object.line_segs_in_direction(world_space_direction)
        for line_seg in line_segs:
            p1 = line_seg.p1
            p2 = line_seg.p2
            p3 = p1 + line_seg.normal * grid_size * scene.cell_size
            p4 = p2 + line_seg.normal * grid_size * scene.cell_size
            triangles = np.array([
                [p1, p2, p3],
                [p2, p4, p3]
            ])
            opposite_direction = (world_space_direction - (num_angles / 2)) % num_angles
            for direction in query_fronts:
                valid_orientation = opposite_direction - direction % num_angles
                for triangle in triangles:
                    write_triangle_to_mask(
                        triangle, 
                        scene, 
                        mask[world_space_direction, valid_orientation]
                    )
            
    return mask