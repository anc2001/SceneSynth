from main.compiler.location_constraints import location_constraint
from main.compiler.orientation_constraints import align, face
from main.config import constraint_types, num_angles, grid_size

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
    pass

def collapse_mask(mask_4d):
    mask_3d = np.zeros((num_angles, grid_size, grid_size))
    for possible_orientation in range(num_angles):
        to_collapse = mask_4d[:, possible_orientation, :, :]
        collapsed_mask_2d = np.zeros((grid_size, grid_size)).astype(int)
        for i in range(num_angles):
            collapsed_mask_2d |= to_collapse[i].astype(int)
        mask_3d[possible_orientation] = collapsed_mask_2d
    return mask_3d

def convert_mask_to_image(mask, scene):
    scene_image = scene.convert_to_image()
    if len(mask.shape) == 4:
        image = np.array([])
        for world_direction in range(num_angles):
            stacked_references = np.rot90(mask[world_direction], axes=(1,2))
            image_row = np.array([])
            for i in range(num_angles):
                mask_image = np.repeat(                
                    np.expand_dims(
                        stacked_references[i], 
                        axis = 2
                    ),
                    3,
                    axis = 2
                )
                image_slice = np.clip(scene_image - mask_image, 0, 1)
                if len(image_row):
                    image_row = np.append(image_row, image_slice, axis = 1)
                else:
                    image_row = image_slice
            if len(image):
                image = np.append(image, image_row, axis = 0)
            else:
                image = image_row
        return image
    elif len(mask.shape) == 3:
        stacked_references = np.rot90(mask, axes=(1,2))
        image = np.array([])
        for i in range(num_angles):
            mask_image = np.repeat(                
                np.expand_dims(
                    stacked_references[i], 
                    axis = 2
                ),
                3,
                axis = 2
            )
            image_slice = np.clip(scene_image - mask_image, 0, 1)
            if len(image):
                image = np.append(image, image_slice, axis = 1)
            else:
                image = image_slice
        return image