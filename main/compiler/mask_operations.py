from main.config import num_angles, grid_size

import numpy as np

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