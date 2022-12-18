from main.config import num_angles
from main.common.utils import get_grid_bounds
from main.common.object import BBox

import numpy as np

def convert_mask_to_image(mask, scene_image):
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