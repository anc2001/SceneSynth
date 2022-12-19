from main.config import num_angles
from main.common.utils import get_grid_bounds, render_orthographic
from main.common.object import BBox

import numpy as np

# def convert_mask_to_image(mask, scene_image):
#     stacked_references = np.rot90(mask, axes=(1,2))
#     image = np.array([])
#     for i in range(num_angles):
#         mask_image = np.repeat(                
#             np.expand_dims(
#                 stacked_references[i], 
#                 axis = 2
#             ),
#             3,
#             axis = 2
#         )
#         image_slice = np.clip(scene_image - mask_image, 0, 1)
#         if len(image):
#             image = np.append(image, image_slice, axis = 1)
#         else:
#             image = image_slice
#     return image

def convert_mask_to_image(mask, scene_image):
    stacked_references = np.rot90(mask, axes=(1,2))
    top_row = np.array([])
    bottom_row = np.array([])
    for i in range(num_angles):
        mask_image = np.expand_dims(stacked_references[i], axis = 2)
        mask_image = np.repeat(mask_image, 3, axis = 2)
        image_slice = np.clip(scene_image - mask_image, 0, 1)

        if i == 0:
            top_row = image_slice
        elif i == 1:
            top_row = np.append(top_row, image_slice, axis = 1)
        elif i == 2:
            bottom_row = image_slice
        elif i == 3:
            bottom_row = np.append(bottom_row, image_slice, axis = 1)
    
    image = np.append(top_row, bottom_row, axis = 0)
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

            object_mask = render_orthographic(bbox.vertices, bbox.faces, scene.corner_pos, scene.cell_size)
            overlap = np.logical_and(scene_mask, object_mask)

            if np.sum(overlap) > 300:
                centroid_mask[possible_orientation, i, j] = 0

            bbox.translate(-translation)