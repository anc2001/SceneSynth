from main.common.scene import get_scene_list
from main.program_extraction import extract_programs
from main.config import grid_size

import matplotlib.image as img
import numpy as np

# image = np.zeros((grid_size, grid_size, 3))
# img.imsave(filepath, image)

scene_list = get_scene_list()
extract_programs(scene_list)

