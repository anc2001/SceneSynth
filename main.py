from main.common.scene import get_scene_list
from main.program_extraction import extract_programs

import numpy as np

scene_list = get_scene_list()
extract_programs(scene_list)
