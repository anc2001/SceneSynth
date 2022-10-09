from main.common import get_scene_list
from main.program_extraction import generate_most_restrictive_program
from main.config import image_filepath

import matplotlib.image as img
import os 

def test(scene, query_object, program_name):
    program = generate_most_restrictive_program(scene, query_object)
    program.evaluate(scene, query_object)
    program.print_program(program_name, scene, query_object)

scene_list = get_scene_list()
scene = scene_list[1]
scene_image = scene.convert_to_image()
img.imsave(os.path.join(image_filepath, "scene.png"), scene_image)
for i, (subscene, query_object) in enumerate(scene.permute()):
    print(i)
    test(subscene, query_object, str(i))
