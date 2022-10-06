from main.common.scene import get_scene_list
from main.program_extraction.data_processing import test
from main.config import grid_size

scene_list = get_scene_list()
scene = scene_list[0]
for i, (subscene, query_object) in enumerate(scene.permute()):
    print(i)
    test(subscene, query_object, str(i))
