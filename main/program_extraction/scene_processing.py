from main.common import get_scene_list
from main.common.utils import write_data
from main.config import data_filepath

from itertools import combinations
import numpy as np
import os
from tqdm import tqdm

def execute_scene_filtering(filename):
    if filename == 'kai_parse.pkl':
        scene_list = get_scene_list('kai_parse.pkl')
        accepted_index_list, rejected_index_list = filter_scenes(scene_list)

        summary_file = os.path.join(data_filepath, "rejected_index_list.txt")
        fin = open(summary_file, 'w')
        for index in rejected_index_list:
            fin.write(f"{index}-{scene_list[index].id}\n")
        fin.close()

        valid_scenes = scene_list[accepted_index_list]
        write_data(valid_scenes, os.path.join(data_filepath, "adrian_parse.pkl"))
        print(f"Percent of remaining scenes: {len(accepted_index_list) / len(scene_list)}")
    elif filename == 'subsampled_rooms.pkl':
        pass

# Filtering step on data to remove scenes that break visual perception assumptions 
def filter_scenes(scene_list):
    accepted_index_list = []
    rejected_index_list = []
    for index, scene in enumerate(tqdm(scene_list)):
        # Check if any of the objects fall outside of the room 
        valid = True
        possible_pairs = list(combinations(range(len(scene.objects) - 1), 2))
        for possible_pair in possible_pairs:
            # Calculate the overlapping area between two objects 
            object_1 = scene.objects[possible_pair[0] + 1]
            object_2 = scene.objects[possible_pair[1] + 1]
            distance = object_2.distance(object_1)
            if not distance:
                min_bound = np.amin(object_1.bbox.vertices, axis = 0)
                max_bound = np.amax(object_1.bbox.vertices, axis = 0)
                sub_quad = np.clip(object_2.bbox.vertices, min_bound, max_bound)
                area = np.linalg.norm(np.cross(sub_quad[1] - sub_quad[0], sub_quad[2] - sub_quad[0]))
                threshold = 0.4
                max_percent_overlap = max(area / object_1.area(), area / object_2.area())
                # If percent overlap significant enough throw out room 
                if max_percent_overlap > threshold:
                    valid = False
                    rejected_index_list.append(index)
                    break
        if valid:
            accepted_index_list.append(scene)          
    
    return accepted_index_list, rejected_index_list

def subsample_rooms(scene_list):
    scenes = []
    query_objects = []
    for scene in tqdm(scene_list):
        for scene, query_object in scene.permute():
            scenes.append(scene)
            query_objects.append(query_object)
    to_write = {
        "scenes" : scenes,
        "query_objects" : query_objects
    }

    filepath =  os.path.join(data_filepath, 'subsampled_rooms.pkl')
    write_data(to_write, filepath)

# Scene similarity metric to get possible candidate scenes for program combination 
def get_similar_scenes(scene, scene_list):
    # Start simple -> find the other scenes that contain the same object types as current scene 
    pass

# Metrics for measuring the specificity and generality of the program for scenes 
def foo(program, candidate_scenes):
    # Measure the number of scenes which the program successfully executes on (validity/specificity)
    # Measure the number of spaces 
    for scene in candidate_scenes:
        pass