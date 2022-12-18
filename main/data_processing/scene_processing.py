from main.common.scene import Scene
from main.common.utils import write_data, read_data
from main.config import data_filepath

from itertools import combinations
import numpy as np
import os
from tqdm import tqdm

def execute_scene_filtering(filename):
    print(filename)
    if filename == 'kai_parse.pkl':
        scene_list = get_scene_list('kai_parse.pkl')
        accepted_index_list, rejected_index_list = filter_scenes(scene_list)
        scene_list = np.array(scene_list)

        new_filename = os.path.join(data_filepath, "adrian_parse.pkl")
        valid_scenes = scene_list[accepted_index_list]
        write_data(valid_scenes, new_filename)
        print(f"Percent of remaining scenes: {len(accepted_index_list) / len(scene_list)}")
    elif filename == 'subsampled_rooms.pkl':
        scenes_and_query_objects = read_data(os.path.join(data_filepath, filename))
        scenes = scenes_and_query_objects['scenes']
        query_objects = scenes_and_query_objects['query_objects']

        # scenes = np.array(scenes)[:500]
        # query_objects = np.array(query_objects)[:500]

        for scene, query_object in zip(scenes, query_objects):
            scene.objects = np.append(scene.objects, query_object)
        
        accepted_index_list, rejected_index_list = filter_scenes(scenes)

        for scene in scenes:
            scene.objects = np.delete(scene.objects, len(scene.objects) - 1)
        
        valid_scenes = np.array(scenes)[accepted_index_list]
        valid_objects = np.array(query_objects)[accepted_index_list]
        valid_pairs = {
            'scenes' : valid_scenes,
            'query_objects' : valid_objects
        }
        
        new_filename = os.path.join(data_filepath, "subsampled_filtered_rooms.pkl")
        write_data(valid_pairs, new_filename)
        print(f"Percent of remaining scenes: {len(accepted_index_list) / len(scenes)}")
    else:
        print(f"Invalid name: {filename}")
        exit()

# Filtering step on data to remove scenes that break visual perception assumptions 
def filter_scenes(scene_list):
    accepted_index_list = []
    rejected_index_list = []
    for index, scene in enumerate(tqdm(scene_list)):
        # Check if any of the objects fall outside of the room 
        valid = True
        if scene.check_if_objects_inside():
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
                    if max_percent_overlap > threshold or area > scene.cell_size * 8:
                        valid = False
                        break
        else:
            valid = False
        
        if valid:
            accepted_index_list.append(index)
        else:
            rejected_index_list.append(index)          
    
    return accepted_index_list, rejected_index_list

def generate_subscenes(filename):
    scene_list = get_scene_list(filename)
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

def get_scene_list(pickle_name):
    filepath = os.path.join(data_filepath, pickle_name)
    if pickle_name == 'kai_parse.pkl':
        scene_list = np.array([])
        scenes = read_data(filepath)
        for room_info in tqdm(scenes):
            scene = Scene(room_info = room_info)
            scene_list = np.append(scene_list, scene)
        return scene_list
    elif pickle_name == 'adrian_parse.pkl':
        return read_data(filepath)
    elif pickle_name == 'subsampled_filtered_rooms.pkl' or pickle_name == 'subsampled_rooms.pkl':
        scenes_and_queries = read_data(filepath)
        scenes = scenes_and_queries['scenes']
        query_objects = scenes_and_queries['query_objects']
        for scene, query_object in tqdm(zip(scenes, query_objects)):
            scene.add_object(query_object)
        return scenes
    else:
        print(f"Invalid filename: {pickle_name}")
        exit()

def get_scene_query_list(pickle_name):
    filepath = os.path.join(data_filepath, pickle_name)
    scenes_and_queries = read_data(filepath)
    scenes = scenes_and_queries['scenes']
    query_objects = scenes_and_queries['query_objects']
    return scenes, query_objects

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