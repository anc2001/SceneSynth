from main.data_processing.program_processing import extract_programs, \
    generate_most_restrictive_program, verify_program_validity
from main.data_processing.scene_processing import execute_scene_filtering, generate_subscenes, \
    get_scene_list, get_scene_query_list
from main.common.utils import write_data, read_data
from main.config import data_filepath

import os 

def execute_program_extraction(filename):
    scenes, query_objects = get_scene_query_list(filename)
    program_data = extract_programs(scenes, query_objects)
    write_data(program_data, os.path.join(data_filepath, 'program_data.pkl'))