from main.program_extraction.program_processing import extract_programs, \
    generate_most_restrictive_program, write_program_data, read_program_data, \
        verify_program_validity
from main.program_extraction.scene_processing import filter_scenes
from main.common import get_scene_list
from main.common.utils import write_data
from main.config import data_filepath

import os 

def execute_program_extraction():
    scene_list = get_scene_list('adrian_parse.pkl')
    xs, ys = extract_programs(scene_list)
    program_data = {
        "xs" : xs,
        "ys" : ys
    }
    filepath = os.path.join(data_filepath, 'program_data.pkl')
    write_data(program_data, filepath)