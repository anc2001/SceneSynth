from main.program_extraction.dataset import get_dataloaders
from main.program_extraction.data_processing import extract_programs, \
    generate_most_restrictive_program, write_program_data, read_program_data, \
        verify_program_validity
from main.common import get_scene_list

def execute_program_extraction():
    scene_list = get_scene_list()
    xs, ys = extract_programs(scene_list)
    write_program_data(xs, ys)