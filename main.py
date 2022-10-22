from main.common import get_scene_list
from main.common.utils import clear_folder
from main.config import image_filepath
from main.program_extraction import generate_most_restrictive_program, \
    execute_program_extraction, \
    verify_program_validity
from main.common.language import verify_program

from main.network import do_everything

import matplotlib.image as img
import os
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np 

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, 
        help="mode to run [program_extraction, train, program_execution]")
    parser.add_argument('-i', '--index', type=int, default=0,
        help="room index to run program_execution on")
    args = parser.parse_args()
    return args

def program_execution(index):
    if not os.path.exists(image_filepath):
        os.mkdir(image_filepath)
    
    room_folder = os.path.join(image_filepath, f"room_{index}")
    if os.path.exists(room_folder):
        clear_folder(room_folder)
    else:
        os.mkdir(room_folder)
    
    scene_list = get_scene_list()
    scene = scene_list[index]
    scene_image = scene.convert_to_image()
    img.imsave(os.path.join(room_folder, "scene.png"), scene_image)
    for i, (subscene, query_object) in tqdm(enumerate(scene.permute())):
        program = generate_most_restrictive_program(subscene, query_object)
        if verify_program_validity(program, subscene, query_object):
            print(f"verified: {i}")
        program.evaluate(subscene, query_object)
        parent_folder = os.path.join(room_folder, str(i))
        os.mkdir(parent_folder)
        program.print_program(subscene, query_object, parent_folder)

def main(args):
    if args.mode == 'program_extraction':
        execute_program_extraction()
    elif args.mode == 'train':
        do_everything()
    elif args.mode == 'program_execution':
        program_execution(args.index)
    else:
        print("Not a recognized mode!")

if __name__ == '__main__':
    args = parseArguments()
    main(args)
    # structure = np.array(['or', 'c', '<pad>', 'c'])
    # c = [[0, 1, 0, 3], [0, 1, 0, 3], [0, 1, 0, 3], [0, 1, 0, 3]]
    # tokens = {
    #     'structure' : structure,
    #     'constraints' : c
    # }
    # valid = verify_program(tokens, 1)
    # print(valid)