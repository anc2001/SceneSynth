from main.common import get_scene_list
from main.common.utils import clear_folder
from main.config import image_filepath
from main.program_extraction import generate_most_restrictive_program, \
    execute_program_extraction, \
    verify_program_validity

import matplotlib.image as img
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np 

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, 
        help="mode to run [program_extraction, program_execution]")
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
    for i, (subscene, query_object) in enumerate(tqdm(scene.permute())):
        program = generate_most_restrictive_program(subscene, query_object)
        if verify_program_validity(program, subscene, query_object):
            print(f"verified: {i}")
        program.evaluate(subscene, query_object)
        fig = program.print_program(subscene, query_object)
        fig.savefig(os.path.join(room_folder, f"{i}.png"))
        plt.close(fig)

    # i = 6
    # subscene, query_object = scene.permute()[i]
    # program = generate_most_restrictive_program(subscene, query_object)
    # if verify_program_validity(program, subscene, query_object):
    #     print(f"verified: {i}")
    # program.evaluate(subscene, query_object)
    # fig = program.print_program(subscene, query_object)
    # fig.savefig(os.path.join(room_folder, f"{i}.png"))
    # plt.close(fig)

def main(args):
    if args.mode == 'program_extraction':
        execute_program_extraction()
    elif args.mode == 'program_execution':
        program_execution(args.index)
    else:
        print("Not a recognized mode!")

if __name__ == '__main__':
    args = parseArguments()
    main(args)