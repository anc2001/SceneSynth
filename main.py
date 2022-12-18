from main.common.utils import clear_folder, read_data
from main.config import image_filepath, data_filepath
from main.data_processing import generate_most_restrictive_program, \
    execute_program_extraction, verify_program_validity, \
    execute_scene_filtering, generate_subscenes, \
    get_scene_list, get_scene_query_list
from main.common.language import ProgramTree

from main.common.mesh_to_mask import render, get_triangles

import matplotlib.image as img
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np 

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, 
        help="mode to run")
    parser.add_argument('-i', '--index', type=int, default=0,
        help="program_execution: room index to run program_execution on")
    parser.add_argument('-n', '--name', type=str, default="",
        help="scenes file name")
    args = parser.parse_args()
    return args

def print_rooms(pickle_name):
    if not os.path.exists(image_filepath):
        os.mkdir(image_filepath)
    
    room_folder = os.path.join(image_filepath, "room_images")
    if os.path.exists(room_folder):
        clear_folder(room_folder)
    else:
        os.mkdir(room_folder)
    
    scene_list = get_scene_list(pickle_name)
    for i, scene in enumerate(tqdm(scene_list)):
        scene_image = scene.convert_to_image()
        img.imsave(os.path.join(room_folder, f"{i}-{scene.id}.png"), scene_image)

def visualize_programs(pickle_name, in_order=True):
    if not os.path.exists(image_filepath):
        os.mkdir(image_filepath)
    
    folder = os.path.join(image_filepath, "program_visualizations")
    if os.path.exists(folder):
        clear_folder(folder)
    else:
        os.mkdir(folder)

    program_data = read_data(os.path.join(data_filepath, pickle_name))
    xs = program_data['xs']
    ys = program_data['ys']
    
    to_iterate = list(zip(xs, ys))
    if not in_order:
        np.random.shuffle(to_iterate)
    for i, item in enumerate(tqdm(to_iterate)):
        (scene, query_object) = item[0]
        program_tokens = item[1]
        program = ProgramTree()
        program.from_tokens(program_tokens)
        program.evaluate(scene, query_object)
        fig = program.print_program(scene, query_object)
        fig.savefig(os.path.join(folder, f"{i}-{scene.id}.png"))
        plt.close(fig)
        break

def main(args):
    if args.mode == 'program_extraction':
        execute_program_extraction(args.name)
    elif args.mode == 'filter_scenes':
        execute_scene_filtering(args.name)
    elif args.mode == 'print_scenes':
        print_rooms(args.name)
    elif args.mode == 'print_programs':
        visualize_programs(args.name)
    elif args.mode == 'subsample':
        generate_subscenes(args.name)
    else:
        print("Not a recognized mode!")

if __name__ == '__main__':
    args = parseArguments()
    main(args)