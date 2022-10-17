from main.common import get_scene_list
from main.common.utils import clear_folder
from main.config import image_filepath, get_network_config
from main.program_extraction import generate_most_restrictive_program, \
    execute_program_extraction, read_program_data, get_dataloaders, \
    verify_program_validity
from main.network import ModelCore, loss_factory, \
    train_network, test_network

import matplotlib.image as img
import os
from argparse import ArgumentParser
from tqdm import tqdm

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, 
        help="mode to run [extract_programs, train, program_execution]")
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

def train_test_model():
    train, test, val = get_dataloaders()
    network_config = get_network_config()
    device = network_config['device']
    model = ModelCore(
        d_model = network_config['Architecture']['d_model'],
        nhead = network_config['Architecture']['nhead'],
        num_layers= network_config['Architecture']['d_model'],
        loss_func=loss_factory(network_config)
    )
    model.to(device)
    train_network(model, train, val, network_config)
    test_network(model, test, network_config)

def main(args):
    if args.mode == 'extract_programs':
        execute_program_extraction()
    elif args.mode == 'train':
        train_test_model()
    elif args.mode == 'program_execution':
        program_execution(args.index)

if __name__ == '__main__':
    args = parseArguments()
    main(args)