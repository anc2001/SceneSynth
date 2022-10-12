from main.common import get_scene_list
from main.config import image_filepath, get_network_config
from main.program_extraction import generate_most_restrictive_program, \
    execute_program_extraction, read_program_data, get_dataloaders
from main.network import ModelCore, loss_factory, \
    train_network, test_network

import matplotlib.image as img
import os
from argparse import ArgumentParser

# def parseArguments():
#     parser = ArgumentParser()
#     parser.add_argument('-s', '--source', type=str, required=True, 
#         help="mode")
#     args = parser.parse_args()
#     return args

# def test(scene, query_object, program_name):
#     program = generate_most_restrictive_program(scene, query_object)
#     program.evaluate(scene, query_object)
#     program.print_program(program_name, scene, query_object)

# scene_list = get_scene_list()
# scene = scene_list[1]
# scene_image = scene.convert_to_image()
# img.imsave(os.path.join(image_filepath, "scene.png"), scene_image)
# for i, (subscene, query_object) in enumerate(scene.permute()):
#     print(i)
#     test(subscene, query_object, str(i))

# execute_program_extraction()
# read_program_data()

# def train_test_model():

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

# def main(args):
#     pass

# if __name__ == '__main__':
#     args = parseArguments()
#     main(args)