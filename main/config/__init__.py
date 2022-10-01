import configparser
import os
import numpy as np
import torch

def get_network_config() -> dict:
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    toReturn = dict()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    toReturn['device'] = device
    
    # Architecture 
    toReturn['Architecture'] = dict()
    toReturn['Architecture']['d_model'] = config.getint('Architecture', 'd_model')
    toReturn['Architecture']['nhead'] = config.getint('Architecture', 'nhead')
    toReturn['Architecture']['num_layers'] = config.getint('Architecture', 'num_layers')

    # Training 
    toReturn['Training'] = dict()
    toReturn['Training']['epochs'] = config.getint('Training', 'epochs')
    toReturn['Training']['batch_size'] = config.getint('Training', 'batch_size')
    toReturn['Training']['lr'] = config.getfloat('Training', 'lr')
    return toReturn

data_filepath = "/Users/adrianchang/CS/research/SceneSynth/data/"
colors = {
        'outside' : np.array([222, 222, 222]) / 256,
        'inside' : np.array([169, 169, 169]) / 256,
        'wall': np.array([0, 0, 0]) / 256,
        'bed' : np.array([38, 70, 83]) / 256, # Dark navy blue 
        'chair' : np.array([33, 158, 188]) / 256, # Light blue 
        'wardrobe' : np.array([42, 157 , 143]) / 256, # Aqua greenish thing 
        'nightstand' : np.array([189, 189 , 38]) / 256, # bright yellow
        'desk' : np.array([244, 162 , 97]) / 256, # Light orange
        'directions' :  np.array([[255,0,0], [0,255,0], [0,0,255], [255,255,0]]) / 255
    }
max_allowed_sideways_reach = 0.6
num_angles = 4
grid_size = 256
bin_width = (2 * np.pi) / num_angles

# Language stuff 
structure_vocab = ['c', 'or', 'and', '<sos>', '<eos>', '<pad>']
constraint_types = ['attach', 'reachable_by_arm', 'align', 'face']
constraint_types_map = {type : idx  for idx, type in enumerate(constraint_types)}
direction_types = ['right', 'up', 'left', 'down', '<pad>']
direction_types_map = {type : idx  for idx, type in enumerate(direction_types)}