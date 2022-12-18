import os
import numpy as np
import yaml
from yaml import Loader

def generate_idx_map(list_of_vals):
    return {val : idx  for idx, val in enumerate(list_of_vals)}

def load_config(filepath):
    with open(filepath, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

data_filepath = "/home/achang57/SceneSynth/data"
# data_filepath = "/Users/adrianchang/CS/research/SceneSynth/data"
image_filepath = os.path.join(data_filepath, "images")
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
max_attach_distance = 0.15
max_visibility_distance = 4
num_angles = 4
grid_size = 256
bin_width = (2 * np.pi) / num_angles

# Language stuff 
structure_vocab = np.array(['c', 'or', 'and', '<sos>', '<eos>', '<pad>'])
structure_vocab_map = generate_idx_map(structure_vocab)
constraint_types = np.array(['attach', 'reachable_by_arm', 'align', 'face', '<sos>'])
constraint_types_map = generate_idx_map(constraint_types)
direction_types = np.array(['right', 'up', 'left', 'down', 'null'])
direction_types_map = generate_idx_map(direction_types)

object_types = np.array(['wall', 'bed', 'wardrobe', 'nightstand', 'desk', 'chair'])
object_types_map = generate_idx_map(object_types)