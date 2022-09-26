# Configuration initilization for global usage in the project

# Training parameters 
#  - BATCH_SIZE
#  - NUM_EPOCHS
#  - Loss function (Loss function factory necessary) 
#  - Accuracy Metric (Accuracy function factory necessary) 

# Language parameters 
#  - NUM_ANGLES -> BIN_WIDTH calculated and available 
#  - GRID_SIZE

import configparser
import os
import numpy as np

def get_config() -> dict:
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
    toReturn = dict()
    for section in config.sections():
        toReturn[section] = dict()
        for key in config[section]:
            toReturn[section][key] = config[section][key]
    # Init config colors 
    COLORS = {
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
    toReturn['Colors'] = COLORS
    toReturn['Language']['bin_width'] = (2 * np.pi) / float(toReturn['Language']['num_angles'])
    return toReturn