import os
import torch
import configparser
import os

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
    toReturn['Architecture']['loss'] = config.get('Architecture', 'loss')

    # Training 
    toReturn['Training'] = dict()
    toReturn['Training']['epochs'] = config.getint('Training', 'epochs')
    toReturn['Training']['batch_size'] = config.getint('Training', 'batch_size')
    toReturn['Training']['lr'] = config.getfloat('Training', 'lr')
    toReturn['Training']['optimizer'] = config.get('Training', 'optimizer')
    return toReturn