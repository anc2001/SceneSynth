from main.program_extraction import get_dataloaders
import torch

def train_network():
    train, test, val = get_dataloaders()