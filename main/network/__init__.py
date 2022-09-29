from main.program_extraction.dataset import get_dataloaders
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_network():
    train, test, val = get_dataloaders()