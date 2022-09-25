from main.program_extraction.dataset import get_dataloaders


def train_network():
    train, test, val = get_dataloaders()