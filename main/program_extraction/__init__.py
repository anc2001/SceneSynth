from main.program_extraction.dataset import ProgramDataset

import torch
from torch.utils.data import DataLoader
from main.config import get_network_config

def get_dataloaders():
    # Return the dataloader
    dataset = ProgramDataset()

    train_size = int(0.8 * len(dataset))
    validation_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size, validation_size]
    )

    network_config = get_network_config()

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size = network_config['Training']['batch_size'], 
        shuffle = True, 
        collate_fn = dataset.collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size = network_config['Training']['batch_size'],
        shuffle = True,
        collate_fn = dataset.collate_fn
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size = network_config['Training']['batch_size'],
        shuffle = True,
        collate_fn = dataset.collate_fn
    )

    return train_dataloader, test_dataloader, validation_dataloader