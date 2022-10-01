from main.program_extraction.data_processing import read_program_data
from main.config import get_network_config

import torch
from torch.utils.data import Dataset

import numpy as np
import torch
from torch.utils.data import DataLoader

network_config = get_network_config()
device = network_config['device']

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

class ProgramDataset(Dataset):
    def __init__(self):
        # Load data
        objects, programs = read_program_data()
        self.x = objects
        self.y = programs

    def __len__(self):
        # Return the length of the dataset
        return len(self.x)

    def __getitem__(self, idx: int):
        # Return the item at the given index
        return self.x[idx], self.y[idx]
    
    def collate_fn(self, samples):
        # Collate the samples into batches
        # return object sequence, object sequence padding mask, constraint sequence, constraint sequence padding mask
        objects_max_length = 0
        constraints_max_length = 0
        object_representation_length = len(samples[0][0][0])
        constraint_representation_length = len(samples[0][1][0])
        for sample in samples:
            objects_max_length = max(len(sample[0]), objects_max_length)
            constraints_max_length = max(len(sample[1]), constraints_max_length)
        
        source = np.array([])
        source_padding_mask = np.array([])
        target = np.array([])
        target_padding_mask = np.array([])
        for sample in samples:
            objs = sample[0]
            objects_padding_mask = np.append(
                np.zeros(len(objs)), 
                np.ones(objects_max_length - len(objs))
            )
            objs = np.append(
                objs, 
                np.zeros([objects_max_length - len(objs), object_representation_length]),
                axis = 0
            )

            if len(source_padding_mask):
                source_padding_mask = np.append(
                    source_padding_mask, 
                    np.expand_dims(objects_padding_mask, axis = 0), 
                    axis = 0
                )
            else:
                source_padding_mask = np.expand_dims(objects_padding_mask, axis = 0)
            
            if len(source):
                source = np.append(
                    source, 
                     np.expand_dims(objs, axis = 1), 
                    axis = 1
                )
            else:
                source = np.expand_dims(objs, axis = 1)
            
            constraints = sample[1]
            constraints_padding_mask = np.append(
                np.zeros(len(constraints)),
                np.ones(constraints_max_length - len(constraints))
            )

            constraints = np.append(
                constraints,
                np.zeros([constraints_max_length - len(constraints), constraint_representation_length]),
                axis = 0
            )

            if len(target_padding_mask):
                target_padding_mask = np.append(
                    target_padding_mask, 
                    np.expand_dims(constraints_padding_mask, axis = 0), 
                    axis = 0
                )
            else:
                target_padding_mask = np.expand_dims(constraints_padding_mask, axis = 0)

            if len(target):
                target = np.append(
                    target, 
                    np.expand_dims(constraints, axis = 1), 
                    axis = 1
                )
            else:
                target = np.expand_dims(constraints, axis = 1)
        
        return (
            torch.tensor(source).float().to(device),
            torch.tensor(source_padding_mask).bool().to(device), 
            torch.tensor(target).float().to(device),
            torch.tensor(target_padding_mask).bool().to(device),
        )