from main.program_extraction.data_processing import read_program_data
from main.config import get_network_config, structure_vocab

import torch
from torch.utils.data import Dataset

import numpy as np
import torch
from torch.utils.data import DataLoader

network_config = get_network_config()
device = network_config['device']
structure_vocab_map = {word : idx  for idx, word in enumerate(structure_vocab)}

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
        data = read_program_data()
        self.x = data['xs']
        self.y = data['ys']

    def __len__(self):
        # Return the length of the dataset
        return len(self.x)

    def __getitem__(self, idx: int):
        # Return the item at the given index
        return self.x[idx], self.y[idx]
    
    def collate_fn(self, samples):
        # Collate the samples into batches
        # return object sequence, object sequence padding mask, structure sequence, constraint sequence 
        objects_max_length = 0
        structure_max_length = 0
        constraints_max_length = 0
        for sample in samples:
            objects_max_length = max(len(sample[0]), objects_max_length)
            structure_max_length = max(len(sample[1]['structure']), structure_max_length)
            constraints_max_length = max(len(sample[1]['constraints']), constraints_max_length)
        
        object_representation_length = len(samples[0][0][0])
        constraint_representation_length = 4

        src = np.array([])
        src_padding_mask = np.array([])
        tgt = np.array([])
        tgt_padding_mask = np.array([])
        tgt_c = np.array([])
        tgt_c_padding_mask = np.array([])

        for sample in samples:
            objs = sample[0]
            objs_padding_mask = np.append(
                np.zeros(len(objs)), 
                np.ones(objects_max_length - len(objs))
            )
            objs = np.append(
                objs, 
                np.zeros([objects_max_length - len(objs), object_representation_length]),
                axis = 0
            )

            if len(src_padding_mask):
                src_padding_mask = np.append(
                    src_padding_mask, 
                    np.expand_dims(objs_padding_mask, axis = 0), 
                    axis = 0
                )
            else:
                src_padding_mask = np.expand_dims(objs_padding_mask, axis = 0)
            
            if len(src):
                src = np.append(
                    src, 
                    np.expand_dims(objs, axis = 1), 
                    axis = 1
                )
            else:
                src = np.expand_dims(objs, axis = 1)
            
            program = sample[1]

            structure_sequence = program['structure']
            structure_sequence = np.array([structure_vocab_map[word] for word in structure_sequence])

            structure_sequence = np.append(
                structure_sequence, 
                np.full(
                    structure_max_length - len(structure_sequence), 
                    structure_vocab_map['<pad>']
                )
            )

            if len(tgt):
                tgt = np.append(
                    tgt,
                    np.expand_dims(structure_sequence, axis = 1),
                    axis = 1
                )
            else:
                tgt = np.expand_dims(structure_sequence, axis = 1)

            constraints = program['constraints']

            constraints_padding_mask = np.append(
                np.zeros(len(constraints)),
                np.ones(constraints_max_length - len(constraints))
            )

            constraints = np.append(
                constraints,
                np.zeros([constraints_max_length - len(constraints), constraint_representation_length]),
                axis = 0
            )

            if len(tgt_c_padding_mask):
                tgt_c_padding_mask= np.append(
                    tgt_c_padding_mask, 
                    np.expand_dims(constraints_padding_mask, axis = 0), 
                    axis = 0
                )
            else:
                tgt_c_padding_mask = np.expand_dims(constraints_padding_mask, axis = 0)

            if len(tgt_c):
                tgt_c = np.append(
                    tgt_c, 
                    np.expand_dims(constraints, axis = 1), 
                    axis = 1
                )
            else:
                tgt_c = np.expand_dims(constraints, axis = 1)
        
        tgt_padding_mask = np.transpose(tgt == structure_vocab_map['<pad>'])

        yield (
            torch.tensor(src).float().to(device),
            torch.tensor(src_padding_mask).bool().to(device), 
            torch.tensor(tgt).float().to(device),
            torch.tensor(tgt_padding_mask).bool().to(device),
            torch.tensor(tgt_c).float().to(device),
            torch.tensor(tgt_c_padding_mask).bool().to(device),
        )