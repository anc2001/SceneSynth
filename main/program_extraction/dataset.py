from main.config import structure_vocab_map
from main.program_extraction.data_processing import read_program_data
from main.common.utils import vectorize_scene

import torch
from torch.utils.data import Dataset

import numpy as np
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_dataset():
    data = read_program_data()
    dataset = ProgramDataset(data)
    return dataset

def get_dataloader(dataset, batch_size):
    # Return the dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        collate_fn = collate_fn
    )

    return dataloader

def collate_fn(samples):
    # Collate the samples into batches
    # return object sequence, object sequence padding mask, structure sequence, constraint sequence 
    objects_max_length = 0
    structure_max_length = 0
    constraints_max_length = 0
    for sample in samples:
        scene, _ = sample[0]
        num_objects = 1 + len(scene.objects)
        program_tokens = sample[1]
        objects_max_length = max(num_objects, objects_max_length)
        structure_max_length = max(len(program_tokens['structure']), structure_max_length)
        constraints_max_length = max(len(program_tokens['constraints']), constraints_max_length)
    
    structure_max_length += 2

    object_representation_length = 7
    constraint_representation_length = 4

    src = np.array([])
    src_padding_mask = np.array([])
    tgt = np.array([])
    tgt_padding_mask = np.array([])
    tgt_c = np.array([])
    tgt_c_padding_mask = np.array([])

    for sample in samples:
        scene, query_object = sample[0]
        objs = vectorize_scene(scene, query_object)
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
        structure_sequence = np.concatenate(
            [
                [structure_vocab_map['<sos>']],
                structure_sequence,
                [structure_vocab_map['<eos>']]
            ]
        )

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
            tgt_c_padding_mask = np.append(
                tgt_c_padding_mask,
                np.expand_dims(constraints_padding_mask, axis =1),
                axis = 1
            )
        else:
            tgt_c_padding_mask = np.expand_dims(constraints_padding_mask, axis = 1)

        if len(tgt_c):
            tgt_c = np.append(
                tgt_c, 
                np.expand_dims(constraints, axis = 1), 
                axis = 1
            )
        else:
            tgt_c = np.expand_dims(constraints, axis = 1)
    
    tgt_padding_mask = np.transpose(
        tgt == structure_vocab_map['<pad>']
    )
    return (
        torch.tensor(src).float().to(device),
        torch.tensor(src_padding_mask).bool().to(device), 
        torch.tensor(tgt).float().to(device),
        torch.tensor(tgt_padding_mask).bool().to(device), 
        torch.tensor(tgt_c).float().to(device),
        torch.tensor(tgt_c_padding_mask).bool().to(device),
    )

class ProgramDataset(Dataset):
    def __init__(self, data):
        # Load data
        self.x = data['xs']
        self.y = data['ys']
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.x)

    def __getitem__(self, idx: int):
        # Return the item at the given index
        return self.x[idx], self.y[idx]
