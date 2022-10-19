from main.network.usage import evaluate_network, \
    train_test_network_with_feedback, infer_program
from main.network.utils import loss_factory, save_model, load_model
from main.network.core import ModelCore
from main.program_extraction.dataset import get_dataset, get_dataloader
from main.config import get_network_config, data_filepath
from main.common.utils import clear_folder

import torch
import os
import numpy as np

def test_infer_program(model, dataset, folder, device):
    indices = np.array(dataset.indices)
    np.random.shuffle(indices)
    program_dataset = dataset.dataset
    for idx in indices:
        (scene, query_object), _ = program_dataset[idx]
        infer_program(model, scene, query_object, folder, device)

def do_everything():
    dataset = get_dataset()
    train_size = int(0.05 * len(dataset))
    validation_size = int(0.01 * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size, validation_size]
    )

    network_config = get_network_config()
    device = network_config['device']
    model = ModelCore(
        d_model = network_config['Architecture']['d_model'],
        nhead = network_config['Architecture']['nhead'],
        num_layers= network_config['Architecture']['d_model'],
        loss_func=loss_factory(network_config)
    )
    model.to(device)

    parent_folder = os.path.join(data_filepath, "network_feedback")
    if os.path.exists(parent_folder):
        clear_folder(parent_folder)
    else:
        os.mkdir(parent_folder)
    
    test_infer_program(model, train_dataset, parent_folder, device)
    train_test_network_with_feedback(
        model, 
        train_dataset, 
        test_dataset, 
        validation_dataset, 
        network_config, 
        parent_folder
    )