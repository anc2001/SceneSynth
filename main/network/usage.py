from main.common.language import ProgramTree
from main.common.utils import vectorize_scene
from main.network.utils import optimizer_factory
from main.program_extraction.dataset import get_dataloader

import torch
from tqdm import tqdm
import numpy as np
import os

def get_network_feedback(model, dataset, parent_folder, device):
    indices = np.array(dataset.indices)
    np.random.shuffle(indices)
    program_dataset = dataset.dataset

    folders_to_write = [os.path.join(parent_folder, str(i)) for i in range(5)]

    for folder_to_write, idx in zip(folders_to_write, indices[:5]):
        (scene, query_object), ground_truth_program_tokens = program_dataset[idx]
        if not os.path.exists(folder_to_write):
            os.mkdir(folder_to_write)
        inferred_tokens =  infer_program(model, scene, query_object, device)
        # Is program valid 
        program = ProgramTree()
        if program_valid:
            pass
        else:
            print("") # Need debugging here 
            program.from_tokens(ground_truth_program_tokens)
            program.print_program(scene, query_object, parent_folder)

def infer_program(model, scene, query_object, device):
    model.eval()
    with torch.no_grad():
        scene_vector = np.expand_dims(
            vectorize_scene(scene, query_object),
            axis = 1
        )
        scene_vector = torch.tensor(scene_vector).to(device)
        structure, constraints = model.infer(scene_vector, device)
        tokens = {
            'structure' : structure,
            'constraints' : constraints
        }

        

        return True
            
def train_test_network_with_feedback(
        model, 
        train_dataset, 
        test_dataset, 
        validation_dataset, 
        network_config,
        parent_folder
    ):
    device = network_config['device']
    optimizer = optimizer_factory(model, network_config)

    train_dataloader = get_dataloader(train_dataset)
    test_dataloader = get_dataloader(test_dataset)
    validation_dataloader = get_dataloader(validation_dataset)

    for epoch in range(network_config['Training']['epochs']):
        epoch_folder = os.path.join(parent_folder, str(epoch))
        if not os.path.exists(epoch_folder):
            os.mkdir(epoch_folder)
        
        epoch_loss = 0
        epoch_accuracies = []
        model.train()
        print("Training epoch {}".format(epoch))
        for vals in tqdm(train_dataloader):
            src, src_padding_mask, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask = vals
            structure_preds, constraint_preds = model(
                src, src_padding_mask, 
                tgt, tgt_padding_mask,
                tgt_c, tgt_c_padding_mask,
                device
            )

            optimizer.zero_grad()
            # Compute Loss
            loss = model.loss(
                structure_preds, 
                constraint_preds, 
                tgt, tgt_padding_mask, 
                tgt_c, tgt_c_padding_mask
            )
            # Backpropagation 
            loss.backward()
            # Update
            optimizer.step()

            accuracy = model.accuracy_fnc(
                structure_preds, tgt, 
                constraint_preds, tgt_c, tgt_c_padding_mask
            )
            epoch_accuracies.append(accuracy)
            epoch_loss += loss.item()

        num_training_examples = len(train_dataloader)
        print("Epoch: {}, Train Loss: {}, Train Accuracy: {}".format(epoch, epoch_loss / num_training_examples, np.mean(epoch_accuracies)))
        
        train_folder = os.path.join(epoch_folder, "train")    
        if not os.path.exists(train_folder):
            os.mkdir(train_folder)
        get_network_feedback(model, train_dataset, train_folder, device)

        validation_folder = os.path.join(epoch_folder, "validation")    
        if not os.path.exists(validation_folder):
            os.mkdir(validation_folder) 
        evaluate_network(model, validation_dataloader, network_config, "Validation")
        get_network_feedback(model, validation_dataset, validation_folder, device)

    test_folder = os.path.join(parent_folder, "validation")    
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)   
    evaluate_network(model, test_dataloader, network_config, "Test")
    get_network_feedback(model, test_dataset, test_folder, device)

def evaluate_network(model, test_dataloader, network_config, evaluation_type):
    device = network_config['device']
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        accuracies = []
        for vals in tqdm(test_dataloader):
            src, src_padding_mask, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask = vals
            structure_preds, constraint_preds = model(
                src, src_padding_mask, 
                tgt, tgt_padding_mask,
                tgt_c, tgt_c_padding_mask,
                device
            )
            loss = model.loss(
                structure_preds, 
                constraint_preds, 
                tgt, tgt_padding_mask, 
                tgt_c, tgt_c_padding_mask
            )
            accuracy = model.accuracy_fnc(
                structure_preds, tgt, 
                constraint_preds, tgt_c, tgt_c_padding_mask
            )
            accuracies.append(accuracy)
            loss_sum += loss.item()

        num_test_examples = len(test_dataloader)
        print("{} Loss: {}, {} Accuracy: {}".format(evaluation_type, loss_sum / num_test_examples, np.mean(accuracies)))
        