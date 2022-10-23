from main.common.language import ProgramTree, verify_program
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

    for folder_to_write, idx in zip(folders_to_write, indices[:10]):
        (scene, query_object), _ = program_dataset[idx]
        if not os.path.exists(folder_to_write):
            os.mkdir(folder_to_write)
        inferred_tokens = infer_program(model, scene, query_object, device)
        # Is program valid 
        program = ProgramTree()
        if verify_program(inferred_tokens, len(scene.objects)):
            program.from_tokens(inferred_tokens)
            program.print_program(scene, query_object, parent_folder)
        else:
            print("Invalid inferred program, running with guarantee") # Need debugging here 
            inferred_tokens = infer_program(model, 
                scene, query_object, device, 
                guarantee_program=True
            )
            verify_program(inferred_tokens, len(scene.objects)) # sanity check 
            program.from_tokens(inferred_tokens)
            program.print_program(scene, query_object, parent_folder)

def infer_program(model, scene, query_object, device, guarantee_program=False):
    model.eval()
    with torch.no_grad():
        scene_vector = np.expand_dims(
            vectorize_scene(scene, query_object),
            axis = 1
        )
        scene_vector = torch.tensor(scene_vector).to(device)
        structure, constraints = model.infer(
            scene_vector, device, 
            guarantee_program = guarantee_program
        )
        tokens = {
            'structure' : structure,
            'constraints' : constraints
        }

        return tokens
            
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
        epoch_folder = os.path.join(parent_folder, "epoch_" + str(epoch))
        train_folder = os.path.join(epoch_folder, "train")
        validation_folder = os.path.join(epoch_folder, "validation")
        for folder in [epoch_folder, train_folder, validation_folder]:
            if not os.path.exists(folder):
                os.mkdir(folder)

        epoch_loss = 0
        epoch_structure_accuracies = []
        epoch_type_accuracies = []
        epoch_object_accuracies = []
        epoch_direction_accuracies = []
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

            (
                structure_accuracy,
                type_accuracy,
                object_accuracy,
                direction_accuracy
            ) = model.accuracy_fnc(
                structure_preds, tgt, 
                constraint_preds, tgt_c, tgt_c_padding_mask
            )

            epoch_structure_accuracies.append(structure_accuracy)
            epoch_type_accuracies.append(type_accuracy)
            epoch_object_accuracies.append(object_accuracy)
            epoch_direction_accuracies.append(direction_accuracy)
            
            epoch_loss += loss.item()

        num_training_examples = len(train_dataloader)
        print("Epoch: {}, Train Loss: {}".format(epoch, epoch_loss / num_training_examples))
        print("Epoch: {}, Train Structure Accuracy: {}".format(epoch, np.mean(epoch_structure_accuracies)))
        print("Epoch: {}, Train Type Accuracy: {}".format(epoch, np.mean(epoch_type_accuracies)))
        print("Epoch: {}, Train Object Accuracy: {}".format(epoch, np.mean(epoch_object_accuracies)))
        print("Epoch: {}, Train Direction Accuracy: {}".format(epoch, np.mean(epoch_direction_accuracies)))
        
        get_network_feedback(model, train_dataset, train_folder, device)
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
        structure_accuracies = []
        type_accuracies = []
        object_accuracies = []
        direction_accuracies = []
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
            (
                structure_accuracy,
                type_accuracy,
                object_accuracy,
                direction_accuracy
            ) = model.accuracy_fnc(
                structure_preds, tgt, 
                constraint_preds, tgt_c, tgt_c_padding_mask
            )
            
            structure_accuracies.append(structure_accuracy)
            type_accuracies.append(type_accuracy)
            object_accuracies.append(object_accuracy)
            direction_accuracies.append(direction_accuracy)
            loss_sum += loss.item()

        num_test_examples = len(test_dataloader)
        print("{} Loss: {}".format(evaluation_type, loss_sum / num_test_examples))
        print("{} Structure Accuracy: {}".format(evaluation_type, np.mean(structure_accuracies)))
        print("{} Type Accuracy: {}".format(evaluation_type, np.mean(type_accuracies)))
        print("{} Object Accuracy: {}".format(evaluation_type, np.mean(object_accuracies)))
        print("{} Direction Accuracy: {}".format(evaluation_type, np.mean(direction_accuracies)))