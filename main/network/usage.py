from main.common.language import ProgramTree, verify_program
from main.common.utils import vectorize_scene
from main.network.utils import optimizer_factory
from main.program_extraction.dataset import get_dataloader

import torch
from tqdm import tqdm
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter

def get_network_feedback(model, dataset, parent_folder, writer, tag, device):
    print("Getting network feedback")
    indices = np.array(dataset.indices)
    np.random.shuffle(indices)
    program_dataset = dataset.dataset

    folders_to_write = [os.path.join(parent_folder, str(i)) for i in range(5)]
    for i, (folder_to_write, idx) in enumerate(zip(folders_to_write, indices[:5])):
        base_tag = tag + f"/{i}"
        (scene, query_object), _ = program_dataset[idx]
        if not os.path.exists(folder_to_write):
            os.mkdir(folder_to_write)
        inferred_tokens = infer_program(model, scene, query_object, device)
        # Is program valid 
        program = ProgramTree()
        if verify_program(inferred_tokens, len(scene.objects)):
            info_string = "without guarantee"
            f = open(os.path.join(folder_to_write, "info.txt"), "w")
            f.write(info_string)
            f.close()

            writer.add_text(base_tag + "/info", info_string)

            program.from_tokens(inferred_tokens)
            program.print_program(
                scene, query_object, folder_to_write,
                writer = writer, base_tag = base_tag, display_on_tensorboard = True
            )
        else: 
            info_string = "with guarantee\n" + "previously inferred program:\n"
            info_string += str(inferred_tokens['structure']) + "\n"
            info_string += str(inferred_tokens['constraints']) + "\n"
            f = open(os.path.join(folder_to_write, "info.txt"), "w")
            f.write(info_string)
            f.close()

            writer.add_text(base_tag + "/info", info_string)

            inferred_tokens = infer_program(model, 
                scene, query_object, device, 
                guarantee_program=True
            )
            if not verify_program(inferred_tokens, len(scene.objects)):
                 # sanity check 
                print("Inferred program with guarantee is not correct, You wrote something wrong!")
            program.from_tokens(inferred_tokens)
            program.evaluate(scene, query_object)
            program.print_program(
                scene, query_object, folder_to_write,
                writer = writer, base_tag = base_tag, display_on_tensorboard = True
            )

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

    num_training_examples = len(train_dataloader)
    writer = SummaryWriter('runs/toy_test')

    for epoch in range(network_config['Training']['epochs']):
        epoch_folder = os.path.join(parent_folder, "epoch_" + str(epoch))
        train_folder = os.path.join(epoch_folder, "train")
        validation_folder = os.path.join(epoch_folder, "validation")
        for folder in [epoch_folder, train_folder, validation_folder]:
            if not os.path.exists(folder):
                os.mkdir(folder)

        epoch_loss = 0
        epoch_accuracies = []
        epoch_structure_accuracies = []
        epoch_type_accuracies = []
        epoch_object_accuracies = []
        epoch_direction_accuracies = []
        model.train()
        print("Training epoch {}".format(epoch))
        for i, vals in enumerate(tqdm(train_dataloader)):
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
                direction_accuracy,
                total_accuracy
            ) = model.accuracy_fnc(
                structure_preds, tgt, 
                constraint_preds, tgt_c, tgt_c_padding_mask
            )

            n_iter = epoch * num_training_examples + i
            writer.add_scalar('Loss/train', loss.item(), n_iter)
            writer.add_scalar('Accuracy/train/total', total_accuracy, n_iter)
            writer.add_scalar('Accuracy/train/structure', structure_accuracy, n_iter)
            writer.add_scalar('Accuracy/train/type', type_accuracy, n_iter)
            writer.add_scalar('Accuracy/train/object', object_accuracy, n_iter)
            writer.add_scalar('Accuracy/train/direction', direction_accuracy, n_iter)

            epoch_loss += loss.item()
            epoch_accuracies.append(total_accuracy)
            epoch_structure_accuracies.append(structure_accuracy)
            epoch_type_accuracies.append(type_accuracy)
            epoch_object_accuracies.append(object_accuracy)
            epoch_direction_accuracies.append(direction_accuracy)

        print("Epoch: {}, Train Loss: {}".format(epoch, epoch_loss / num_training_examples))
        print("Epoch: {}, Train Accuracy: {}".format(epoch, np.mean(epoch_accuracies)))
        print("Epoch: {}, Train Structure Accuracy: {}".format(epoch, np.mean(epoch_structure_accuracies)))
        print("Epoch: {}, Train Type Accuracy: {}".format(epoch, np.mean(epoch_type_accuracies)))
        print("Epoch: {}, Train Object Accuracy: {}".format(epoch, np.mean(epoch_object_accuracies)))
        print("Epoch: {}, Train Direction Accuracy: {}".format(epoch, np.mean(epoch_direction_accuracies)))

        train_tag = f"feedback/epoch_{epoch}/train"
        get_network_feedback(model, train_dataset, train_folder, writer, train_tag, device)

        evaluate_network(model, validation_dataloader, network_config, "validation", writer, epoch)

        val_tag = f"feedback/epoch_{epoch}/validation"
        get_network_feedback(model, validation_dataset, validation_folder, writer, val_tag, device)

    test_folder = os.path.join(parent_folder, "test")    
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)   
    evaluate_network(model, test_dataloader, network_config, "test", writer, 0)

    test_tag = "feedback/test"
    get_network_feedback(model, test_dataset, test_folder, writer, test_tag, device)

def evaluate_network(
        model, test_dataloader, network_config, evaluation_type,
        writer, epoch
    ):
    device = network_config['device']
    model.eval()
    with torch.no_grad():
        num_test_examples = len(test_dataloader)

        loss_sum = 0
        accuracies = []
        structure_accuracies = []
        type_accuracies = []
        object_accuracies = []
        direction_accuracies = []
        for i, vals in enumerate(tqdm(test_dataloader)):
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
                direction_accuracy,
                total_accuracy
            ) = model.accuracy_fnc(
                structure_preds, tgt, 
                constraint_preds, tgt_c, tgt_c_padding_mask
            )

            n_iter = epoch * num_test_examples + i
            writer.add_scalar(f'Loss/{evaluation_type}', loss.item(), n_iter)
            writer.add_scalar(f'Accuracy/{evaluation_type}/total', total_accuracy, n_iter)
            writer.add_scalar(f'Accuracy/{evaluation_type}/structure', structure_accuracy, n_iter)
            writer.add_scalar(f'Accuracy/{evaluation_type}/type', type_accuracy, n_iter)
            writer.add_scalar(f'Accuracy/{evaluation_type}/object', object_accuracy, n_iter)
            writer.add_scalar(f'Accuracy/{evaluation_type}/direction', direction_accuracy, n_iter)

            loss_sum += loss.item()
            accuracies.append(total_accuracy)
            structure_accuracies.append(structure_accuracy)
            type_accuracies.append(type_accuracy)
            object_accuracies.append(object_accuracy)
            direction_accuracies.append(direction_accuracy)  

        print("{} Loss: {}".format(evaluation_type, loss_sum / num_test_examples))
        print("{} Accuracy: {}".format(evaluation_type, np.mean(accuracies)))
        print("{} Structure Accuracy: {}".format(evaluation_type, np.mean(structure_accuracies)))
        print("{} Type Accuracy: {}".format(evaluation_type, np.mean(type_accuracies)))
        print("{} Object Accuracy: {}".format(evaluation_type, np.mean(object_accuracies)))
        print("{} Direction Accuracy: {}".format(evaluation_type, np.mean(direction_accuracies)))