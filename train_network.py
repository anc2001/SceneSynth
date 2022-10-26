from main.network import ModelCore, \
    optimizer_factory, loss_factory, \
    save_model, load_model
from main.program_extraction.dataset import \
    get_dataset, get_dataloader

from main.config import load_config, data_filepath
from main.common.language import ProgramTree, verify_program
from main.common.utils import vectorize_scene, clear_folder
from main.program_extraction.dataset import get_dataloader

import torch
from tqdm import tqdm
import numpy as np
import os
import wandb
from argparse import ArgumentParser

def get_network_feedback(model, dataset, base_tag, device):
    print("Getting network feedback")
    indices = np.array(dataset.indices)
    np.random.shuffle(indices)
    program_dataset = dataset.dataset

    columns = ["guarantee_required", "previous_structure", "previous_constraints", "program"]
    data = []
    tags = [base_tag + f"/example_{i}" for i in range(5)]
    for tag, idx in zip(tags, indices[:5]):
        (scene, query_object), _ = program_dataset[idx]
        data_entry = []
        inferred_tokens = infer_program(model, scene, query_object, device)
        # Is program valid 
        if verify_program(inferred_tokens, len(scene.objects)):
            data_entry.append("no")
            data_entry.append("")
            data_entry.append("")
        else: 
            data_entry.append("yes")
            data_entry.append(str(inferred_tokens['structure']))
            data_entry.append(str(inferred_tokens['constraints']))

            inferred_tokens = infer_program(model, 
                scene, query_object, device, 
                guarantee_program=True
            )
        
        # sanity check 
        if not verify_program(inferred_tokens, len(scene.objects)):
            print("Inferred program with guarantee is not correct, You wrote something wrong!")
        
        program = ProgramTree()
        program.from_tokens(inferred_tokens)
        program.evaluate(scene, query_object)
        program_string, images, image_names = program.print_program(
            scene, query_object
        )
        data_entry.append(program_string)
        data.append(data_entry)

        wandb_images = [wandb.Image(image, caption=name) for name, image in zip(image_names, images)]
        wandb.log({tag : wandb_images})    
    
    table = wandb.Table(data=data, columns=columns)
    wandb.log({base_tag : table})

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

def iterate_through_data(model, dataloader, device, type, optimizer=None):
    for vals in tqdm(dataloader):
        src, src_padding_mask, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask = vals
        structure_preds, constraint_preds = model(
            src, src_padding_mask, 
            tgt, tgt_padding_mask,
            tgt_c, tgt_c_padding_mask,
            device
        )

        if type == "train":
            optimizer.zero_grad()
        # Compute Loss
        loss = model.loss(
            structure_preds, 
            constraint_preds, 
            tgt, tgt_padding_mask, 
            tgt_c, tgt_c_padding_mask
        )

        if type == "train":
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

        log = {
            "loss" : loss.item(),
            "accuracy" : total_accuracy,
            "structure_accuracy" : structure_accuracy,
            "type_accuracy" : type_accuracy,
            "object_accuracy" : object_accuracy,
            "direction_accuracy" : direction_accuracy
        }
        wandb.log({type : log})

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default = os.path.join(os.path.dirname(__file__), 'main/config/config.yaml'), 
        help="config to use for run")
    parser.add_argument('--checkpoint', type=str, default="", 
        help="checkpoint to run from")
    args = parser.parse_args()
    return args

def main(args):
    config = load_config(args.config)
    wandb.init(
        project = config['project'],
        name = config['name'],
        config = config
    )

    parent_folder = os.path.join(data_filepath, "network_feedback")
    if os.path.exists(parent_folder):
        clear_folder(parent_folder)
    else:
        os.mkdir(parent_folder)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = get_dataset()
    train_size = int(wandb.config.training['train_split'] * len(dataset))
    validation_size = int(wandb.config.training['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size, validation_size]
    )
    train_dataloader = get_dataloader(train_dataset, wandb.config.training['batch_size'])
    test_dataloader = get_dataloader(test_dataset, wandb.config.training['batch_size'])
    val_dataloader = get_dataloader(val_dataset, wandb.config.training['batch_size'])

    if len(args.checkpoint):
        pass
    else:
        model = ModelCore(
            d_model = wandb.config.architecture['d_model'],
            nhead = wandb.config.architecture['nhead'],
            num_layers= wandb.config.architecture['num_layers'],
            loss_func=loss_factory(wandb.config.architecture['loss'])
        )
    model.to(device)

    optimizer = optimizer_factory(
        model, 
        wandb.config.training['optimizer'], 
        wandb.config.training['lr']
    )

    # Training and validation loop 
    wandb.watch(model)
    epochs = wandb.config.training['epochs']
    for epoch in range(epochs):
        base_tag = f"epoch_{epoch}"

        iterate_through_data(model, train_dataloader, device, "train", optimizer)
        get_network_feedback(
            model, train_dataset, 
            base_tag + "/train/examples",
            device
        )

        model.eval()
        with torch.no_grad():
            iterate_through_data(model, val_dataloader, device, "val")
            get_network_feedback(
                model, val_dataset,
                base_tag + "/val/examples",
                device
            )
    
    iterate_through_data(model, test_dataloader, device, "test")
    model_save = os.path.join(parent_folder, "model.pt")
    save_model(model, model_save)


if __name__ == '__main__':
    args = parseArguments()
    main(args)