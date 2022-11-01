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

def get_network_feedback(model, dataset, base_tag, device, num_examples = 10, wandb = False):
    print("Getting network feedback")
    indices = np.array(dataset.indices)
    np.random.shuffle(indices)
    program_dataset = dataset.dataset

    columns = ["guarantee_required", "previous_structure", "previous_constraints"]
    data = []

    tags = [base_tag + f"/example_{i}" for i in range(num_examples)]
    for tag, idx in zip(tags, indices[:num_examples]):
        (scene, query_object), ground_truth_program_tokens = program_dataset[idx]
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
        fig = program.print_program(scene, query_object)

        if wandb:
            wandb.log({tag + "_inferred": fig})
        else:
            pass

        program = ProgramTree()
        program.from_tokens(ground_truth_program_tokens)
        program.evaluate(scene, query_object)
        fig = program.print_program(scene, query_object)  
    
        if wandb:
            wandb.log({tag + "_ground_truth": fig})
        else:
            pass
    
    if wandb:
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

def iterate_through_data(model, dataloader, device, type, wandb = False, optimizer=None):
    total_log = {
            "loss" : [],
            "accuracy" : [],
            "structure_accuracy" : [],
            "type_accuracy" : [],
            "object_accuracy" : [],
            "direction_accuracy" : []
        }
    for vals in tqdm(dataloader):
        src, src_padding_mask, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask, tgt_c_padding_mask_types = vals
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
            tgt_c, tgt_c_padding_mask, tgt_c_padding_mask_types
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
            structure_preds, constraint_preds,
            tgt, tgt_c, tgt_c_padding_mask, tgt_c_padding_mask_types
        )

        log = {
            "loss" : loss.item(),
            "accuracy" : total_accuracy,
            "structure_accuracy" : structure_accuracy,
            "type_accuracy" : type_accuracy,
            "object_accuracy" : object_accuracy,
            "direction_accuracy" : direction_accuracy
        }
        if wandb and type == "train":
            wandb.log({"train" : log})
        
        for key in log.keys():
            total_log[key].append(log[key])
    
    return {key : np.mean(value) for (key, value) in total_log.items()}

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--wandb', type=bool, default=False, 
        help="enables wandb run")
    parser.add_argument('--config', type=str, default = os.path.join(os.path.dirname(__file__), 'main/config/config.yaml'), 
        help="config to use for run")
    parser.add_argument('--checkpoint', type=str, default="", 
        help="checkpoint to run from")
    args = parser.parse_args()
    return args

def main(args):
    config = load_config(args.config)
    if args.wandb:
        wandb.init(
            project = config['project'],
            name = config['name'],
            config = config
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = get_dataset()
    train_size = int(config['training']['train_split'] * len(dataset))
    validation_size = int(config['training']['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - validation_size
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, test_size, validation_size]
    )
    train_dataloader = get_dataloader(train_dataset, config['training']['batch_size'])
    test_dataloader = get_dataloader(test_dataset, config['training']['batch_size'])
    val_dataloader = get_dataloader(val_dataset, config['training']['batch_size'])

    if len(args.checkpoint):
        pass
    else:
        model = ModelCore(
            d_model = config['architecture']['d_model'],
            nhead = config['architecture']['nhead'],
            num_layers= config['architecture']['num_layers'],
            max_num_objects = 50,
            loss_func=loss_factory(config['architecture']['loss'])
        )
    model.to(device)

    optimizer = optimizer_factory(
        model, 
        config['training']['optimizer'], 
        config['training']['lr']
    )

    # Training and validation loop 
    if args.wandb:
        wandb.watch(model)
    
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        
        model.train()
        log = iterate_through_data(
            model, train_dataloader, device, "train", 
            optimizer = optimizer, wandb = args.wandb
        )
        if not args.wandb:
            print(log)
        
        get_network_feedback(
            model, train_dataset, 
            f"examples/train/epoch_{epoch}",
            device
        )

        model.eval()
        with torch.inference_mode():
            log = iterate_through_data(
                model, val_dataloader, device, "val", 
                wandb = args.wandb
            )
            if args.wandb:
                wandb.log({"val" : log})
            else:
                print(log)

            get_network_feedback(
                model, val_dataset,
                f"examples/val/epoch_{epoch}",
                device
            )
    
    iterate_through_data(model, test_dataloader, device, "test", wandb = args.wandb)
    model_save = os.path.join(data_filepath, "model.pt")
    save_model(model, model_save)

def overfit_to_one(args):
    config = load_config(args.config)
    if args.wandb:
        wandb.init(
            project = config['project'],
            name = config['name'],
            config = config
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset()
    index = 182
    # for i in range(500):
    #     (scene, query_object), program_tokens = dataset[i]
    #     print(f"{i}: {len(scene.objects)} {program_tokens['structure']}")
    
    # (scene, query_object), program_tokens = dataset[index]

    # program = ProgramTree()
    # program.from_tokens(program_tokens)
    # program.evaluate(scene, query_object)
    # fig = program.print_program(scene, query_object)
    # fig.savefig("/Users/adrianchang/CS/research/SceneSynth/tree.png")
    
    single_point_dataset = torch.utils.data.Subset(dataset, [index])
    single_point_dataloader = get_dataloader(single_point_dataset, 1)

    model = ModelCore(
        d_model = config['architecture']['d_model'],
        nhead = config['architecture']['nhead'],
        num_layers= config['architecture']['num_layers'],
        max_num_objects = 50,
        max_program_length= 30,
        loss_func=loss_factory(config['architecture']['loss'])
    )
    model.to(device)

    optimizer = optimizer_factory(
        model, 
        config['training']['optimizer'], 
        config['training']['lr']
    )

    epochs = 200
    for epoch in range(epochs):
        model.train()
        log = iterate_through_data(
            model, single_point_dataloader, device, "train", 
            optimizer = optimizer, wandb = args.wandb
        )
        if not args.wandb:
            print(log)
        
        get_network_feedback(
            model, single_point_dataset, 
            f"examples/train/epoch_{epoch}",
            device,
            num_examples = 1
        )

if __name__ == '__main__':
    args = parseArguments()
    # main(args)
    overfit_to_one(args)