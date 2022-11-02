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
import matplotlib.pyplot as plt

def get_network_feedback(model, dataset, base_tag, device, num_examples = 5, with_wandb = False):
    print("Getting network feedback")
    indices = np.array(dataset.indices)
    np.random.shuffle(indices)
    program_dataset = dataset.dataset

    columns = ["type", "feedback"]
    data = []

    tags = [base_tag + f"/example_{i}" for i in range(num_examples)]
    for tag, idx in tqdm(zip(tags, indices[:num_examples])):
        (scene, query_object), ground_truth_program_tokens = program_dataset[idx]
        data_entry = []
        inferred_tokens = infer_program(model, scene, query_object, device)

        validity, feedback = verify_program(inferred_tokens, len(scene.objects))
        if validity:
            data_entry.append("inferred, no help")
            data_entry.append(feedback)
        else: 
            data_entry.append("inferred, help required")
            data_entry.append(feedback)

            inferred_tokens = infer_program(model,
                scene, query_object, device, 
                guarantee_program=True
            )
        data.append(data_entry)
        
        validity, feedback = verify_program(inferred_tokens, len(scene.objects))
        # sanity check 
        if not validity:
            print("Inferred program with guarantee is not correct, You wrote something wrong!")
        
        program = ProgramTree()
        program.from_tokens(inferred_tokens)
        program.evaluate(scene, query_object)
        fig = program.print_program(scene, query_object)

        if with_wandb:
            wandb.log({tag + "_inferred": fig})
            
        plt.close(fig)

        program = ProgramTree()
        program.from_tokens(ground_truth_program_tokens)
        program.evaluate(scene, query_object)
        fig = program.print_program(scene, query_object)  
    
        if with_wandb:
            wandb.log({tag + "_ground_truth": fig})
        plt.close(fig)
    
    if with_wandb:
        table = wandb.Table(data=data, columns=columns)
        wandb.log({base_tag + "/table" : table})

def infer_program(model, scene, query_object, device, guarantee_program=False):
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

def iterate_through_data(model, dataloader, device, type, with_wandb = False, optimizer=None):
    total_log = {
            "loss" : [],
            "accuracy" : [],
            "structure_accuracy" : [],
            "type_accuracy" : [],
            "object_accuracy" : [],
            "direction_accuracy" : []
        }
    
    for vals in tqdm(dataloader):
        # Extract vals from dataloader 
        (
            src, src_padding_mask, 
            tgt, tgt_padding_mask, tgt_fill_counter, 
            tgt_c, tgt_c_padding_mask, tgt_c_padding_mask_types
        ) = vals

        if type == "train":
            optimizer.zero_grad()

        structure_preds, constraint_preds = model(
            src, src_padding_mask, 
            tgt, tgt_padding_mask, tgt_fill_counter,
            tgt_c, tgt_c_padding_mask, 
            device
        )

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
            structure_preds, 
            constraint_preds,
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

        if with_wandb and type == "train":
            wandb.log({"train" : log})
        
        for key in log.keys():
            total_log[key].append(log[key])
    
    return {key : np.mean(value) for (key, value) in total_log.items()}

def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--with_wandb', action="store_true",
        help="enables wandb run")
    parser.add_argument('--config', type=str, default = os.path.join(os.path.dirname(__file__), 'main/config/config.yaml'), 
        help="config to use for run")
    parser.add_argument('--checkpoint', type=str, default="", 
        help="checkpoint to run from")
    args = parser.parse_args()
    return args

def main(args):
    config = load_config(args.config)
    if args.with_wandb:
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
            max_program_length= 30,
            loss_func=loss_factory(config['architecture']['loss'])
        )
    model.to(device)

    optimizer = optimizer_factory(
        model, 
        config['training']['optimizer'], 
        config['training']['lr']
    )

    # Training and validation loop 
    if args.with_wandb:
        wandb.watch(model)
    
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        
        model.train()
        log = iterate_through_data(
            model, train_dataloader, device, "train", 
            optimizer = optimizer, with_wandb = args.with_wandb
        )

        if not args.with_wandb:
            print(log)
        
        get_network_feedback(
            model, train_dataset, 
            f"examples/train/epoch_{epoch}",
            device,
            with_wandb = args.with_wandb
        )

        model.eval()
        with torch.inference_mode():
            log = iterate_through_data(
                model, val_dataloader, device, "val", 
                with_wandb = args.with_wandb
            )
            if args.with_wandb:
                wandb.log({"val" : log})
            else:
                print(log)

            get_network_feedback(
                model, val_dataset,
                f"examples/val/epoch_{epoch}",
                device,
                with_wandb = args.with_wandb
            )
    
    log = iterate_through_data(model, test_dataloader, device, "test", with_wandb = args.with_wandb)
    if args.with_wandb:
        cols = []
        data = []
        for key, value  in log.items():
            cols.append(key)
            data.append(value)
        table = wandb.Table(data=data, columns=cols)
        wandb.log({"test_summary" : table})
    else:
        print(log)
    
    model_save = os.path.join(data_filepath, "model.pt")
    save_model(model, model_save)

def overfit_to_one(args):
    config = load_config(args.config)
    if args.with_wandb:
        wandb.init(
            project = config['project'],
            name = config['name'],
            config = config
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = get_dataset()
    index = 467
    
    (scene, query_object), program_tokens = dataset[index]
    print(program_tokens['structure'])
    print(program_tokens['constraints'])
    
    single_point_dataset = torch.utils.data.Subset(dataset, [index])
    single_point_dataloader = get_dataloader(single_point_dataset, config['training']['batch_size'])

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

    for _ in range(1000):
        model.train()
        log = iterate_through_data(
            model, single_point_dataloader, device, "train", 
            optimizer = optimizer, with_wandb = args.with_wandb
        )

        if not args.with_wandb:
            print(log)
    
    get_network_feedback(
        model, single_point_dataset, 
        "examples",
        device,
        num_examples = 1,
        with_wandb = args.with_wandb
    )

if __name__ == '__main__':
    args = parseArguments()
    main(args)
    # overfit_to_one(args)