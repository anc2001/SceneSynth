from main.common.utils import vectorize_scene, clear_folder
from main.config import constraint_types, direction_types
from main.network.stat_logger import StatLogger

import torch
import numpy as np
from tqdm import tqdm
from main.common.language import ProgramTree, verify_program
import wandb
import matplotlib.pyplot as plt

def get_network_feedback(model, dataset, base_tag, device, num_examples = 5, with_wandb = False):
    print("Getting network feedback")
    indices = np.array(dataset.indices)
    np.random.shuffle(indices)
    program_dataset = dataset.dataset

    columns = ["type", "feedback"]
    data = []

    tags = [base_tag + f"/example_{i}" for i in range(num_examples)]
    for tag, idx in tqdm(zip(tags, indices[:num_examples]), total=len(tags)):
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

def exposure_bias(
    model, device,
    src, src_padding_mask, 
    tgt, tgt_padding_mask, tgt_fill_counter,
    tgt_c, tgt_c_padding_mask, tgt_c_padding_mask_types,
    objects_max_length
):
    # Corrupt structure [T_S, N] - randomly flip token 
    T_S = tgt.size(0)
    T_C = tgt_c.size(0)
    tgt_corrupt = torch.clone(tgt)
    tgt_c_corrupt = torch.clone(tgt_c)
    for sequence_idx in range(tgt.size(1)):
        # Structure 
        num_tokens = T_S - torch.sum(tgt_padding_mask[sequence_idx, :]).item()
        changed_token_idx = np.random.choice(int(num_tokens * 0.25)) # Choose from the first quarter 
        token = tgt[changed_token_idx, sequence_idx]
        new_token = token
        while new_token == token:
            new_token = np.random.choice(4)
        tgt_corrupt[changed_token_idx, sequence_idx] = new_token

        # Constraints 
        num_tokens = T_C- torch.sum(tgt_c_padding_mask[sequence_idx, :]).item()
        changed_token_idx = np.random.choice(int(num_tokens * 0.25)) # Choose from the first quarter 
        relative_index = changed_token_idx % 4
        token = tgt_c[changed_token_idx, sequence_idx]
        new_token = token
        if relative_index == 0:
            while new_token == token:
                new_token = np.random.choice(len(constraint_types))
        if relative_index == 0 or relative_index == 2:
            new_token = np.random.choice(token)
        elif relative_index == 3:
            while new_token == token:
                new_token = np.random.choice(len(direction_types))
        tgt_c_corrupt[changed_token_idx, sequence_idx] = new_token

    # Only structure 
    structure_preds, constraint_preds = model(
        src, src_padding_mask, 
        tgt_corrupt, tgt_padding_mask, tgt_fill_counter,
        tgt_c, tgt_c_padding_mask, 
        device
    )

    structure_only_statistics = model.accuracy_fnc(
        structure_preds, 
        constraint_preds,
        tgt, tgt_c, tgt_c_padding_mask, tgt_c_padding_mask_types,
        objects_max_length
    )

    # Only constraints 
    structure_preds, constraint_preds = model(
        src, src_padding_mask, 
        tgt, tgt_padding_mask, tgt_fill_counter,
        tgt_c_corrupt, tgt_c_padding_mask, 
        device
    )

    constraint_only_statistics = model.accuracy_fnc(
        structure_preds, 
        constraint_preds,
        tgt, tgt_c, tgt_c_padding_mask, tgt_c_padding_mask_types,
        objects_max_length
    )

    # Structure + constraints 
    structure_preds, constraint_preds = model(
        src, src_padding_mask, 
        tgt_corrupt, tgt_padding_mask, tgt_fill_counter,
        tgt_c_corrupt, tgt_c_padding_mask, 
        device
    )

    with_both_statistics = model.accuracy_fnc(
        structure_preds, 
        constraint_preds,
        tgt, tgt_c, tgt_c_padding_mask, tgt_c_padding_mask_types,
        objects_max_length
    )

    log = {
        "corrupt_structure_accuracy" : structure_only_statistics["accuracy"],
        "corrupt_constraint_accuracy" : constraint_only_statistics["accuracy"],
        "corrupt_both_exposure_accuracy" : with_both_statistics["accuracy"]
    }
    return log

def iterate_through_data(
    model, dataloader, device, type, logger,
    optimizer=None, with_wandb = False
):     
    for vals in tqdm(dataloader):
        # Extract vals from dataloader 
        (
            src, src_padding_mask, 
            tgt, tgt_padding_mask, tgt_fill_counter, 
            tgt_c, tgt_c_padding_mask, tgt_c_padding_mask_types,
            objects_max_length
        ) = vals

        # Calculate for batch the max number of objects 

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

        statistics = model.accuracy_fnc(
            structure_preds, 
            constraint_preds,
            tgt, tgt_c, tgt_c_padding_mask, tgt_c_padding_mask_types,
            objects_max_length
        )

        log = {
            "loss" : loss.item(),
            "accuracy" : statistics["accuracy"],
            "f1_score" : statistics["f1_score"]
        }
        
        if type == "train":
            log["exposure_bias"] = exposure_bias(
                model, device, 
                src, src_padding_mask, 
                tgt, tgt_padding_mask, tgt_fill_counter, 
                tgt_c, tgt_c_padding_mask, tgt_c_padding_mask_types, 
                objects_max_length
            )

        logger.log(log)