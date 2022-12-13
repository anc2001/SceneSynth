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

def iterate_through_data(
    model, dataloader, device, type, 
    epoch = 0, logger = None, 
    optimizer=None, with_wandb = False
):     
    for vals in tqdm(dataloader):
        # Calculate for batch the max number of objects 

        if type == "train":
            optimizer.zero_grad()
        
        # Forward Pass
        structure_preds, constraint_preds = model(vals, device)
        # Compute Loss
        loss = model.loss(structure_preds, constraint_preds, vals)
        
        if type == "train":
            # Backpropagation 
            loss.backward()
            # Update
            optimizer.step()

        statistics = model.accuracy_fnc(structure_preds, constraint_preds, vals)

        log = {
            "loss" : loss.item(),
            "accuracy" : statistics["accuracy"],
            "f1_score" : statistics["f1_score"]
        }
        
        if with_wandb and type == "train":
            logger.log(log)
        else:
            logger.accumulate(log)
    
    if with_wandb and type != "train":
        summary_stats = logger.get_summary()
        if type == "val":
            wandb.log({type : summary_stats, "epoch" : epoch})
        else:
            data = [val for val in summary_stats.values()]
            columns = [col for col in summary_stats.keys()]
            wandb.log({type, wandb.Table(data = data, columns = columns)})