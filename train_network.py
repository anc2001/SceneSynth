from main.network import ModelCore, \
    optimizer_factory, loss_factory, \
    save_model, load_model
from main.data_processing.dataset import \
    get_dataset, get_dataloader, collate_fn
from main.network.usage import iterate_through_data, \
    get_network_feedback, infer_program
from main.network.stat_logger import StatLogger
from main.common.utils import clear_folder, read_data

from main.config import load_config, data_filepath
from main.data_processing.dataset import get_dataloader

import torch
import os
import wandb
from argparse import ArgumentParser
import lovely_tensors as lt

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
    train_log = StatLogger("train")
    val_log = StatLogger("val")
    for epoch in range(epochs):
        model.train()
        iterate_through_data(
            model, train_dataloader, device, "train", 
            logger = train_log, optimizer = optimizer, with_wandb = args.with_wandb
        )
        
        get_network_feedback(
            model, train_dataset, 
            f"examples/train/epoch_{epoch}",
            device,
            with_wandb = args.with_wandb
        )

        model.eval()
        with torch.inference_mode():
            iterate_through_data(
                model, val_dataloader, device, "val",
                epoch = epoch, logger = val_log, with_wandb = args.with_wandb
            )

            get_network_feedback(
                model, val_dataset,
                f"examples/val/epoch_{epoch}",
                device,
                with_wandb = args.with_wandb
            )
    
    test_log = StatLogger("test")
    iterate_through_data(
        model, test_dataloader, device, "test", 
        logger = test_log, with_wandb = args.with_wandb
    )
    
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
    index = 0
        
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

    train_log = StatLogger("train")
    for _ in range(500):
        model.train()
        iterate_through_data(
            model, single_point_dataloader, device, "train", train_log,
            optimizer = optimizer, with_wandb = args.with_wandb
        )

    get_network_feedback(
        model, single_point_dataset, 
        "examples",
        device,
        num_examples = 1,
        with_wandb = args.with_wandb
    )

def test_thing():
    lt.monkey_patch()

    config = load_config(args.config)
    model = ModelCore(
        d_model = config['architecture']['d_model'],
        nhead = config['architecture']['nhead'],
        num_layers= config['architecture']['num_layers'],
        max_num_objects = 50,
        max_program_length= 30,
        loss_func=loss_factory(config['architecture']['loss'])
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model(model, os.path.join(data_filepath, "model.pt"))
    model.to(device)

    program_data = read_data(os.path.join(data_filepath, "program_data.pkl"))
    xs = program_data['xs']
    ys = program_data['ys']
    
    to_iterate = list(zip(xs, ys))
    item = to_iterate[0]
    (scene, query_object) = item[0]
    program_tokens = item[1]

    with torch.inference_mode():
        model.eval()
        collated_vals = collate_fn([item])

        structure_preds, constraint_preds = model(collated_vals, device)
        statistics = model.accuracy_fnc(structure_preds, constraint_preds, collated_vals)

        program = infer_program(model, scene, query_object, device)

if __name__ == '__main__':
    args = parseArguments()
    main(args)
    # overfit_to_one(args)
    # test_thing()