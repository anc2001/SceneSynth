from main.program_extraction import get_dataloaders
from main.network.utils import \
    optimizer_factory, loss_factory
from main.config import get_network_config, data_filepath
from main.network.core import ModelCore
from main.network.utils import save_model

import torch
from tqdm import tqdm
import numpy as np
import os

def train_network():
    train, test, val = get_dataloaders()
    network_config = get_network_config()
    device = network_config['device']
    model = ModelCore(
        d_model = network_config['Architecture']['d_model'],
        nhead = network_config['Architecture']['nhead'],
        num_layers= network_config['Architecture']['d_model'],
        loss_func=loss_factory(network_config)
    )
    model.to(device)

    optimizer = optimizer_factory(model, network_config)

    for epoch in range(network_config['Training']['epochs']):
        epoch_loss = 0
        epoch_accuracies = []
        model.train()
        print("Training epoch {}".format(epoch))
        for vals in tqdm(train):
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

        num_training_examples = len(train)
        print("Epoch: {}, Train Loss: {}, Train Accuracy: {}".format(epoch, epoch_loss / num_training_examples, np.mean(epoch_accuracies)))

        model.eval()
        validation_accuracies = []
        validation_loss = 0
        with torch.no_grad():
            print("Validating epoch {}".format(epoch))
            for vals in tqdm(val):
                src, src_padding_mask, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask = vals
                structure_preds, constraint_preds = model(
                    src, src_padding_mask, 
                    tgt, tgt_padding_mask,
                    tgt_c, tgt_c_padding_mask
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
                validation_loss += loss.item()
                validation_accuracies.append(accuracy)  
            
        print("Epoch: {}, Validation Loss: {}, Validation Accuracy: {}".format(epoch, validation_loss / num_training_examples, np.mean(validation_accuracies)))
    
    save_model(model, os.path.join(data_filepath, "model.pt"))
# def test(model, test_dataloader):
#     model.eval()
#     with torch.no_grad():
#         loss_sum = 0
#         accuracies = []
#         for src, src_key_padding_mask, tgt, tgt_key_padding_mask in tqdm(test_dataloader):
#             output = model(src, src_key_padding_mask, tgt, tgt_key_padding_mask)
#             loss = model.loss(output, tgt, tgt_key_padding_mask)
#             accuracy = model.accuracy_fnc(output, tgt, tgt_key_padding_mask)
#             loss_sum += loss.item()
#             accuracies.append(accuracy)

#         num_test_examples = len(test_dataloader)
#         print("Test Loss: {}, Test Accuracy: {}".format(loss_sum / num_test_examples, np.mean(accuracies)))