from main.network.utils import \
    optimizer_factory

import torch
from tqdm import tqdm
import numpy as np

def train_network(model, train_dataloader, val_dataloader, network_config):
    device = network_config['device']
    optimizer = optimizer_factory(model, network_config)

    for epoch in range(network_config['Training']['epochs']):
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

        model.eval()
        validation_accuracies = []
        validation_loss = 0
        with torch.no_grad():
            print("Validating epoch {}".format(epoch))
            for vals in tqdm(val_dataloader):
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
                validation_accuracies.append(accuracy)  
                validation_loss += loss.item()
        
        num_training_examples = len(val_dataloader)
        print("Epoch: {}, Validation Loss: {}, Validation Accuracy: {}".format(epoch, validation_loss / num_training_examples, np.mean(validation_accuracies)))

