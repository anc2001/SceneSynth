from main.program_extraction import get_dataloaders
from main.network.utils import \
    optimizer_factory, loss_factory
from main.config import get_network_config
from main.network.core import ModelCore

import torch
from tqdm import tqdm
import numpy as np

def train_network():
    train, test, val = get_dataloaders()
    network_config = get_network_config()
    model = ModelCore(
        d_model = network_config['Architecture']['d_model'],
        nhead = network_config['Architecture']['nhead'],
        num_layers= network_config['Architecture']['d_model'],
        loss_func=loss_factory(network_config)
    )

    optimizer = optimizer_factory(model, network_config)

    for epoch in range(network_config['Training']['epochs']):
        epoch_loss = 0
        epoch_accuracies = []
        model.train()
        print("Training epoch {}".format(epoch))
        for src, src_padding_mask, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask in tqdm(train):
            output = model(
                src, src_padding_mask, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask
            )

            optimizer.zero_grad()
            # Compute Loss
            loss = model.loss(output, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask)
            # Backpropagation 
            loss.backward()
            # Update
            optimizer.step()

            accuracy = model.accuracy_fnc(output, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask)
            epoch_accuracies.append(accuracy)
            epoch_loss += loss.item()

        num_training_examples = len(train)
        print("Epoch: {}, Train Loss: {}, Train Accuracy: {}".format(epoch, epoch_loss / num_training_examples, np.mean(epoch_accuracies)))

        model.eval()
        validation_accuracies = []
        validation_loss = 0
        with torch.no_grad():
            print("Validating epoch {}".format(epoch))
            for src, src_padding_mask, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask in tqdm(val):
                output = model(
                    src, src_padding_mask, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask
                )
                loss = model.loss(output, tgt, tgt_padding_mask)
                accuracy = model.accuracy_fnc(output, tgt, tgt_padding_mask, tgt_c, tgt_c_padding_mask)
                validation_loss += loss.item()
                validation_accuracies.append(accuracy)  
            
        print("Epoch: {}, Validation Loss: {}, Validation Accuracy: {}".format(epoch, validation_loss / num_training_examples, np.mean(validation_accuracies)))