import torch
from tqdm import tqdm
import numpy as np

def test_network(model, test_dataloader, network_config):
    device = network_config['device']
    model.eval()
    with torch.no_grad():
        loss_sum = 0
        accuracies = []
        for vals in tqdm(test_dataloader):
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
            accuracies.append(accuracy)
            loss_sum += loss.item()

        num_test_examples = len(test_dataloader)
        print("Test Loss: {}, Test Accuracy: {}".format(loss_sum / num_test_examples, np.mean(accuracies)))

def infer_program(model, room, query_object):
    pass