from main.network.object_encoder import ObjectEncoderModel
from main.network.constraint_decoder import ConstraintDecoderModel
from main.network.utils import PositionalEncoding, generate_square_subsequent_mask
from main.config import \
    object_types, \
    structure_vocab, structure_vocab_map, \
    constraint_types, \
    direction_types

import torch
from torch import nn
import numpy as np

# Full implementation of the model end to end 
class ModelCore(nn.Module):
    def __init__(
            self, 
            d_model : int, 
            nhead : int, 
            num_layers : int, 
            max_num_objects : int, 
            max_program_length : int,
            loss_func
        ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.max_program_length = max_program_length
        self.loss_fnc = loss_func

        self.object_encoder = ObjectEncoderModel(
            d_model=self.d_model,
            num_obj_categories=len(object_types) + 1, 
        )
        self.structure_embedding = nn.Embedding(
            len(structure_vocab), 
            self.d_model, 
            structure_vocab_map['<pad>']
        )

        self.positional_encoding = PositionalEncoding(self.d_model, max_len = max_num_objects)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward= 4 * d_model
            ),
            num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward= 4 * d_model
            ),
            num_layers=num_layers
        )

        self.structure_head = nn.Linear(self.d_model, len(structure_vocab))

        self.constraint_decoder = ConstraintDecoderModel(self.d_model, nhead, num_layers, max_program_length)

    def forward(self, collated_vals, device):
        (
            src, src_padding_mask, 
            tgt, tgt_padding_mask,
            tgt_c, tgt_c_padding_mask,
        ) = collated_vals 

        src_e = self.object_encoder(src)
        src_e = self.positional_encoding(src_e)
        memory = self.transformer_encoder(
            src_e, 
            src_key_padding_mask=src_padding_mask
        )

        tgt_e = self.structure_embedding(tgt.int())
        tgt_e = self.positional_encoding(tgt_e)

        # Structure prediction 
        tgt_mask = generate_square_subsequent_mask(tgt.size(0), tgt.size(1), self.nhead, device)

        decoded_output = self.transformer_decoder(
            tgt_e, 
            memory,
            tgt_mask = tgt_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = src_padding_mask
        )
        structure_preds = self.structure_head(decoded_output)

        # Constraint attribute prediction 
        constraint_preds = self.constraint_decoder(
            tgt_e, tgt_padding_mask, 
            tgt_c, tgt_c_padding_mask,
            src_e, src_padding_mask,
            device
        )

        return structure_preds, constraint_preds
    
    def infer(self, src, device, guarantee_program=False):
        src_e = self.object_encoder(src)
        src_e = self.positional_encoding(src_e)
        memory = self.transformer_encoder(src_e)

        # First predict structure
        tgt = torch.tensor([[structure_vocab_map['<sos>']]]).to(device)
        tgt_e = self.structure_embedding(tgt.int())
        tgt_e = self.positional_encoding(tgt_e)

        # guarantee program 
        # Base mask with <sos>, <eos>, and <pad> masked out 
        num_spots_to_fill = 1
        base_mask = torch.tensor([0, 0, 0, 1, 1, 1]).bool().to(device)
        need_to_end = False
        constraint_only_mask = torch.tensor([0, 1, 1, 1, 1, 1]).bool().to(device)
        while len(tgt) < self.max_program_length:
            # Regenerate the mask because lol? 
            tgt_mask = generate_square_subsequent_mask(tgt.size(0), tgt.size(1), self.nhead, device)

            decoded_output = self.transformer_decoder(tgt_e, memory, tgt_mask = tgt_mask)
            logits = torch.squeeze(self.structure_head(decoded_output[-1]))
            # guarantee program 
            if guarantee_program:
                mask = constraint_only_mask if need_to_end else base_mask
                logits[mask] = -float('inf')
            
            predicted_token = torch.argmax(logits).item()

            if predicted_token == structure_vocab_map['c']:
                num_spots_to_fill -= 1
            else:
                num_spots_to_fill += 1

            # guarantee program 
            if guarantee_program:
                if num_spots_to_fill == 0:
                    to_add = torch.tensor([[predicted_token]]).to(device)
                    tgt = torch.cat([tgt, to_add], dim = 0)
                    predicted_token = structure_vocab_map['<eos>']
                
                if 1 + len(tgt) + num_spots_to_fill == self.max_program_length:
                    need_to_end = True
                    print("Need to end flag set")
            
            to_add = torch.tensor([[predicted_token]]).to(device)
            
            to_add_e = self.structure_embedding(to_add.int())
            to_add_e = self.positional_encoding.encode_single(to_add_e, len(tgt))

            tgt = torch.cat([tgt, to_add], dim = 0)
            tgt_e = torch.cat([tgt_e, to_add_e], dim = 0)

            if predicted_token == structure_vocab_map['<eos>']:
                break

        
        # Then predict the constraints 
        num_constraints = torch.sum(tgt == structure_vocab_map['c']).item()
        constraints = self.constraint_decoder.infer(
            src_e, tgt_e, num_constraints, device, 
            guarantee_program = guarantee_program
        )

        program_structure = [structure_vocab[index] for index in tgt[1:-1]]
        return program_structure, constraints
    
    def loss(self, structure_preds, constraint_preds, collated_vals):
        (
            src, src_padding_mask, 
            tgt, tgt_padding_mask, 
            tgt_c, tgt_c_padding_mask,
        ) = collated_vals 

        y_structure_pred = torch.flatten(structure_preds, start_dim = 0, end_dim = 1)
        gt = torch.clone(tgt)
        y_structure_gt = torch.flatten(torch.roll(gt, -1, dims = 0), start_dim = 0, end_dim = 1)
        structure_loss = self.loss_fnc(y_structure_pred, y_structure_gt.long())
        # mask padding from structure loss 
        padding_mask = ~(y_structure_gt == structure_vocab_map['<pad>'])
        structure_loss *= padding_mask

        # Last direction should need to pick 
        constraints_flattened = tgt_c[~tgt_c_padding_mask]
        type_selections, object_selections, direction_selections = constraint_preds

        types_loss = self.loss_fnc(type_selections[~tgt_c_padding_mask], constraints_flattened[:, 0].long())
        objects_loss = self.loss_fnc(object_selections[~tgt_c_padding_mask], constraints_flattened[:, 2].long())
        directions_loss  = self.loss_fnc(direction_selections[~tgt_c_padding_mask], constraints_flattened[:, 3].long())

        loss = torch.mean(structure_loss) + torch.mean(types_loss) + torch.mean(objects_loss + directions_loss)
        return loss

    def evaluate_metrics(self, structure_preds, constraint_preds, collated_vals):
        (
            src, src_padding_mask, 
            tgt, tgt_padding_mask, 
            tgt_c, tgt_c_padding_mask,
        ) = collated_vals 

        total_tokens = 0
        total_correct_tokens = 0

        # Structure accuracy 
        gt = torch.clone(tgt)
        pred = torch.flatten(torch.argmax(structure_preds, dim = 2), start_dim = 0, end_dim = 1)
        gt = torch.flatten(torch.roll(gt, -1, dims = 0), start_dim = 0, end_dim = 1).int()
        padding_mask = ~(gt == structure_vocab_map['<pad>'])

        total_structure_tokens = torch.sum(padding_mask).item()
        total_structure_correct = torch.sum((pred == gt) * padding_mask).item()

        structure_accuracy = total_structure_correct / total_structure_tokens
        total_tokens += total_structure_tokens
        total_correct_tokens += total_structure_correct 

        # Constraint attribute accuracies 

        # Initialize relevant tensors 
        type_selections, object_selections, direction_selections = constraint_preds
        type_selections = type_selections[~tgt_c_padding_mask]
        object_selections = object_selections[~tgt_c_padding_mask]
        direction_selections = direction_selections[~tgt_c_padding_mask]

        n_c = object_selections.size(0)
        constraints_flattened = tgt_c[~tgt_c_padding_mask]

        # Constraint type accuracy 
        total_type_tokens = n_c
        pred = torch.argmax(type_selections, dim = 1)
        gt = constraints_flattened[:, 0]
        total_type_correct = torch.sum(pred == gt).item()

        type_accuracy = total_type_correct / total_type_tokens

        total_tokens += total_type_tokens
        total_correct_tokens += total_type_correct 

        # Object selection accuracy 
        total_object_tokens = n_c
        pred = torch.argmax(object_selections, dim = 1)
        gt = constraints_flattened[:, 2]
        total_object_correct = torch.sum(pred == gt).item()
        
        object_accuracy = total_object_correct / total_object_tokens

        total_tokens += total_object_tokens
        total_correct_tokens += total_object_correct 

        # Direction selection accuracy 
        total_direction_tokens = n_c
        pred = torch.argmax(direction_selections, dim = 1)
        gt = constraints_flattened[:, 3]
        total_direction_correct = torch.sum((pred == gt)).item()

        direction_accuracy = total_direction_correct / total_direction_tokens

        total_tokens += total_direction_tokens
        total_correct_tokens += total_direction_correct 

        total_accuracy = total_correct_tokens / total_tokens 
        
        statistics = {
            "accuracy": {
                "structure" : structure_accuracy,
                "type" : type_accuracy,
                "object" : object_accuracy,
                "direction" : direction_accuracy,
                "total" : total_accuracy                
            },
        }

        return statistics