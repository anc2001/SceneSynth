from main.network.object_encoder import ObjectEncoderModel
from main.network.constraint_decoder import ConstraintDecoderModel
from main.network.utils import PositionalEncoding, generate_square_subsequent_mask
from main.config import \
    object_types, \
    structure_vocab, structure_vocab_map, \
    constraint_types_map

import torch
from torch import nn, Tensor

# Full implementation of the model end to end 
class ModelCore(nn.Module):
    def __init__(self, d_model : int, nhead : int, num_layers : int, loss_func):
        # d_model: dimension of the model
        # nhead: number of heads in the multiheadattention models
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.object_encoder = ObjectEncoderModel(
            d_model=self.d_model,
            num_obj_categories=len(object_types) + 1, 
        )
        self.structure_embedding = nn.Embedding(
            len(structure_vocab), 
            self.d_model, 
            structure_vocab_map['<pad>']
        )

        self.positional_encoding = PositionalEncoding(self.d_model)
        self.structure_head = nn.Linear(self.d_model, len(structure_vocab))
        self.constraint_decoder = ConstraintDecoderModel(self.d_model)
        self.loss_fnc = loss_func

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=2 * d_model
            ),
            num_layers=num_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward= 2 * d_model
            ),
            num_layers=num_layers
        )

    def forward(
        self, 
        src, src_padding_mask, 
        tgt, tgt_padding_mask,  
        tgt_c, tgt_c_padding_mask,
        device
    ):
        src_e = self.object_encoder(src)
        src_e = self.positional_encoding(src_e)

        memory = self.transformer_encoder(
            src_e, 
            src_key_padding_mask=src_padding_mask
        )

        tgt_e = self.structure_embedding(tgt.int())
        tgt_e = self.positional_encoding(tgt_e)

        tgt_mask = generate_square_subsequent_mask(tgt.size()[0], device)
        tgt_mask = torch.unsqueeze(tgt_mask, dim = 0)
        tgt_mask = tgt_mask.expand(src.shape[1] * self.nhead, -1, -1)

        decoded_output = self.transformer_decoder(
            tgt_e, 
            memory,
            tgt_mask = tgt_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = src_padding_mask
        )
        structure_preds = self.structure_head(decoded_output)

        constraint_preds = self.constraint_decoder(
            decoded_output, tgt,
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
        base_mask = torch.tensor([1, 1, 1, -float('inf'), -float('inf'), -float('inf')]).to(device)
        while len(tgt) < 50:
            decoded_output = self.transformer_decoder(tgt_e, memory)
            logits = self.structure_head(decoded_output[-1])
            # guarantee program 
            if guarantee_program:
                logits *= base_mask
            
            predicted_token = torch.argmax(logits).item()

            # guarantee program 
            if guarantee_program:
                if predicted_token == structure_vocab_map['c']:
                    num_spots_to_fill -= 1
                else:
                    num_spots_to_fill += 1
                if num_spots_to_fill == 0:
                    to_add = torch.tensor([[predicted_token]]).to(device)
                    tgt = torch.cat([tgt, to_add], dim = 0)
                    predicted_token = structure_vocab_map['<eos>']
            
            to_add = torch.tensor([[predicted_token]]).to(device)
            tgt = torch.cat([tgt, to_add], dim = 0)
            tgt_e = self.structure_embedding(tgt.int())
            tgt_e = self.positional_encoding(tgt_e)

            if predicted_token == structure_vocab_map['<eos>']:
                break

        # Then predict the constraints 
        structure_preds = self.transformer_decoder(tgt_e, memory)
        constraints = self.constraint_decoder.infer(
            structure_preds, tgt, src_e, device, 
            guarantee_program = guarantee_program
        )

        program_structure = [structure_vocab[index] for index in tgt[1:]]
        return program_structure, constraints.tolist()
    
    def loss(
        self, 
        structure_preds, constraint_preds, 
        tgt, tgt_padding_mask, 
        tgt_c, tgt_c_padding_mask
    ):
        x_structure = torch.flatten(structure_preds, start_dim = 0, end_dim = 1)
        y_structure = torch.flatten(torch.roll(tgt, -1, dims = 0), start_dim = 0, end_dim = 1)
        structure_loss = self.loss_fnc(x_structure, y_structure.long())
        # mask padding from structure loss 
        padding_mask = ~torch.flatten(
            tgt == structure_vocab_map['<pad>'],
            start_dim = 0,
            end_dim = 1
        )
        structure_loss *= padding_mask

        constraints_flattened = tgt_c[~tgt_c_padding_mask]
        type_selections, object_selections, direction_selections = constraint_preds

        types_loss = self.loss_fnc(type_selections, constraints_flattened[:, 0].long())
        objects_loss = self.loss_fnc(object_selections, constraints_flattened[:, 2].long())
        directions_loss  = self.loss_fnc(direction_selections, constraints_flattened[:, 3].long())
        # mask orientation constraints from directions loss 
        directions_mask = torch.logical_or(
            constraints_flattened[:, 0] == constraint_types_map['attach'],
            constraints_flattened[:, 0] == constraint_types_map['reachable_by_arm']
        )
        directions_loss *= directions_mask

        return torch.mean(structure_loss) + torch.mean(types_loss + objects_loss + directions_loss)
    
    def accuracy_fnc(
        self, 
        structure_preds, tgt, 
        constraint_preds, tgt_c, tgt_c_padding_mask
    ):
        type_selections, object_selections, direction_selections = constraint_preds
        n_c = type_selections.size(0)
        constraints_flattened = tgt_c[~tgt_c_padding_mask]
        padding_mask = ~torch.flatten(
            tgt == structure_vocab_map['<pad>'],
            start_dim = 0,
            end_dim = 1
        )
        directions_mask = torch.logical_or(
            constraints_flattened[:, 0] == constraint_types_map['attach'],
            constraints_flattened[:, 0] == constraint_types_map['reachable_by_arm']
        )

        # Structure accuracy 
        total_structure_tokens = torch.sum(padding_mask).item()
        total_structure_correct = torch.sum(
            (
                torch.flatten(
                    torch.argmax(structure_preds, dim = 2),
                    start_dim = 0,
                    end_dim = 1
                )
            ==
                torch.flatten(
                    torch.roll(tgt, -1, dims = 0),
                    start_dim = 0,
                    end_dim = 1
                )
            )
            * padding_mask
        ).item()
        structure_accuracy = total_structure_correct / total_structure_tokens

        # Constraint type accuracy 
        total_type_tokens = n_c
        total_type_correct = torch.sum(
            torch.argmax(type_selections, dim = 1) == constraints_flattened[:, 0]
        ).item()
        type_accuracy = total_type_correct / total_type_tokens

        # Object selection accuracy 
        total_object_tokens = n_c
        total_object_correct = torch.sum(
            torch.argmax(object_selections, dim = 1) == constraints_flattened[:, 2]
        ).item()
        object_accuracy = total_object_correct / total_object_tokens

        # Direction selection accuracy 
        total_direction_tokens = torch.sum(directions_mask).item()
        total_direction_correct = torch.sum(
            (torch.argmax(direction_selections, dim = 1) == constraints_flattened[:, 3]) * directions_mask
        ).item()
        direction_accuracy = total_direction_correct / total_direction_tokens

        return (
            structure_accuracy,
            type_accuracy,
            object_accuracy,
            direction_accuracy
        )