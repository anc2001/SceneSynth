from main.config import \
    constraint_types, direction_types, \
    structure_vocab_map, constraint_types_map, direction_types_map
from main.network.utils import PositionalEncoding, generate_square_subsequent_mask

import torch
import torch.nn as nn
import numpy as np

class ConstraintDecoderModel(nn.Module):
    def __init__(self, d_model : int, nhead : int, num_layers : int, max_num_constraints : int):
        # d_model: dimension of the model
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.type_embedding = nn.Embedding(len(constraint_types), self.d_model)
        self.direction_embedding = nn.Embedding(
            len(direction_types), 
            self.d_model, 
            padding_idx=direction_types_map['<pad>']
        )
        self.objects_or_tree = nn.Embedding(2, self.d_model) # Differentiate sequence from being from objects vs being from tree
        self.relative_position_embedding = nn.Embedding(4, self.d_model)
        self.constraint_index_embedding = nn.Embedding(max_num_constraints, self.d_model)

        self.pe = PositionalEncoding(d_model, max_len = 4 * max_num_constraints)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward= 4 * d_model
            ),
            num_layers=num_layers
        )

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward= 4 * d_model
            ),
            num_layers=num_layers
        )

        self.type_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), # head
            nn.LeakyReLU(),
            nn.Linear(2 * d_model, len(constraint_types))
        )

        self.pointer_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), # head
            nn.LeakyReLU(),
            nn.Linear(2 * d_model, d_model)
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), # head
            nn.LeakyReLU(),
            nn.Linear(2 * d_model, len(direction_types))
        )

    def forward(
        self,
        tgt_e, # tree structure embedding 
        tgt_padding_mask,
        tgt_c, 
        tgt_c_padding_mask,
        src_e, # Objects 
        src_padding_mask,
        device
    ):
        batch_size = tgt_e.size(1)
        context = torch.cat(
            [
                src_e + self.objects_or_tree(torch.Tensor([0]).long()), 
                tgt_e + self.objects_or_tree(torch.Tensor([1]).long())
            ], 
            dim = 0
        )
        context_padding = torch.cat([src_padding_mask, tgt_padding_mask], dim = 1)

        t_e = self.type_embedding(tgt_c[:, :, 0].int())
        q_e = torch.gather(
            src_e,
            0,
            tgt_c[:, :, 1:2].expand(-1, -1, self.d_model).long()
        )
        r_e = torch.gather(
            src_e,
            0,
            tgt_c[:, :, 2:3].expand(-1, -1, self.d_model).long()
        )
        d_e = self.direction_embedding(tgt_c[:, :, 3].int())

        # Create s_e, sequence embedding for constraint attributes 
        s_e = self.type_embedding(
            torch.Tensor([[constraint_types_map['<sos>']]]).long() # Start all with <sos> 
        ).expand(-1, batch_size, -1) 
        s_e_padding = torch.Tensor([[False]]).expand(batch_size, -1) # Start all with false
        for c_idx in range(tgt_c.size(0)):
            c_e = torch.cat(
                [
                    t_e[c_idx:c_idx+1], 
                    q_e[c_idx:c_idx+1], 
                    r_e[c_idx:c_idx+1], 
                    d_e[c_idx:c_idx+1]
                ], 
                dim = 0
            )
            c_e += torch.unsqueeze(
                self.relative_position_embedding(torch.arange(4)), 
                dim = 1
            ) # broadcast over each batch 
            c_e += self.constraint_index_embedding(torch.Tensor([c_idx]).long())
            padding = torch.unsqueeze(tgt_c_padding_mask[c_idx], dim = 1).expand(-1, 4)
            s_e = torch.cat([s_e, c_e], dim = 0)
            s_e_padding = torch.cat([s_e_padding, padding], dim = 1)
        
        s_e = self.pe(s_e)

        memory = self.transformer_encoder(context, src_key_padding_mask = context_padding)

        tgt_mask = generate_square_subsequent_mask(s_e.size()[0], device)
        tgt_mask = torch.unsqueeze(tgt_mask, dim = 0)
        tgt_mask = tgt_mask.expand(s_e.shape[1] * self.nhead, -1, -1)

        decoded_outputs = self.transformer_decoder(
            s_e, memory, 
            tgt_key_padding_mask = s_e_padding, 
            memory_key_padding_mask = context_padding
        )
        
        type_selections = []
        reference_selections = []
        direction_selections = []
        indices = torch.arange(decoded_outputs.size(0))
        indices = indices[indices % 4 != 1]
        for i in indices:
            head = decoded_outputs[i]
            if i % 4 == 0:
                type_selection = self.type_head(head) # predict type 
                if len(type_selections):
                    type_selections = torch.cat(
                        [
                            type_selections,
                            torch.unsqueeze(type_selection, dim = 0)
                        ],
                        dim = 0
                    )
                else:
                    type_selections = torch.unsqueeze(type_selection, dim = 0)
            elif i % 4 == 2:
                pointer_embedding = self.pointer_head(head) # predict reference object index 
                
                # Vectorize this begin 
                reference_selection = []
                for batch_idx in range(batch_size):
                    logits = torch.tensordot(
                        src_e[:, batch_idx, :], 
                        pointer_embedding[batch_idx], 
                        dims = 1
                    )
                    # Set all src padding as -infty so 0 during softmax 
                    logits[src_padding_mask[batch_idx]] = -float('inf')
                    if len(reference_selection):
                        reference_selection = torch.cat(
                            [
                                reference_selection,
                                torch.unsqueeze(logits, dim = 0)
                            ],
                            dim = 0
                        )
                    else:
                        reference_selection = torch.unsqueeze(logits, dim = 0)
                # Vectorize this end 

                if len(reference_selections):
                    reference_selections = torch.cat(
                        [
                            reference_selections,
                            torch.unsqueeze(reference_selection, dim = 0)
                        ],
                        dim = 0
                    )
                else:
                    reference_selections = torch.unsqueeze(reference_selection, dim = 0)
            elif i % 4 == 3:
                direction_selection = self.direction_head(head) # predict direction 
                if len(direction_selections):
                    direction_selections = torch.cat(
                        [
                            direction_selections,
                            torch.unsqueeze(direction_selection, dim = 0)
                        ],
                        dim = 0
                    )
                else:
                    direction_selections = torch.unsqueeze(direction_selection, dim = 0)
                
        return (
            torch.Tensor(type_selections[:-1]).to(device),
            torch.Tensor(reference_selections).to(device),
            torch.Tensor(direction_selections).to(device)
        )
    
    def infer(self, src_e, tgt_e, num_constraints, device,  guarantee_program=False):
        context = torch.cat(
            [
                src_e + self.objects_or_tree(torch.Tensor([0]).long()), 
                tgt_e + self.objects_or_tree(torch.Tensor([1]).long())
            ], 
            dim = 0
        )
        memory = self.transformer_encoder(context)

        constraints = []
        c_e = self.type_embedding(
            torch.tensor([[constraint_types_map['<sos>']]]).int()
        )

        constraints_left = num_constraints
        relative_index = 0
        absolute_index = 0
        current_constraint = []
        while constraints_left:
            decoded_outputs = self.transformer_decoder(c_e, memory)
            head = decoded_outputs[-1]
            if relative_index == 0: # predict type 
                logits = self.type_head(head)
                if guarantee_program:
                    predicted_token = torch.argmax(logits[:, :4], dim = 1)
                else:
                    predicted_token = torch.argmax(logits, dim = 1)
                
                c_e_to_add = self.type_embedding(predicted_token)
                predicted_token = predicted_token.item()
                
            elif relative_index == 1:
                predicted_token = len(src_e) - 1
                c_e_to_add = src_e[predicted_token]
            elif relative_index == 2: # predict reference object index
                pointer_embedding = self.pointer_head(head)
                logits = torch.tensordot(
                    src_e[:, 0, :], 
                    pointer_embedding[0], 
                    dims = 1
                )
                if guarantee_program:
                    logits[-1] = -float('inf')
                
                predicted_token = torch.argmax(logits)
                predicted_token = predicted_token.item()
                c_e_to_add = src_e[predicted_token]
            elif relative_index == 3: # predict direction (if applicable)
                if current_constraint[0] == constraint_types_map['attach'] or current_constraint[0] == constraint_types_map['reachable_by_arm']:
                    logits = self.direction_head(head)
                    predicted_token = torch.argmax(logits, dim = 1)
                else:
                    predicted_token = torch.Tensor([direction_types_map['<pad>']]).int()
                c_e_to_add = self.direction_embedding(predicted_token)
                predicted_token = predicted_token.item()
            
            c_e_to_add += self.relative_position_embedding(
                torch.Tensor([relative_index]).int()
            )
            c_e_to_add += self.constraint_index_embedding(
                torch.Tensor([num_constraints - constraints_left]).int()
            )
            c_e_to_add = self.pe.encode_single(c_e_to_add, absolute_index)
            c_e = torch.cat(
                [c_e, torch.unsqueeze(c_e_to_add, dim = 1)], 
                dim = 0
            )

            absolute_index += 1
            current_constraint.append(int(predicted_token))
            if relative_index == 3:
                relative_index = 0
                constraints_left -= 1
                constraints.append(current_constraint)
                current_constraint = []
            else:
                relative_index += 1
        
        return constraints
