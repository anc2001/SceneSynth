from main.config import \
    constraint_types, direction_types, \
    structure_vocab_map, constraint_types_map, direction_types_map
from main.network.utils import PositionalEncoding, generate_square_subsequent_mask

import torch
import torch.nn as nn

class ConstraintDecoderModel(nn.Module):
    def __init__(self, d_model : int, nhead : int, num_layers : int, max_num_objects : int):
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
        self.constraint_index_embedding = nn.Embedding(max_num_objects, self.d_model)
        # 0 -> <pad>, 1 -> <mask>, 2 -> <sos>, 3 -> <eos> 
        self.token_embedding = nn.Embedding(4, self.d_model, padding_idx=0) 

        self.pe = PositionalEncoding(d_model, max_len = 100)

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
        src_e,
        src_padding_mask,
        device
    ):
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
        s_e = torch.Tensor() # Start all with <sos> 
        s_e_padding = [] # Start all with false, guarantee to always have at least 1 constraint 
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
            s_e = torch.cat([s_e, c_e], dim = 0) if len(s_e) else c_e
            s_e_padding = torch.cat([s_e_padding, padding], dim = 1) if len(s_e_padding) else padding
        
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
        batch_size = decoded_outputs.size(1)
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
            torch.Tensor(type_selections).to(device),
            torch.Tensor(reference_selections).to(device),
            torch.Tensor(direction_selections).to(device)
        )
    # def forward(
    #     self, 
    #     decoded_output,
    #     tgt,
    #     tgt_c,
    #     tgt_c_padding_mask, 
    #     src_e, 
    #     src_padding_mask,
    #     device
    # ):
    #     # Predict constraints autoregressively 
    #     batch_size = tgt_c.size(0)
    #     for batch_idx in range(batch_size):
    #         pass

    #     structure_c_mask = tgt == structure_vocab_map['c']
    #     only_constraint_heads = decoded_output[structure_c_mask] # done with tgt

    #     c_mask = ~tgt_c_padding_mask
    #     type_selections = self.constraint_type_selection(only_constraint_heads) # number of constraints x number of types 
        
    #     # Use ground truth types and query objects
    #     types = tgt_c[:, :, 0]
    #     types = types[c_mask]
    #     types = self.type_embedding(types.int())

    #     q_e = torch.gather(
    #         src_e,
    #         0,
    #         tgt_c[:, :, 1:2].expand(-1, -1, self.d_model).long()
    #     )
    #     q_e = q_e[c_mask]

    #     object_selection_input = torch.cat([only_constraint_heads, types, q_e], dim = 1)
    #     pointer_embedding = self.object_selection(object_selection_input) # number of constraints x d_model
    #     # For each pointer embedding, what batch is it from? 
    #     batch_reference_guide = torch.arange(tgt_c.shape[1]).expand(tgt_c.shape[0], -1)[c_mask]
    #     n_c = batch_reference_guide.size(0)
    #     s = src_e.size(0)
    #     object_selections = torch.zeros([n_c, s])
    #     # Way to vectorize this?? 
    #     for embedding_idx, batch_idx in enumerate(batch_reference_guide):
    #         logits = torch.tensordot(
    #             src_e[:, batch_idx, :], 
    #             pointer_embedding[embedding_idx], 
    #             dims = 1
    #         )
    #         # Set all src padding as -infty so 0 during softmax 
    #         logits[src_padding_mask[batch_idx]] = -float('inf')
    #         object_selections[embedding_idx, :] = logits

    #     # Use ground truth reference objects 
    #     r_e = torch.gather(
    #         src_e,
    #         0,
    #         tgt_c[:, :, 2:3].expand(-1, -1, self.d_model).long()
    #     )
    #     r_e = r_e[c_mask]

    #     direction_selection_input = torch.cat([object_selection_input, r_e], dim = 1)
    #     direction_selections = self.direction_selection(direction_selection_input)

    #     return (
    #         type_selections.to(device),
    #         object_selections.to(device),
    #         direction_selections.to(device)
    #     )
    
    def infer(self, decoded_output, tgt, src_e, device, guarantee_program=False):
        structure_c_mask = tgt == structure_vocab_map['c']
        only_constraint_heads = decoded_output[structure_c_mask] # done with tgt
        n_c = only_constraint_heads.size(0)
        types = torch.argmax(
            self.constraint_type_selection(only_constraint_heads),
            dim = 1
        )

        query_e = src_e[-1].expand(n_c, -1)
        type_e = self.type_embedding(types.int())
        object_selection_input = torch.cat(
            [only_constraint_heads, type_e, query_e],
            dim = 1
        )
        pointer_embeddings = self.object_selection(object_selection_input)

        # Vectorize this
        object_selections = []
        for pointer_embedding in pointer_embeddings:
            logits = torch.squeeze(torch.tensordot(src_e, pointer_embedding, dims = 1))
            if guarantee_program:
                logits[-1] = -float('inf') # Mask out logits for query object index 
            
            object_selection = torch.argmax(logits).item()
            object_selections.append(object_selection)
        object_selections = torch.Tensor(object_selections).long().to(device)

        r_e = torch.squeeze(src_e, dim = 1)[object_selections.long()]
        direction_selection_input = torch.cat([object_selection_input, r_e], dim = 1)
        direction_selections = torch.argmax(
            self.direction_selection(direction_selection_input),
            dim = 1
        )

        if guarantee_program:
            padding_directions = torch.full(
                direction_selections.size(), 
                direction_types_map['<pad>']
            ).to(device)

            is_location_constraint = torch.logical_or(
                types == constraint_types_map['attach'],
                types == constraint_types_map['reachable_by_arm']
            )

            direction_selections = torch.where(
                is_location_constraint, 
                direction_selections, 
                padding_directions
            )

        query_object_indices = torch.full(types.size(), len(src_e) - 1).to(device)
        constraints = torch.cat(
            [
                torch.unsqueeze(types, dim = 1),
                torch.unsqueeze(query_object_indices, dim = 1),
                torch.unsqueeze(object_selections, dim = 1),
                torch.unsqueeze(direction_selections, dim = 1)
            ],
            dim = 1
        )

        return constraints.to(device)