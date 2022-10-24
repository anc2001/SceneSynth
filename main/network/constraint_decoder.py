from main.config import \
    constraint_types, direction_types, \
    structure_vocab_map, constraint_types_map, direction_types_map

import torch
import torch.nn as nn

class ConstraintDecoderModel(nn.Module):
    def __init__(self, d_model):
        # d_model: dimension of the model
        super().__init__()
        self.d_model = d_model
        self.type_embedding = nn.Embedding(len(constraint_types), self.d_model)

        self.constraint_type_selection = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.LeakyReLU(),
            nn.Linear(2 * d_model, len(constraint_types))
        )
        self.object_selection = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.LeakyReLU(),
            nn.Linear(2 * d_model, d_model)
        )
        self.direction_selection = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.LeakyReLU(),
            nn.Linear(2 * d_model, len(direction_types))
        )

    def forward(
        self, 
        decoded_output,
        tgt,
        tgt_c,
        tgt_c_padding_mask, 
        src_e, 
        src_padding_mask,
        device
    ):
        structure_c_mask = tgt == structure_vocab_map['c']
        only_constraint_heads = decoded_output[structure_c_mask] # done with tgt

        c_mask = ~tgt_c_padding_mask
        type_selections = self.constraint_type_selection(only_constraint_heads) # number of constraints x number of types 
        
        # Use ground truth types and query objects
        types = tgt_c[:, :, 0]
        types = types[c_mask]
        types = self.type_embedding(types.int())

        q_e = torch.gather(
            src_e,
            0,
            tgt_c[:, :, 1:2].expand(-1, -1, self.d_model).long()
        )
        q_e = q_e[c_mask]

        object_selection_input = torch.cat([only_constraint_heads, types, q_e], dim = 1)
        pointer_embedding = self.object_selection(object_selection_input) # number of constraints x d_model
        # For each pointer embedding, what batch is it from? 
        batch_reference_guide = torch.arange(tgt_c.shape[1]).expand(tgt_c.shape[0], -1)[c_mask]
        n_c = batch_reference_guide.size(0)
        s = src_e.size(0)
        object_selections = torch.zeros([n_c, s])
        # Way to vectorize this?? 
        for embedding_idx, batch_idx in enumerate(batch_reference_guide):
            logits = torch.tensordot(
                src_e[:, batch_idx, :], 
                pointer_embedding[embedding_idx], 
                dims = 1
            )
            # Set all src padding as -infty so 0 during softmax 
            logits[src_padding_mask[batch_idx]] = -float('inf')
            object_selections[embedding_idx, :] = logits

        # Use ground truth reference objects 
        r_e = torch.gather(
            src_e,
            0,
            tgt_c[:, :, 2:3].expand(-1, -1, self.d_model).long()
        )
        r_e = r_e[c_mask]

        direction_selection_input = torch.cat([object_selection_input, r_e], dim = 1)
        direction_selections = self.direction_selection(direction_selection_input)

        return (
            type_selections.to(device),
            object_selections.to(device),
            direction_selections.to(device)
        )
    
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
            logits = torch.tensordot(src_e, pointer_embedding, dims = 1)
            if guarantee_program:
                logits[-1] = 0 # Mask out logits for query object index 
            
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