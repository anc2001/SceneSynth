from main.config import \
    constraint_types, direction_types, \
    structure_vocab_map

import torch
import torch.nn as nn

class ConstraintDecoderModel(nn.Module):
    def __init__(self, d_model):
        # d_model: dimension of the model
        super().__init__()
        self.d_model = d_model
        self.type_embedding = nn.Embedding(len(constraint_types), self.d_model)

        self.constraint_type_selection = nn.Linear(d_model, len(constraint_types))
        self.object_selection = nn.Linear(3 * d_model, d_model)
        self.direction_selection = nn.Linear(4 * d_model, len(direction_types))

    def forward(
        self, 
        decoded_output,
        tgt,
        tgt_c,
        tgt_c_padding_mask, 
        src_e, 
        src_padding_mask
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
            type_selections,
            object_selections,
            direction_selections
        )
    
    # def inference(self, decoded_output, src_e, constraint_encoder_model):
    #     type_selection = torch.argmax(
    #         self.constraint_type_selection(
    #             decoded_output[-1]
    #         ), 
    #         dim = 1
    #     ).item()

    #     if type_selection == 5:
    #         # End of sequence constraint
    #         return [5, 0, 0, 0]
        
    #     type_e = constraint_encoder_model.constraint_type_embedding(torch.tensor([type_selection]))
    #     query_e = src_e[-1]
    #     object_selection_input = torch.cat([decoded_output[-1], type_e, query_e], dim = 1)

    #     pointer_embedding = torch.squeeze(self.object_selection(object_selection_input))
    #     object_selection = torch.argmax(
    #         torch.tensordot(
    #             src_e, pointer_embedding, dims = 1
    #         ),
    #         dim = 0
    #     ).item()
    #     if type_selection == 2 or type_selection == 3:
    #         # Orientation constraint
    #         return [type_selection, len(src_e) - 1, object_selection, 4]
        
    #     r_e = src_e[object_selection]
    #     diretion_selection_input = torch.cat([object_selection_input, r_e], dim = 1)
    #     direction_selection = torch.argmax(self.direction_selection(diretion_selection_input), dim = 1).item()

    #     return [type_selection, len(src_e) - 1, object_selection, direction_selection]