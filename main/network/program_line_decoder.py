import torch
import torch.nn as nn

class ProgramLineDecoderModel(nn.Module):
    def __init__(self, d_model):
        # d_model: dimension of the model
        super().__init__()
        self.d_model = d_model
        self.constraint_type_selection = nn.Linear(d_model, len(variables.CONSTRAINT_TYPES))
        self.object_selection = nn.Linear(3 * d_model, d_model)
        self.direction_selection = nn.Linear(4 * d_model, len(variables.DIRECTIONS))

    def inference(self, decoded_output, src_e, constraint_encoder_model):
        type_selection = torch.argmax(
            self.constraint_type_selection(
                decoded_output[-1]
            ), 
            dim = 1
        ).item()

        if type_selection == 5:
            # End of sequence constraint
            return [5, 0, 0, 0]
        
        type_e = constraint_encoder_model.constraint_type_embedding(torch.tensor([type_selection]))
        query_e = src_e[-1]
        object_selection_input = torch.cat([decoded_output[-1], type_e, query_e], dim = 1)

        pointer_embedding = torch.squeeze(self.object_selection(object_selection_input))
        object_selection = torch.argmax(
            torch.tensordot(
                src_e, pointer_embedding, dims = 1
            ),
            dim = 0
        ).item()
        if type_selection == 2 or type_selection == 3:
            # Orientation constraint
            return [type_selection, len(src_e) - 1, object_selection, 4]
        
        r_e = src_e[object_selection]
        diretion_selection_input = torch.cat([object_selection_input, r_e], dim = 1)
        direction_selection = torch.argmax(self.direction_selection(diretion_selection_input), dim = 1).item()

        return [type_selection, len(src_e) - 1, object_selection, direction_selection]

    def forward(
        self, 
        decoded_output, 
        tgt, 
        src_e, 
        src_key_padding_mask,
        constraint_encoder_model
    ):
        # decoded_output: [sequence_length, batch_size, d_model]
        # tgt: [sequence_length, batch_size, constraint_representation_length]
        # src_e: [sequence_length, batch_size, d_model]
        # src_key_padding_mask: [batch_size, sequence_length]
        # constraint_encoder_model: ConstraintEncoderModel

        type_selections = self.constraint_type_selection(decoded_output)
        
        # Use ground truth types and query objects
        types = tgt[:, :, 0]
        types = constraint_encoder_model.constraint_type_embedding(types.int())
        q_e = torch.gather(
            src_e,
            0,
            tgt[:, :, 1:2].expand(-1, -1, self.d_model).long()
        )

        # Need to shift ground truth prediction values to the left so that each element is predicting the next
        types = torch.roll(types, -1, dims = 0)
        q_e = torch.roll(q_e, -1, dims = 0)
        object_selection_input = torch.cat([decoded_output, types, q_e], dim = 2)
        pointer_embedding = self.object_selection(object_selection_input)

        n = src_e.size(1)
        s = src_e.size(0)
        t = tgt.size(0)

        # Update for a faster vectorized version 
        object_selections = torch.zeros([t, n, s])
        for tgt_sequence_idx in range(t):
            for batch_n in range(n):
                object_selections[tgt_sequence_idx, batch_n, :] = torch.tensordot(
                    src_e[:, batch_n, :],
                    pointer_embedding[tgt_sequence_idx][batch_n],
                    dims = 1
                )

        object_mask = torch.unsqueeze(src_key_padding_mask, dim = 0).expand(t, -1, -1)
        object_selections[object_mask] = -float('inf')

        # Use ground truth reference objects 
        r_e = torch.gather(
            src_e,
            0,
            tgt[:, :, 2:3].expand(-1, -1, self.d_model).long()
        )
        r_e = torch.roll(r_e, -1, 0)
        direction_selection_input = torch.cat([object_selection_input, r_e], dim = 2)
        direction_selections = self.direction_selection(direction_selection_input)

        return (
            type_selections,
            object_selections,
            direction_selections
        )