import torch
import torch.nn as nn

class ProgramLineEncoderModel(nn.Module):
    def __init__(self, d_model, num_constraint_categories, num_directions):
        # d_model: dimension of the model
        # num_constraint_categories: number of constraint categories
        # num_directions: number of directions
        super().__init__()
        self.d_model = d_model
        self.num_constraint_categories = num_constraint_categories
        self.constraint_type_embedding = nn.Embedding(num_constraint_categories, d_model)
        self.direction_embedding = nn.Embedding(num_directions, d_model)
        self.fc = nn.Sequential(
            nn.Linear(4 * d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model)
        )
        
    def forward(
            self,
            constraints, 
            constraints_key_padding_mask, 
            obj_e
        ):
        # constraints: [constraint_sequence_length, batch_size, constraint_representation_length]
        # constraints_key_padding_mask: [batch_size, constraint_sequence_length]
        # obj_e: [object_sequence_length, batch_size, d_model]

        t_e = self.constraint_type_embedding(constraints[:, :, 0].int())
        q_e = torch.gather(
            obj_e,
            0,
            constraints[:, :, 1:2].expand(-1, -1, self.d_model).long()
        )
        r_e = torch.gather(
            obj_e, 
            0, 
            constraints[:, :, 2:3].expand(-1, -1, self.d_model).long()
        )
        d_e = self.direction_embedding(constraints[:, :, 3].int())

        # For all orientation constraints, mask the directional embedding 
        constraints_orientation_mask = torch.logical_or(
            constraints[:, :, 0] == 2,
            constraints[:, :, 0] == 3
        )
        d_e[constraints_orientation_mask] = torch.zeros(self.d_model)
        input = torch.cat([q_e, r_e, d_e], dim = 2)

        # Mask the arguments to the start and end constraints 
        constraints_start_end = torch.logical_or(
            constraints[:, :, 0] == 4,
            constraints[:, :, 0] == 5       
        )
        input[constraints_start_end] = torch.zeros(3 * self.d_model)
        input = torch.cat([t_e, input], dim = 2)
        # Mask out all padded constraints
        input[torch.transpose(constraints_key_padding_mask, 0, 1)] = torch.zeros(4 * self.d_model)
        out = self.fc(input)
        return out