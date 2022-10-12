import torch
import torch.nn as nn
from main.network.utils import FixedPositionalEncoding

class ObjectEncoderModel(nn.Module):
    def __init__(self, d_model, num_obj_categories):
        # d_model: dimension of the model
        # num_obj_categories: number of object categories
        super().__init__()
        self.d_model = d_model
        self.num_obj_categories = num_obj_categories
        self.hidden_dim = d_model // 2

        self.embedding = nn.Embedding(num_obj_categories, self.hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(1 + self.hidden_dim * 6, 2 * self.d_model),
            nn.ReLU(),
            nn.Linear(2 * self.d_model, self.d_model)
        )
        
        self.pe_size_x = FixedPositionalEncoding(proj_dims=self.hidden_dim)
        self.pe_size_y = FixedPositionalEncoding(proj_dims=self.hidden_dim)
        self.pe_pos_x = FixedPositionalEncoding(proj_dims=self.hidden_dim)
        self.pe_pos_y = FixedPositionalEncoding(proj_dims=self.hidden_dim)
        self.pe_rot = FixedPositionalEncoding(proj_dims=self.hidden_dim)

    def forward(self, objs):
        # objs: [object_sequence_length, batch_size, num_obj_categories]
        # obj_key_padding_mask: [batch_size, object_sequence_length]
        # return: [object_sequence_length, batch_size, d_model]

        input = torch.cat(
            [
                self.embedding((objs[:, :, 0]).int()), 
                self.pe_size_x(objs[:, :, 1:2]),
                self.pe_size_y(objs[:, :, 2:3]),
                self.pe_pos_x(objs[:, :, 3:4]),
                self.pe_pos_y(objs[:, :, 4:5]),
                self.pe_rot(objs[:, :, 5:6]),
                objs[:, :, 6:7],
            ],
            dim = 2
        )
        out = self.fc(input)
        return out