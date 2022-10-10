from main.config import constraint_types, object_types, direction_types

# Full implementation of the model end to end 
import torch
from torch import nn, Tensor
from object_encoder import ObjectEncoderModel
from program_line_encoder import ProgramLineEncoderModel
from program_line_decoder import ProgramLineDecoderModel
from utils import PositionalEncoding, generate_square_subsequent_mask

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
        self.constraint_encoder = ProgramLineEncoderModel(
            d_model=self.d_model,
            num_constraint_categories=len(constraint_types) + 1,
            num_directions=len(direction_types) + 1
        )

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

        self.positional_encoding = PositionalEncoding(self.d_model)
        self.program_line_decoder = ProgramLineDecoderModel(self.d_model)
        self.loss_fnc = loss_func

    def inference(self, objects : list) -> Tensor:
        src_e = self.object_encoder(
            torch.unsqueeze(torch.tensor(objects).float(), 1), 
            torch.zeros(1, len(objects)).bool()
        )
        memory = self.transformer_encoder(src_e)
        
        tgt = torch.tensor([[[4, 0, 0, 0]]]).float()
        tgt_e = self.constraint_encoder(
            tgt, 
            torch.zeros(1, 1).bool(),
            src_e
        )
        tgt_e = self.positional_encoding(tgt_e)
        while len(tgt) < 10:
            decoded_output = self.transformer_decoder(
                tgt_e, 
                memory,
            )
            constraint = self.program_line_decoder.inference(decoded_output, src_e, self.constraint_encoder)
            if constraint[0] == 5:
                break # End of sequence symbol detected 
            tgt = torch.cat([tgt, torch.tensor([[constraint]])], dim=0)
            tgt_e = self.constraint_encoder(
                tgt,
                torch.zeros(1, len(tgt)).bool(),
                src_e
            )
            tgt_e = self.positional_encoding(tgt_e)
        
        tgt = torch.cat([tgt, torch.tensor([[[5, 0, 0, 0]]])], dim=0)
        return tgt 

    def forward(
        self, 
        src : Tensor, 
        src_key_padding_mask : Tensor, 
        tgt : Tensor, 
        tgt_key_padding_mask : Tensor,
    ):
        # src: object sequence [object_sequence_length, batch_size, object_representation_length]
        # src_key_padding_mask: object sequence padding mask [batch_size, object_sequence_length]
        # tgt: constraint sequence [constraint_sequence_length, batch_size, constraint_representation_length]
        # tgt_key_padding_mask: constraint sequence padding mask [batch_size, constraint_sequence_length]

        src_e = self.object_encoder(src, src_key_padding_mask)
        src_e = self.positional_encoding(src_e)

        memory = self.transformer_encoder(
            src_e, 
            src_key_padding_mask=src_key_padding_mask
        )

        tgt_e = self.constraint_encoder(
            tgt, 
            tgt_key_padding_mask, 
            src_e
        )

        tgt_e = self.positional_encoding(tgt_e)
        tgt_mask = generate_square_subsequent_mask(tgt.size()[0])
        tgt_mask = torch.unsqueeze(tgt_mask, dim = 0)
        tgt_mask = tgt_mask.expand(src.shape[1] * self.nhead, -1, -1)

        decoded_output = self.transformer_decoder(
            tgt_e, 
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        preds = self.program_line_decoder(
            decoded_output, 
            tgt, 
            src_e, 
            src_key_padding_mask,
            self.constraint_encoder
        )
        return preds

    def loss(self, preds, tgt, tgt_key_padding_mask):
        # preds: [constraint_sequence_length, batch_size, num_constraint_categories]
        # tgt: [constraint_sequence_length, batch_size, constraint_representation_length]
        # tgt_key_padding_mask: [batch_size, constraint_sequence_length]

        # Prediction labels should be the next constraint 
        labels = torch.roll(tgt, -1, dims = 0)
        types, objects, directions = preds

        x_types = torch.flatten(types, start_dim = 0, end_dim = 1)
        y_types = torch.flatten(labels[:, :, 0], start_dim = 0, end_dim = 1)
        types_loss = self.loss_fnc(x_types, y_types.long())

        x_objects = torch.flatten(objects, start_dim = 0, end_dim = 1)
        y_objects = torch.flatten(labels[:, :, 2], start_dim = 0, end_dim = 1)
        objects_loss = self.loss_fnc(x_objects, y_objects.long())

        x_directions = torch.flatten(directions, start_dim = 0, end_dim = 1)
        y_directions = torch.flatten(labels[:, :, 3], start_dim = 0, end_dim = 1)
        directions_loss = self.loss_fnc(x_directions, y_directions.long())

        # Mask out padding loss + end constraint for all 
        # Padding and the end constraint predictions should not contribute to the loss 
        padding_end_mask = ~torch.flatten(
            torch.logical_or(
                torch.transpose(tgt_key_padding_mask, 0, 1), 
                tgt[:, :, 0] == 5
            ),
            start_dim = 0, 
            end_dim = 1
        )
        types_loss *= padding_end_mask
        objects_loss *= padding_end_mask
        directions_loss *= padding_end_mask
        # Mask out object loss and direction loss for the index predicting the end constraint 
        predicting_end_mask = ~torch.flatten(
            labels[:, :, 0] == 5,
            start_dim = 0,
            end_dim = 1
        )
        objects_loss *= predicting_end_mask
        directions_loss *= predicting_end_mask

        # Mask out direction loss for all places that are supposed to predict orientation constraints 
        tgt_orientation_mask = ~torch.flatten(
            torch.logical_or(
                labels[:, :, 0] == 2,
                labels[:, :, 0] == 3
            ),
            start_dim = 0,
            end_dim = 1
        )
        directions_loss *= tgt_orientation_mask

        loss = types_loss + objects_loss + directions_loss

        return torch.mean(loss)
    
    def accuracy_fnc(self, preds, tgt, tgt_key_padding_mask):
        # preds: [constraint_sequence_length, batch_size, num_constraint_categories]
        # tgt: [constraint_sequence_length, batch_size, constraint_representation_length]
        # tgt_key_padding_mask: [batch_size, constraint_sequence_length]

        # Measure total correct predictions per module 
        total_correct = 0
        total_tokens = 0

        labels = torch.roll(tgt, -1, dims = 0)        
        types, objects, directions = preds
        
        padding_end_mask = ~torch.logical_or(
            torch.transpose(tgt_key_padding_mask, 0, 1), 
            tgt[:, :, 0] == 5
        )
        predicting_end_mask = ~(labels[:, :, 0] == 5)
        tgt_orientation_mask = ~torch.logical_or(
            labels[:, :, 0] == 2,
            labels[:, :, 0] == 3
        )

        decoded_types = torch.argmax(types, dim = 2)
        type_mask = padding_end_mask
        decoded_objects = torch.argmax(objects, dim = 2)
        object_mask = padding_end_mask * predicting_end_mask
        decoded_directions = torch.argmax(directions, dim = 2)
        direction_mask = padding_end_mask * predicting_end_mask * tgt_orientation_mask

        total_tokens += torch.sum(type_mask).item()
        total_correct += torch.sum((decoded_types == labels[:, :, 0]) * type_mask).item()
        total_tokens += torch.sum(object_mask).item()
        total_correct += torch.sum((decoded_objects == labels[:, :, 2]) * object_mask).item()
        total_tokens += torch.sum(direction_mask).item()
        total_correct += torch.sum((decoded_directions == labels[:, :, 3]) * direction_mask).item()

        return total_correct / total_tokens