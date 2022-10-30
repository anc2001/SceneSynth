import torch
from torch import nn, Tensor
import math

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))

def save_checkpoint():
    pass

def load_checkpoint():
    pass

def optimizer_factory(model, type, lr):
    if type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, eps=1e-6)
    elif type == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError('Optimizer not supported')

def loss_factory(type):
    if type == 'cross_entropy':
        return torch.nn.CrossEntropyLoss(reduction='none')
    else:
        raise ValueError('Loss not supported')

def generate_square_subsequent_mask(sz, device):
    # sz: sequence length
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

# For object encoder 
class FixedPositionalEncoding(nn.Module):
    def __init__(self, proj_dims, val=0.1):
        super().__init__()
        ll = proj_dims//2
        exb = 2 * torch.linspace(0, ll-1, ll) / proj_dims
        self.sigma = 1.0 / torch.pow(val, exb).view(1, -1)
        self.sigma = 2 * torch.pi * self.sigma

    def forward(self, x):
        return torch.cat([
            torch.sin(x * self.sigma.to(x.device)),
            torch.cos(x * self.sigma.to(x.device))
        ], dim=-1)

# For positional encoding of transformer inputs 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.dropout(x + self.pe[:x.size(0)])
    
    def encode_single(self, x : Tensor, idx : int):
        return self.dropout(x + self.pe[idx])