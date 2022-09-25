# Initialize Room list
# Initialize constraint superset 
# Initialize variables from config

import torch
from main.config import get_config

config = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')