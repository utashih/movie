import torch 
import torch.nn as nn 
import torch.nn.functional as F
from IPython import embed

class DeepSet(nn.Module):
    def __init__(self, phi: nn.Module, rho: nn.Module):
        super().__init__()
        self.phi = phi 
        self.rho = rho 

    def forward(self, x):
        x = self.phi(x)
        x = torch.sum(x, dim=1, keepdim=False)
        out = self.rho(x)
        return out 

HIDDEN_DIM = 100
    
class MoviePhi(nn.Module):
    def __init__(self, num_cast):
        super().__init__()
        self.num_cast = num_cast 
        self.emb = nn.Embedding(self.num_cast, HIDDEN_DIM)
        self.fc1 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
    
    def forward(self, x):
        x = self.emb(x)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        out = torch.tanh(x)
        return out 

class MovieRho(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(HIDDEN_DIM, 12)
        self.fc2 = nn.Linear(12, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        
        out = self.fc2(x)
        return out.reshape(-1)

    
