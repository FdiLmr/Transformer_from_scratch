import torch
import torch.nn
import math


class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sine to the even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cosine to the odd positions
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # positional encoding is not trainable
        return self.dropout(x)
    
    
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps # small value to avoid division by zero
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative factor
        self.bias = nn.Parameter(torch.zeros(1)) # Additive factor
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and b2
        
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x)))) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)

        