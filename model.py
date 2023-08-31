import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    # convert the raw words into an embedded matrix
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(d_model, vocab_size)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    # add positional/contextual data to the input through the formula given in paper
    def __init__(self, d_model: int, seq_length: int, dropout: torch.float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        positional_matrix = torch.zeroes(self.seq_length, self.d_model)
        position = torch.arange(0, seq_length-1, dtype=torch.float).unsqueeze(-1) # shape --> (seq_length, 1)
        for pos in position:
            for i in range(d_model):
                if i % 2 == 0:
                    # apply sine to even indices
                    positional_matrix[pos][i] = torch.sin(pos/self.den(i, self.d_model))
                else:
                    # apply cosine to odd indices
                    positional_matrix[pos][i] = torch.cos(pos/self.den(i, self.d_model))

        positional_matrix = positional_matrix.unsqueeze(0) # shape --> (batch_size=1, seq_length, d_model)
        # save the positional_matrix values along with the file as it will never change. 
        self.register_buffer("positional_matrix", positional_matrix)  

    def den(i: int, d_model: int):
        # denominator for the positional expression as given in paper
        return 10000**(2*i/d_model)

    def forward(self, x):
        # add the positional information to input data
        x = x + self.positional_matrix[:, :x.shape[1], :].requires_grad_(False) # shape --> (batch, seq_length, d_model)
        return self.dropout(x)
    

class LayerNorm(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        # nn.Parameter() enables variables to become a learnable parameter
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # normalize, scale and shift the input using alpha and beta
        mean = x.mean(dim=-1, keepdims=True)
        std = x.std(dim=-1, keepdims=True)
        norm = (x-mean) / (std + self.eps)
        return self.alpha * norm + self.beta
    

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: int):
        super().__init__()
        self.linear1 = nn.Linear1(d_model, d_ff) # w1 and b1
        self.linear2 = nn.Linear2(d_ff, d_model) # w2 and b2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # apply feedforward layer to the input
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
    