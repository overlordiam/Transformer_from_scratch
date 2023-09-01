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
        self.shared_weights = self.embedding.weight

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model), self.shared_weights



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
    

class MutliHeadAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h

        # check whether the tensor can be split into h equal parts
        assert d_model % h == 0, "d_model not divisible by h"
        
        self.d_k = d_model // h

        # initialize weights for query, key and value
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # calculate the attention for the given chunks of query, key and value
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        # apply -1e9 to any position of tensor where value == 0
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # apply softmax to the last dimension of input
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores        

    def forward(self, query, key, value, mask):
        
        # calculate weighted values of query, value and key 
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        # split the q, k and v values into h parts and apply attention
        query = query.view(query.shape[0], self.h, query.shape[1], self.d_k)
        key = query.view(key.shape[0], self.h, key.shape[1], self.d_k)
        value = value.view(value.shape[0], self.h, value.shape[1], self.d_k)

        # calculate the word representation with position and context (attention)
        x, self.attention_scores = MutliHeadAttention().attention(query, key, value, mask, self.dropout)

        # combine the heads to form the original matrix
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h*self.d_k) # shape -> (batch, seq_len, d_model)

        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm()

    def forward(self, x, sublayer):
        # output of sub layer = LayerNorm(x + Sublayer(x)) 
        return self.dropout(self.norm(x + sublayer(x)))
    

class EncoderBlock(nn.Module):
    # Assemble both the sub layers and apply layer norm to finalize the encoder block
    def __init__(self, self_attention_block: MutliHeadAttention, feed_forward_block:FeedForward, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # We apply lambda function so that the MutliHeadAttention object executes inside the residual function,
        # which gives data flow. The feed_forward_block does not need the input as it feeds it inside the ResidualConnection class 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    # handles the N stacks of the encoder. Output of each stack is fed in as the input to the next stack.
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        
    
class DecoderBlock(nn.Module):
    # assemble a self attention layer, a cross attention layer and a feed forward layer to form the decoder block
    def __init__(self, self_attention_block: MutliHeadAttention, cross_attention_block: MutliHeadAttention, feed_forward: FeedForward, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.residual_connections = nn.ModuleList([ResidualConnection(self.dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, trg_mask):
        # similar to the encoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, trg_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward)
        return x
    

class Decoder(nn.Module):
    # assemble N stacks of the decoder block
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)
    

class Projection(nn.Module):
    # final layer that maps the output (last dimension -> d_model) from the decoder to the the vocabulary
    def __init__(self, d_model, vocab_size, shared_weights):
        # the linear layer uses the same weights as the embedding layer
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.linear.weight = shared_weights

    def forward(self, x):
        x = self.linear(x)
        # returns the probability distribution of every word through the softmax function
        return torch.log_softmax(x, dim=-1)

    