import torch.nn as nn
import torch
import numpy as np

# Encoder
# Self-attention block, Add and Norm block, position-wise feedforward block, encoder block

# print(torch.__version__)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, seq_len, num_heads, batch_size):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, mask):
        q = q.reshape(self.batch_size, self.seq_len, self.num_heads, self.d_k)
        k = k.reshape(self.batch_size, self.seq_len, self.num_heads, self.d_k)
        v = v.reshape(self.batch_size, self.seq_len, self.num_heads, self.d_k)
        print(q.shape)
        product = q @ k.transpose(-2, -1)
        scaled_product = product / np.sqrt(self.d_k)
        if mask:
            scaled_product += mask
        scaled_product = nn.functional.softmax(scaled_product, dim=-1)
        v = scaled_product @ v
        return v

    def forward(self, x, mask=None):
        # batch_size, max_sequence_length, d_model = x.size()
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        x = self.attention(q, k, v, mask)
        x = x.reshape(self.batch_size, self.seq_len, self.num_heads * self.d_k)
        x = self.output(x)
        return x
    

class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdims=True)
        std = torch.sqrt(var + self.eps)
        x = (x - mean) / std
        return self.alpha * x + self.gamma
    

class PositionWiseFeedback(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class EnocoderBlock(nn.Module):
    def __init__(self, d_model, seq_len, num_heads, batch_size, d_ff, drop_prob):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, seq_len, num_heads, batch_size)
        self.feed_forward = PositionWiseFeedback(d_model, d_ff, drop_prob)
        self.layer_norm = LayerNormalization()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        output_1 = self.layer_norm(x + self.dropout(self.multi_head_attention(x, mask=False)))
        output_2 = self.layer_norm(output_1 + self.dropout(self.feed_forward(output_1)))
        return output_2

        
class Encoder(nn.Module):
    def __init__(self, d_model, seq_len, num_heads, batch_size, d_ff, drop_prob, N):
        super().__init__()
        self.N = N
        self.encoder = EnocoderBlock(d_model, seq_len, num_heads, batch_size, d_ff, drop_prob)

    def forward(self, x):
        for _ in range(self.N):
            x = self.encoder(x)
        return x
    

if __name__ == '__main__':
    d_model = 512
    num_heads = 8
    drop_prob = 0.1
    batch_size = 30
    max_sequence_length = 200
    ffn_hidden = 2048
    num_layers = 5

    encoder = Encoder(d_model, max_sequence_length, num_heads, batch_size, ffn_hidden, drop_prob, num_layers)

    x = torch.randn( (batch_size, max_sequence_length, d_model) ) # includes positional encoding
    out = encoder(x)
    print("----------------------------------------------------")
    print(out.shape)
    # print(out[0][0])





