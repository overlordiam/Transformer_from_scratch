import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    # convert the raw words into an embedded matrix
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)



class PositionalEmbedding(nn.Module):
    # add positional/contextual data to the input through the formula given in paper
    def __init__(self, d_model: int, seq_length: int, dropout: torch.float):
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(-1) # shape --> (seq_length, 1)
        even_index = torch.arange(0, d_model, 2)
        denominator = self.den(even_index, d_model)
        even_pos, odd_pos = torch.sin(position, denominator), torch.cos(position, denominator)
        positional_matrix = torch.stack([even_pos, odd_pos], dim=-1)
        positional_matrix = torch.flatten(positional_matrix, start_dim=-2, end_dim=-1)
        positional_matrix = positional_matrix.unsqueeze(0) # shape --> (batch_size=1, seq_length, d_model)
        # save the positional_matrix values along with the file as it will never change. 
        self.register_buffer("positional_matrix", positional_matrix)  

    def den(self, i: int, d_model: int):
        # denominator for the positional expression as given in paper
        return 10000**(i/d_model)

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
    

class MultiHeadAttention(nn.Module):
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
            # mask_ = torch.tril(query.shape[-2], query.shape[-2])
            # mask_[mask_ == 0] = -1e9
            # mask_[mask_ == 1] = 0
            # res = attention_scores + mask_.unsqueeze(0)
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
        x, self.attention_scores = MultiHeadAttention().attention(query, key, value, mask, self.dropout)

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
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block:FeedForward, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # We apply lambda function so that the MultiHeadAttention object executes inside the residual function,
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
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward: FeedForward, dropout: float):
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

    
class Transformer(nn.Module):
    # assemble the transformer which includes the 2 main components: encoder and decoder

    def __init__(self, encoder: Encoder, decoder: Decoder, enc_embedding: InputEmbedding, 
                 dec_embedding: InputEmbedding, enc_pos_embedding: PositionalEmbedding, 
                 dec_pos_embedding: PositionalEmbedding, projection: Projection):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_embedding = enc_embedding
        self.dec_embedding = dec_embedding
        self.enc_pos_embedding = enc_pos_embedding
        self.dec_pos_embedding = dec_pos_embedding
        self.projection = projection

    def encode(self, src, src_mask):
        # initialize the encoder
        src = self.enc_embedding(src)
        src = self.enc_pos_embedding(src)
        src = self.encoder(src, src_mask)
        return src

    def decode(self, trg, encoder_output, src_mask, trg_mask):
        # initialize the decoder
        trg = self.dec_embedding(trg)
        trg = self.dec_pos_embedding(trg)
        trg = self.decoder(trg, encoder_output, src_mask, trg_mask)
        return trg
    
    def project(self, x):
        # final layer that provides output indices
        return self.projection(x)
    

def build_transformer(src_vocab: int, trg_vocab: int, src_seq_length: int, trg_seq_length: int,
                      h: int = 68, d_model: int = 512, N: int = 6, dropout: float = 0.1,
                      d_ff: int = 2048):
    # build the transformer
    
    src_embedding = InputEmbedding(src_vocab, d_model)
    shared_weights = src_embedding.weight # weights shared with other embedding layer and final linear layer
    trg_embedding = InputEmbedding(trg_vocab, d_model)
    trg_embedding.weight = shared_weights

    src_pos_embedding = PositionalEmbedding(d_model, src_seq_length, dropout)
    trg_pos_embedding = PositionalEmbedding(d_model, trg_seq_length, dropout)

    # build stack of N encoders 
    encoder_blocks = []
    for _ in range(N):
        self_attention_encoder = MultiHeadAttention(h, d_model, dropout)
        feed_forward_encoder = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_encoder, feed_forward_encoder, dropout)
        encoder_blocks.append(encoder_block)

    # build stack of N decoders
    decoder_blocks = []
    for _ in range(N):
        self_attention_decoder = MultiHeadAttention(h, d_model, dropout)
        cross_attention_decoder = MultiHeadAttention(h, d_model, dropout)
        feed_forward_decoder = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_decoder, cross_attention_decoder, 
        feed_forward_decoder, dropout)
        decoder_blocks.append(decoder_block)

    # assemble encoder and decoder stack
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection = Projection(d_model, trg_vocab, shared_weights)

    transformer = Transformer(encoder, decoder, src_embedding, trg_embedding, src_pos_embedding, 
                              trg_pos_embedding, projection)

    # initialize weights with xavier's
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

    