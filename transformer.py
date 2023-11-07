import torch.nn as nn
import torch
import numpy as np

# Encoder
# Self-attention block, Add and Norm block, position-wise feedforward block, encoder block

# print(torch.__version__)

class InputEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        x = self.embedding(x)
        return x / np.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        position = torch.arange(max_seq_len).reshape(max_seq_len, 1)
        even_index = torch.arange(0, d_model, 2).float()
        denominator = torch.pow(10000, even_index/d_model)
        odd_pos, even_pos = torch.cos(position / denominator), torch.sin(position / denominator)
        positional_encoding = torch.stack([even_pos, odd_pos], dim=2)
        self.positional_encoding = torch.flatten(positional_encoding, start_dim=1, end_dim=2)

    def forward(self, x):
        return x + self.positional_encoding
    

class EmbeddedInput(nn.Module):
    def __init__(self, d_model, max_seq_len, word_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, drop_prob):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_embedding = InputEmbedding(max_seq_len, d_model)
        self.positional_encoding = PositionalEncoding(max_seq_len, d_model)
        self.word_to_index = word_to_index
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.dropout = nn.Dropout(drop_prob)

    def batch_tokenize(self, batch, start_token, end_token):
        def tokenize(sentence, start_token, end_token):
            tokenized_sentence = [self.word_to_index(word) for word in list(sentence)]
            if start_token:
                tokenized_sentence.insert(0, self.word_to_index[self.START_TOKEN])
            if end_token:
                tokenized_sentence.append(self.word_to_index[self.END_TOKEN])
            for _ in range(len(sentence), self.max_seq_len - 1):
                tokenized_sentence.append(self.word_to_index[self.PADDING_TOKEN])
            return torch.tensor(tokenized_sentence)
        
        tokenized_sentences = []
        for num in range(len(batch)):
            sentence = tokenize(batch[num], start_token, end_token)
            tokenized_sentences.append(sentence)
        print("shape of tokenized sentence before stacking: ", tokenized_sentences.shape)
        tokenized_sentences = torch.stack(tokenized_sentences)
        print("shape of tokenized sentence after stacking: ", tokenized_sentences.shape)
        return tokenized_sentences


    def forward(self, x, start_token, end_token):
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.input_embedding(x, start_token, end_token)
        x = self.positional_encoding(x)
        return self.dropout(x)


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

    def attention(self, query, key, value, mask):
        q, k, v = self.w_q(query), self.w_k(key), self.w_v(value)
        q = q.reshape(self.batch_size, self.seq_len, self.num_heads, self.d_k)
        k = k.reshape(self.batch_size, self.seq_len, self.num_heads, self.d_k)
        v = v.reshape(self.batch_size, self.seq_len, self.num_heads, self.d_k)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        print(q.shape)
        product = q @ k.transpose(-1, -2)
        scaled_product = product / np.sqrt(self.d_k)
        if mask is not None:
            # print(f"-------SHAPE OF MASK-------{mask.size()}")
            # print(f"-------SHAPE OF SCALED PRODUCT------{scaled_product.size()}")
            scaled_product += mask
        scaled_product = nn.functional.softmax(scaled_product, dim=-1)
        v = scaled_product @ v
        return v

    def forward(self, q, k, v, mask=None):
        # batch_size, max_sequence_length, d_model = x.size()
        print(q.size(), k.size(), v.size())
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
        self.positionwise_feed_forward = PositionWiseFeedback(d_model, d_ff, drop_prob)
        self.layer_norm = LayerNormalization()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, self_att_mask):
        output_1 = self.layer_norm(x + self.dropout(self.multi_head_attention(x, x, x, mask=self_att_mask)))
        output_2 = self.layer_norm(output_1 + self.dropout(self.positionwise_feed_forward(output_1)))
        return output_2

        
class EncoderSequence(nn.Module):
    def __init__(self, d_model, max_seq_len, num_heads, batch_size, d_ff, drop_prob, N):
        super().__init__()
        self.N = N
        self.encoder = EnocoderBlock(d_model, max_seq_len, num_heads, batch_size, d_ff, drop_prob)

    def forward(self, x, self_att_mask):
        for _ in range(self.N):
            x = self.encoder(x, self_att_mask)
        return x

    
class Encoder(nn.Module):
    def __init__(self, d_model, max_seq_len, num_heads, batch_size, d_ff, drop_prob, N, word_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.embedded_input = EmbeddedInput(d_model, max_seq_len, word_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, drop_prob)
        self.embedding_sequence = EncoderSequence(d_model, max_seq_len, num_heads, batch_size, d_ff, drop_prob, N)
    
    def forward(self, x, self_att_mask, start_token, end_token):
        x = self.embedded_input(x, start_token, end_token)
        x = self.embedding_sequence(x, self_att_mask)
        return x
        


class DecoderBlock(nn.Module):
    def __init__(self, d_model, seq_len, num_heads, batch_size, d_ff, drop_prob):
        super().__init__()
        self.multi_head_self_attention = MultiHeadAttention(d_model, seq_len, num_heads, batch_size)
        self.layer_norm_1 = LayerNormalization()
        self.dropout_1 = nn.Dropout(drop_prob)

        self.multi_head_cross_attention = MultiHeadAttention(d_model, seq_len, num_heads, batch_size)
        self.layer_norm_2 = LayerNormalization()
        self.dropout_2 = nn.Dropout(drop_prob)

        self.positionwise_feed_forward = PositionWiseFeedback(d_model, d_ff, drop_prob)
        self.layer_norm_3 = LayerNormalization()
        self.dropout_3 = nn.Dropout(drop_prob)
        
        
    def forward(self, x, encoder_output, self_att_mask, cross_att_mask):
        out_1 = self.multi_head_self_attention(x, x, x, mask=self_att_mask)
        x = self.layer_norm_1(x + self.dropout_1(out_1))
        out_2 = self.multi_head_cross_attention(x, encoder_output, encoder_output, mask=cross_att_mask)
        x = self.layer_norm_2(x + self.dropout_2(out_2))
        out_3 = self.positionwise_feed_forward(x)
        x = self.layer_norm_3(x + self.dropout_3(out_3))
        return x


class DecoderSequence(nn.Module):
    def __init__(self, d_model, seq_len, num_heads, batch_size, d_ff, drop_prob, N):
        super().__init__()
        self.decoder = DecoderBlock(d_model, seq_len, num_heads, batch_size, d_ff, drop_prob)
        self.N = N

    def forward(self, x, encoder_output, self_att_mask, cross_att_mask):
        for _ in range(self.N):
            x = self.decoder(x, encoder_output, self_att_mask, cross_att_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, max_seq_len, num_heads, batch_size, d_ff, drop_prob, N, START_TOKEN, END_TOKEN, PADDING_TOKEN, word_to_index):
        super().__init__()
        self.embedded_input = EmbeddedInput(d_model, max_seq_len, word_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, drop_prob)
        self.decoder_sequence = DecoderSequence(d_model, max_seq_len, num_heads, batch_size, d_ff, drop_prob, N)

    def forward(self, x, encoder_output, self_att_mask, cross_att_mask, start_token, end_token):
        x = self.embedded_input(x, start_token, end_token)
        x = self.decoder_sequence(x, encoder_output, self_att_mask, cross_att_mask)
        return x
    

class Transformer(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, ):
        pass


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
    y = torch.randn( (batch_size, max_sequence_length, d_model) ) 
    out = encoder(x)
    print("----------------------------------------------------")
    print(out.shape)
    # print(out[0][0])

    print("Decoder ----------------------------------------------------")
    decoder = Decoder(d_model, max_sequence_length, num_heads, batch_size, ffn_hidden, drop_prob, out, num_layers)
    final = decoder(y)
    # print(final)





