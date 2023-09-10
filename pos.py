import torch

def pos(seq_length=6, d_model=10):
    position = torch.arange(seq_length).unsqueeze(-1)
    even_index = torch.arange(0, d_model, 2)
    even_den = torch.pow(10000, even_index / d_model)
    # print("even den: ", even_den)
    odd_index = torch.arange(1, d_model, 2) 
    odd_den = torch.pow(10000, (odd_index - 1) / d_model)
    # print("odd den: ", odd_den)
    even_pos = torch.sin(position / even_den)
    # print(even_pos)
    odd_pos = torch.cos(position / odd_den)
    # print(odd_pos)
    positional_mat = torch.stack([even_pos, odd_pos], dim=-1)
    positional_mat = torch.flatten(positional_mat, start_dim=-2, end_dim=-1)
    print(positional_mat.shape)

pos()