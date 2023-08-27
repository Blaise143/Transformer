import torch
import torch.nn as nn
from torch import sin, cos
# import torch.sin as sin
# embedding = nn.Embedding(num_embeddings=6, embedding_dim=10)
# input_data = torch.tensor([[2, 1, 2, 3], [2, 3, 1, 1]])

# print(embedding(input_data).shape)
# a = 4*torch.ones(4, 32)
# b = torch.ones(32, 4)

# print(a)
# print(b)
# print(torch.matmul(a, b))

a = torch.ones(1, 5)[0]*2*torch.pi
b = torch.ones(1, 5) * 2*torch.pi
print(cos(b))
print(sin(a))
# print(sin(90))
print(a)
