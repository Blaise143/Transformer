import torch
import torch.nn as nn

a = torch.tril(torch.ones((5, 5)))
b = torch.masked_fill(a, mask=a == 0, value=float("-inf"))
print(a)
print(b)
softed = nn.Softmax()(b)
print(softed)
