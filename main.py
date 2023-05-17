import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torchsummary import summary
from typing import List
import random
from torch.nn.init import xavier_uniform_


def set_seed(num: int) -> None:
    """Sets a whole bunch of seeds for reproducibility

    Args:
        num (int): The seed to be set
    """
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    np.random.seed(num)
    random.seed(num)


set_seed(42)


class SelfAttention(nn.Module):
    """
    An implementation of self attention in pytorch
    """

    def __init__(self, dim: int) -> None:
        """Takes in a tensor and spits out a tensor of the same dimension after undergoing self attention

        Args:
            input_dim (torch.Tensor): The input dimension of the embedding
        """
        super(SelfAttention, self).__init__()
        self.dim = dim
        ones = torch.ones(dim, dim)

        self.Q = nn.Linear(dim, dim, bias=False)
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """The input tensor

        Args:
            X (tensor): the input

        Returns:
            Tensor: an output after self attention is acted on it
        """

        Q = self.Q(X)
        K = self.K(X)
        V = self.V(X)

        # multihead_attn = nn.MultiheadAttention(self.dim, num_heads=8, bias=False)
        # z = multi_head()
        # attn_output, attn_output_weights = multihead_attn(Q, K, V)

        scores = torch.matmul(Q, K.T)
        scores_normalized = scores / np.sqrt(self.dim)
        softed = nn.Softmax(dim=-1)(scores_normalized)
        z = torch.matmul(softed, V)
        return z


class MultiHeadedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        """An Implementation of Multi headed attention

        Args:
            dim (int): The dimention of the embedding
            num_heads (int, optional): The number of self attention heads. Defaults to 8.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

    def forward(self, X):
        # print(f"Xshape: {X.shape}")
        attentions = list()
        for _ in range(self.num_heads):
            attentions.append(SelfAttention(self.dim)(X))
        out = torch.cat(attentions, -1)
        print(f"Printing out: {out}")
        layer = nn.Linear(out.shape[1], self.dim, bias=False)
        ones = torch.ones(self.dim, out.shape[1])
        # layer.weight = nn.Parameter(ones)
        print(f"Printing layer {layer}")
        out = layer(out)
        return out


if __name__ == "__main__":
    random_tensor = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1.], [1, 1, 1, 1], [1, 1, 1, 1]])
    print(random_tensor)
    print(SelfAttention(4).forward(random_tensor))
