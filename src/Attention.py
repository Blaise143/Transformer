import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from typing import List
import random
from torch.nn.init import xavier_uniform_
import torch.nn.init as init


def set_seed(num: int) -> None:
    """
    Sets a whole bunch of seeds for reproducibility. 
    Will remove the method after testing if the implementation is bug free.

    Args:
        num (int): The seed to be set
    """
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    np.random.seed(num)
    random.seed(num)


set_seed(0)


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

        scores = torch.matmul(Q, K.T)
        scores_normalized = scores / np.sqrt(self.dim)
        softed = nn.Softmax(dim=-1)(scores_normalized)
        z = torch.matmul(softed, V)
        return z


class MultiHeadedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """Performs a multiheaded attention peration on imput data

        Args:
            dim (int): The dimension of the input vectors
            num_heads (int, optional): The number of attention heads. Defaults to 8.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Performs a multiheaded attention operation on input tensor X

        Args:
            X (torch.tensor): The input tensor before a multiheaded attention operation

        Returns:
            torch.tensor: The output tensor after a multiheaded attention operation
        """
        acc = list()

        for _ in range(self.num_heads):
            acc.append(SelfAttention(dim=self.dim)(X))
        attentions = torch.cat(acc, -1)
        print(f"Attentions:\n{attentions}")
        W = nn.Linear(
            in_features=attentions.shape[1], out_features=self.dim)
        out = W(attentions)
        return out


if __name__ == "__main__":
    random_tensor = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 1.], [1, 1, 1, 1], [1, 1, 1, 1]])
    print(random_tensor)
    print(SelfAttention(4)(random_tensor))
    print(MultiHeadedAttention(4)(random_tensor))
