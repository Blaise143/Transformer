import torch.nn as nn
import torch
from Attention import MultiHeadedAttention


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8) -> None:
        """The econder block of the Transformer 

        Args:
            dim (int): The dimension of the input vectors
            num_heads (int): The number of heads of the multiheaded attention, defaults to 8
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Forward pass throught the network

        Args:
            X (torch.tensor): The input tensor

        Returns:
            torch.tensor: The tensor after passign through the encoder block
        """
        x_cache = X
        x = MultiHeadedAttention(self.dim, self.num_heads)(X)
        x = x_cache+x
        x = nn.LayerNorm(self.dim)(x)

        x_cache = x
        fcn = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
            nn.ReLU()
        )
        x = fcn(x)
        x = x+x_cache
        return x


class Encoder(nn.Module):
    """
    stacks identity encoders together. 6 autoencoders are used here as in the original Transformers paper
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.encoders = nn.Sequential(
            EncoderBlock(dim=dim, num_heads=num_heads),
            EncoderBlock(dim=dim, num_heads=num_heads),
            EncoderBlock(dim=dim, num_heads=num_heads),
            EncoderBlock(dim=dim, num_heads=num_heads),
            EncoderBlock(dim=dim, num_heads=num_heads),
            EncoderBlock(dim=dim, num_heads=num_heads)
        )

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Peforms a forward pass through the network

        Args:
            X (torch.tensor): Input tensor

        Returns:

        """
        x = self.encoders(X)
        return x


if __name__ == "__main__":
    tensor = torch.rand(9, 5)
    print(EncoderBlock(5)(tensor).shape)
    print("Full Encoders")
    print(Encoder(5)(tensor).shape)
    # print(tensor)
