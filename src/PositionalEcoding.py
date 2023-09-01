import torch
import math
import matplotlib.pyplot as plt


class PositionalEncoding:
    def __init__(self, dim: int, max_position: int) -> None:
        """Implements a sinusoidal positional encoding

        Args:
            dim (int): The dimension of the input embeddings
            max_position (int): The maximun number of positions (the number of embeddings)
        """
        self.dim = dim
        self.max_position = max_position

    def generate(self) -> torch.tensor:
        """Generates the positional encoding

        Returns:
            torch.tensor: The positional encodings genetated
        """
        positions = torch.arange(0, self.max_position).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2)
                             * - math.log(10_000) / self.dim)
        sin_embedding = torch.sin(positions*div_term)
        cos_embedding = torch.cos(positions*div_term)

        embedding = torch.stack((sin_embedding, cos_embedding), dim=2)
        embedding = embedding.view(self.max_position, -1)
        return embedding


if __name__ == "__main__":
    p_e = PositionalEncoding(dim=2000, max_position=2000)
    plt.matshow(p_e.generate())
    plt.show()
