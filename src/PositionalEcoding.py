import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
