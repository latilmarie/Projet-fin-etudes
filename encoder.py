# encoder.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiHeadAttention import MultiHeadAttention
from utils import PositionwiseFeedForward


"""
____________________________________________________________________________________
Credits:

The Transformer model has been inspired by the following paper:
'Attention Is All You Need': https://arxiv.org/abs/1706.03762

The implementation applied to time series data has been inspired by the following repository:
https://github.com/maxjcohen/transformer
____________________________________________________________________________________
"""

class Encoder(nn.Module):
    """Encoder block from Attention is All You Need.

    Apply Multi Head Attention block followed by a Point-wise Feed Forward block.
    Residual sum and normalization are applied at each step.

    Parameters
    ----------
    d_model:
        Dimension of the input vector.
    q:
        Dimension of all query matrix.
    v:
        Dimension of all value matrix.
    h:
        Number of heads.
    attention_size:
        Number of backward elements to apply attention.
        Deactivated if ``None``. Default is ``None``.
    dropout:
        Dropout probability after each MHA or PFF block.
    """
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 attention_size: int = None,
                 dropout: float = 0.3):
        """Initialize the Encoder block"""
        super().__init__()
        self._selfAttention = MultiHeadAttention(d_model, q, v, h, attention_size=attention_size)
        self._feedForward = PositionwiseFeedForward(d_model)
        self._layerNorm1 = nn.BatchNorm1d(d_model) # input size (batch_size, d_model, seq_length)
        self._layerNorm2 = nn.BatchNorm1d(d_model)
        self._dopout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input through the Encoder block.

        Apply the Multi Head Attention block, add residual and normalize.
        Apply the Point-wise Feed Forward block, add residual and normalize.

        Parameters
        ----------
        x:
            Input tensor with shape (batch_size, K, d_model).

        Returns
        -------
            Output tensor with shape (batch_size, K, d_model).
        """
        # Self attention
        residual = x
        x = self._selfAttention(query=x, key=x, value=x)
        x = self._dopout(x)
        
        x = x.permute(0, 2, 1) # input size for norm (batch_size, d_model, seq_length)
        residual = residual.permute(0, 2, 1)
        
        x = self._layerNorm1(x + residual)
        
        x = x.permute(0, 2, 1)
        residual = residual.permute(0, 2, 1)

        # Feed forward
        residual = x
        x = self._feedForward(x)
        x = self._dopout(x)
        
        x = x.permute(0, 2, 1)
        residual = residual.permute(0, 2, 1)
        x = self._layerNorm2(x + residual)
        x = x.permute(0, 2, 1)
        residual = residual.permute(0, 2, 1)

        return x

    @property
    def attention_map(self) -> torch.Tensor:
        """Attention map after a forward propagation,
        variable `score` in the original paper.
        """
        return self._selfAttention.attention_map