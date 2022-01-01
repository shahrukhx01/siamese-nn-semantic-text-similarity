"""
boilerplate borrowed from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.autograd import Variable
from siamese_sts.utils.utils import similarity_score


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size)
        )
        pe = torch.zeros(max_len, 1, embedding_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SiameseTransformer(nn.Module):
    def __init__(
        self,
        batch_size,
        vocab_size: int,
        embedding_size: int,
        nhead: int,
        hidden_size: int,
        transformer_layers: int,
        embedding_weights: torch.Tensor,
        device: str,
        max_sequence_len: int,
        lstm_layers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size
        self.device = device
        self.pos_encoder = PositionalEncoding(embedding_size, dropout, max_sequence_len)
        encoder_layers = TransformerEncoderLayer(
            embedding_size, nhead, hidden_size, dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, transformer_layers
        )
        self.encoder = nn.Embedding(vocab_size, embedding_size)

        # assigning the look-up table to the pre-trained fasttext word embeddings.
        self.encoder.weight = nn.Parameter(
            embedding_weights.to(self.device), requires_grad=True
        )
        self.embedding_size = embedding_size
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=lstm_layers)

    def forward_once(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, vocab_size]
        """
        src = self.encoder(src) * math.sqrt(self.embedding_size)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src)
        # permute embedded input to shape = (sequence_len, batch_size, embedding_size)
        embeddings = output.permute(1, 0, 2)

        # perform forward pass of LSTM
        output, (final_hidden_state, final_cell_state) = self.lstm(
            embeddings, self.hidden
        )

        return final_hidden_state[-1]

    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """
        return (
            Variable(
                torch.zeros(self.lstm_layers, batch_size, self.hidden_size).to(
                    self.device
                )
            ),
            Variable(torch.zeros(self.lstm_layers, batch_size, self.hidden_size)).to(
                self.device
            ),
        )

    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths) -> Tensor:
        ## init context and hidden weights for lstm cell
        self.hidden = self.init_hidden(sent1_batch.size(0))
        sent1_out = self.forward_once(sent1_batch)
        sent2_out = self.forward_once(sent2_batch)

        return similarity_score(sent1_out, sent2_out)
