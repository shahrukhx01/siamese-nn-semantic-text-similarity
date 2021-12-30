import torch
import torch.nn as nn
from torch.autograd import Variable
from siamese_sts.utils.utils import similarity_score

"""
Wrapper class using Pytorch nn.Module to create the architecture for our 
binary classification model
"""


class SiameseLSTM(nn.Module):
    def __init__(
        self,
        batch_size: int,
        output_size: int,
        hidden_size: int,
        vocab_size: int,
        embedding_size: int,
        embedding_weights: torch.TensorType,
        lstm_layers: int,
        device: str,
    ):
        super(SiameseLSTM, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_layers = lstm_layers
        self.device = device

        ## model layers
        # initializing the look-up table.
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        # assigning the look-up table to the pre-trained fasttext word embeddings.
        self.word_embeddings.weight = nn.Parameter(
            embedding_weights.to(self.device), requires_grad=True
        )

        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=lstm_layers)

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

    def forward_once(self, batch, lengths):
        # embedded input of shape = (batch_size, sequence_len,  embedding_size)
        embeddings = self.word_embeddings(batch)

        # permute embedded input to shape = (sequence_len, batch_size, embedding_size)
        embeddings = embeddings.permute(1, 0, 2)

        # perform forward pass of LSTM
        output, (final_hidden_state, final_cell_state) = self.lstm(
            embeddings, self.hidden
        )

        return final_hidden_state[-1]

    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths):
        """
        Performs the forward pass for each batch
        """
        ## init context and hidden weights for lstm cell
        self.hidden = self.init_hidden(sent1_batch.size(0))

        self.sent1_out = self.forward_once(sent1_batch, sent1_lengths)
        self.sent2_out = self.forward_once(sent2_batch, sent2_lengths)
        similarity = similarity_score(self.sent1_out, self.sent2_out)
        return similarity
