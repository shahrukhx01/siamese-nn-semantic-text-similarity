import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
Wrapper class using Pytorch nn.Module to create the architecture for our 
binary classification model
"""


class SiameseLSTM(nn.Module):
    def __init__(
        self,
        batch_size,
        output_size,
        hidden_size,
        vocab_size,
        embedding_size,
        embedding_weights,
        lstm_layers,
        device,
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

    def similarity_score(self, input1, input2):
        # Get similarity predictions:
        dif = input1.squeeze() - input2.squeeze()

        norm = torch.norm(dif, p=1, dim=dif.dim() - 1)
        y_hat = torch.exp(-norm)
        y_hat = torch.clamp(y_hat, min=1e-7, max=1.0 - 1e-7)
        return y_hat

    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths):
        """
        Performs the forward pass for each batch
        """
        ## init context and hidden weights for lstm cell
        self.hidden = self.init_hidden(sent1_batch.size(0))

        self.sent1_out = self.forward_once(sent1_batch, sent1_lengths)
        self.sent2_out = self.forward_once(sent2_batch, sent2_lengths)
        similarity = self.similarity_score(self.sent1_out, self.sent2_out)
        return similarity
