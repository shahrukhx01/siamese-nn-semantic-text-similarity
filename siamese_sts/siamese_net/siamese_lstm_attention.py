import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from siamese_sts.utils.utils import similarity_score


"""
Wrapper class using Pytorch nn.Module to create the architecture for our model
Architecture is based on the paper: 
A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
https://arxiv.org/pdf/1703.03130.pdf
"""


class SiameseBiLSTMAttention(nn.Module):
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
        bidirectional,
        self_attention_config,
        fc_hidden_size,
    ):
        super(SiameseBiLSTMAttention, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lstm_layers = lstm_layers
        self.device = device
        self.bidirectional = bidirectional
        self.fc_hidden_size = fc_hidden_size
        self.lstm_directions = (
            2 if self.bidirectional else 1
        )  ## decide directions based on input flag

        ## model layers
        # initializing the look-up table.
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)

        # assigning the look-up table to the pre-trained fasttext word embeddings.
        self.word_embeddings.weight = nn.Parameter(
            embedding_weights.to(self.device), requires_grad=True
        )

        ## initializng lstm layer
        self.bilstm = nn.LSTM(
            self.embedding_size,
            self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            bidirectional=self.bidirectional,
            dropout=0.5,
        )

        ## initializing self attention layers
        self.self_attention = None
        self.fc_layer = None

        ## incase we are using bi-directional lstm we'd have to take care of bi-directional outputs in
        ## subsequent layers

        self.self_attention = SelfAttention(
            self.lstm_hidden_size * self.lstm_directions,
            self_attention_config["hidden_size"],
            self_attention_config["output_size"],
        )
        ## this layer comes right after self attention computation
        self.fc_layer = nn.Linear(
            self.lstm_directions
            * self.lstm_hidden_size
            * self_attention_config["output_size"],
            self.fc_hidden_size,
        )

    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """
        layer_size = self.lstm_layers
        if self.bidirectional:
            layer_size *= 2  # since we have two layers instantiated for each lstm layer of bi-lstm
        return (
            Variable(
                torch.zeros(layer_size, batch_size, self.lstm_hidden_size).to(
                    self.device
                )
            ),
            Variable(torch.zeros(layer_size, batch_size, self.lstm_hidden_size)).to(
                self.device
            ),
        )

    def forward_once(self, batch, lengths):
        """
        Performs the forward pass for each batch
        """

        ## batch shape: (num_sequences, batch_size)
        ## embeddings shape: (seq_len, batch_size, embedding_size)

        embeddings = self.word_embeddings(batch)

        # permute embedded input to shape = (sequence_len, batch_size, embedding_size)
        embeddings = embeddings.permute(1, 0, 2)

        """
		here padded output refers to variable 'H' from the ICLR paper
		as LSTM's output contains all the hidden states for a given a sequence
		hence we use this as variable 'H'
		output shape : (seq_len, batch_size, num_lstm_layers * num_directions)
		"""
        output, (final_hidden_state, final_cell_state) = self.bilstm(
            embeddings, self.hidden
        )

        # output post permute shape: (batch_size , seq_len, num_lstm_layers * num_directions)
        output = output.permute(1, 0, 2)

        ## refers to annotation matrix 'A' in ICLR paper
        annotation_weight_matrix = self.self_attention(output)

        """
		in the final step we compute output matrix 'M = AH' which has sentnece embeddings
		here the bmm (batch matrix mul) inputs have following shapes
		annotation_weight_matrix : (batch_size, attention_out, seq_len)
		output: (batch_size , seq_len, lstm_hidden_size * num_directions)
		sentence_embedding shape: (batch_size , attention_out, lstm_hidden_size * num_directions)
		"""
        sentence_embedding = torch.bmm(annotation_weight_matrix, output)

        """
		transforming the two lstm directions with attention output in sentence embedding matrix 
		for fully connected layer
		sentence_embedding shape: (batch_size, lstm_directions*lstm_hidden_size*self_attention_output_size)
		"""
        sentence_embedding = sentence_embedding.view(
            -1, sentence_embedding.size()[1] * sentence_embedding.size()[2]
        )

        ## feeding sentence_embedding result to fully connected
        fc_out = self.fc_layer(sentence_embedding)

        return fc_out, annotation_weight_matrix

    def forward(self, sent1_batch, sent2_batch, sent1_lengths, sent2_lengths):
        """
        Performs the forward pass for each batch
        """
        ## init context and hidden weights for lstm cell
        self.hidden = self.init_hidden(sent1_batch.size(0))

        self.sent1_out, sent1_annotation_weight_matrix = self.forward_once(
            sent1_batch, sent1_lengths
        )
        self.sent2_out, sent2_annotation_weight_matrix = self.forward_once(
            sent2_batch, sent2_lengths
        )
        similarity = similarity_score(self.sent1_out, self.sent2_out)
        return (
            similarity,
            sent1_annotation_weight_matrix,
            sent2_annotation_weight_matrix,
        )


class SelfAttention(nn.Module):
    """
    Implementation of the attention block
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        ## corresponds to variable Ws1 in ICLR paper, we don't use the bias term as suggested in paper
        self.layer1 = nn.Linear(input_size, hidden_size, bias=False)
        ## corresponds to variable Ws2 in ICLR paper, we don't use the bias term as suggested in paper
        self.layer2 = nn.Linear(hidden_size, output_size, bias=False)

    ## the forward function would receive lstm's all hidden states as input
    def forward(self, attention_input):
        ## expected input shape: (batch_size , seq_len, num_lstm_layers * num_directions)
        out = self.layer1(attention_input)
        # out shape: (batch_size, seq_len, attention_hidden_size)
        out = torch.tanh(out)
        # out shape: (batch_size, seq_len, attention_out)
        out = self.layer2(out)
        ## out shape post permute: (batch_size, attention_out, seq_len)
        out = out.permute(0, 2, 1)
        out = F.softmax(out, dim=2)  ## softmax dimenion as per the paper

        return out  ## out shape: (batch_size, attention_out, seq_len)
