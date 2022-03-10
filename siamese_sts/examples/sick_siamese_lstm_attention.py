from torch.functional import norm
from siamese_sts.data_loader import STSData
from siamese_sts.siamese_net import SiameseBiLSTMAttention
from siamese_sts.trainer.train import train_model
import torch
from torch import nn


def main():
    ## define configurations and hyperparameters
    columns_mapping = {
        "sent1": "sentence_A",
        "sent2": "sentence_B",
        "label": "relatedness_score",
    }
    dataset_name = "sick"
    sick_data = STSData(dataset_name=dataset_name, columns_mapping=columns_mapping)
    sick_dataloaders = sick_data.get_data_loader()
    batch_size = 64
    output_size = 1
    hidden_size = 128
    vocab_size = len(sick_data.vocab)
    embedding_size = 300
    embedding_weights = sick_data.vocab.vectors
    lstm_layers = 4
    learning_rate = 1e-1
    fc_hidden_size = 64
    max_epochs = 20
    bidirectional = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## self attention config
    self_attention_config = {
        "hidden_size": 150,  ## refers to variable 'da' in the ICLR paper
        "output_size": 20,  ## refers to variable 'r' in the ICLR paper
        "penalty": 0.6,  ## refers to penalty coefficient term in the ICLR paper
    }

    ## init siamese lstm
    siamese_lstm_attention = SiameseBiLSTMAttention(
        batch_size=batch_size,
        output_size=output_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        embedding_weights=embedding_weights,
        lstm_layers=lstm_layers,
        self_attention_config=self_attention_config,
        fc_hidden_size=fc_hidden_size,
        device=device,
        bidirectional=bidirectional,
    )

    ## define optimizer and loss function
    optimizer = torch.optim.Adam(params=siamese_lstm_attention.parameters())

    train_model(
        model=siamese_lstm_attention,
        optimizer=optimizer,
        dataloader=sick_dataloaders,
        data=sick_data,
        max_epochs=max_epochs,
        config_dict={
            "device": device,
            "model_name": "siamese_lstm_attention",
            "self_attention_config": self_attention_config,
        },
    )


if __name__ == "__main__":
    main()
