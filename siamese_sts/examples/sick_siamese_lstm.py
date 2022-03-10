from torch.functional import norm
from siamese_sts.data_loader import STSData
from siamese_sts.siamese_net.siamese_lstm import SiameseLSTM
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
    max_epochs = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## init siamese lstm
    siamese_lstm = SiameseLSTM(
        batch_size=batch_size,
        output_size=output_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        embedding_weights=embedding_weights,
        lstm_layers=lstm_layers,
        device=device,
    )

    ## define optimizer and loss function
    optimizer = torch.optim.Adam(params=siamese_lstm.parameters())

    train_model(
        model=siamese_lstm,
        optimizer=optimizer,
        dataloader=sick_dataloaders,
        data=sick_data,
        max_epochs=max_epochs,
        config_dict={"device": device, "model_name": "siamese_lstm"},
    )
    ##print(sick_dataloaders.keys())


if __name__ == "__main__":
    main()
