from torch.functional import norm
from siamese_sts.data_loader import STSData
from siamese_sts.siamese_net.siamese_transformer import SiameseTransformer
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
    max_sequence_len = 128
    dataset_name = "sick"
    sick_data = STSData(
        dataset_name=dataset_name,
        columns_mapping=columns_mapping,
        model_name="transformer",
        max_sequence_len=max_sequence_len,
    )
    sick_dataloaders = sick_data.get_data_loader()
    batch_size = 64
    output_size = 1
    hidden_size = 128
    attention_heads = 4
    vocab_size = len(sick_data.vocab)
    embedding_size = 300
    embedding_weights = sick_data.vocab.vectors
    transformer_layers = 4
    lstm_layers = 4
    learning_rate = 1e-1

    dropout = 0.5
    max_epochs = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## init siamese lstm
    siamese_transformer = SiameseTransformer(
        batch_size=batch_size,
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        nhead=attention_heads,
        hidden_size=hidden_size,
        transformer_layers=transformer_layers,
        embedding_weights=embedding_weights,
        device=device,
        dropout=dropout,
        max_sequence_len=max_sequence_len,
        lstm_layers=lstm_layers,
    )

    ## define optimizer and loss function
    optimizer = torch.optim.Adam(params=siamese_transformer.parameters())

    train_model(
        model=siamese_transformer,
        optimizer=optimizer,
        dataloader=sick_dataloaders,
        data=sick_data,
        max_epochs=max_epochs,
        config_dict={"device": device, "model_name": "siamese_transformer"},
    )


if __name__ == "__main__":
    main()
