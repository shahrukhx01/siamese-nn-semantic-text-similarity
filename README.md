# siamese-nn-semantic-text-similarity
A repository containing comprehensive Neural Networks based PyTorch implementations for the semantic text similarity task, including architectures such as Siamese-LSTM, Siamese-BiLSTM-Attention, Siamese-Transformer and Siamese-BERT.

![4-Figure1-1](https://user-images.githubusercontent.com/6007894/147794847-04eee203-c0ba-42f8-abe1-0e66757e46f2.png)

# Usage
[Siamese LSTM Example](https://github.com/shahrukhx01/siamese-nn-semantic-text-similarity/blob/main/siamese_sts/examples/sick_siamese_lstm.py)
```python
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
```
download spacy en model for tokenization
```bash
python -m spacy download en
```
