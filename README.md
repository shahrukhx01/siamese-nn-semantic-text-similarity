# Siamese Deep Neural Networks for Semantic Text Similarity PyTorch
A repository containing comprehensive Neural Networks based PyTorch implementations for the semantic text similarity task, including architectures such as Siamese-LSTM, Siamese-BiLSTM-Attention, Siamese-Transformer and Siamese-BERT.

![4-Figure1-1](https://user-images.githubusercontent.com/6007894/147794847-04eee203-c0ba-42f8-abe1-0e66757e46f2.png)

# Usage
* install dependencies
```bash
pip install -r requirements.txt
```
* download spacy en model for tokenization
```bash
python -m spacy download en
```

## Siamese LSTM
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

    ## define optimizer
    optimizer = torch.optim.Adam(params=siamese_lstm.parameters())
   
   ## train model
    train_model(
        model=siamese_lstm,
        optimizer=optimizer,
        dataloader=sick_dataloaders,
        data=sick_data,
        max_epochs=max_epochs,
        config_dict={"device": device, "model_name": "siamese_lstm"},
    )
```

## Siamese BiLSTM with Attention
[Siamese BiLSTM with Attention Example](https://github.com/shahrukhx01/siamese-nn-semantic-text-similarity/blob/main/siamese_sts/examples/sick_siamese_lstm_attention.py)
```python
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
    
    ## define optimizer
    optimizer = torch.optim.Adam(params=siamese_lstm_attention.parameters())
   
   ## train model
    train_model(
        model=siamese_lstm_attention,
        optimizer=optimizer,
        dataloader=sick_dataloaders,
        data=sick_data,
        max_epochs=max_epochs,
        config_dict={"device": device, "model_name": "siamese_lstm_attention"},
    )
```

## Siamese Transformer
[Siamese Transformer Example](https://github.com/shahrukhx01/siamese-nn-semantic-text-similarity/blob/main/siamese_sts/examples/sick_siamese_transformer.py)
```python
    ## init siamese bilstm with attention
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

    ## define optimizer
    optimizer = torch.optim.Adam(params=siamese_lstm.parameters())
   
   ## train model
    train_model(
        model=siamese_lstm,
        optimizer=optimizer,
        dataloader=sick_dataloaders,
        data=sick_data,
        max_epochs=max_epochs,
        config_dict={"device": device, "model_name": "siamese_lstm"},
    )
```

## Siamese BERT
[Siamese BERT Example](https://github.com/shahrukhx01/siamese-nn-semantic-text-similarity/blob/main/siamese_sts/examples/sick_siamese_bert.py)
```python
    from siamese_sts.siamese_net.siamese_bert import BertForSequenceClassification
    ## init siamese bert
    siamese_bert = BertForSequenceClassification.from_pretrained(model_name)

    ## train model
    trainer = transformers.Trainer(
        model=siamese_bert,
        args=transformers.TrainingArguments(
            output_dir="./output",
            overwrite_output_dir=True,
            learning_rate=1e-5,
            do_train=True,
            num_train_epochs=num_epochs,
            # Adjust batch size if this doesn't fit on the Colab GPU
            per_device_train_batch_size=batch_size,
            save_steps=3000,
        ),
        train_dataset=sick_dataloader,
    )
    trainer.train()

```

