import logging
import torch
import nltk
import numpy as np
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from tqdm import tqdm as tqdm1
import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import set_seed
from transformers.file_utils import is_offline_mode
from torch.functional import norm
from siamese_sts.data_loader import STSData
from siamese_sts.siamese_net.siamese_bert import BertForSequenceClassification
from siamese_sts.trainer.train import train_model
import torch
from torch import nn
from pathlib import Path


logger = logging.getLogger(__name__)


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


def main():
    ## define configurations and hyperparameters
    model_name = "prajjwal1/bert-mini"
    num_epochs = 1
    batch_size = 64
    columns_mapping = {
        "sent1": "sentence_A",
        "sent2": "sentence_B",
        "label": "relatedness_score",
    }
    ## load dataset
    dataset_name = "sick"
    sick_data = STSData(
        dataset_name=dataset_name,
        columns_mapping=columns_mapping,
        normalize_labels=True,
        normalization_const=5.0,
    )
    sick_dataloader = sick_data.get_dataset_bert()

    ## init model
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


if __name__ == "__main__":
    main()
