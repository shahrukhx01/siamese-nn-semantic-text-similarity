import pandas as pd
import numpy as np
from preprocess import Preprocess
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from hasoc_dataset import HASOCDataset

logging.basicConfig(level=logging.INFO)

"""
For loading HASOC data loading and preprocessing
"""


class HASOCData:
    def __init__(self, file_path):
        """
        Loads data into memory and create vocabulary from text field.
        """
        self.load_data(file_path)  ## load data file into memory
        self.create_vocab()  ## create vocabulary over entire dataset before train/test split

    def load_data(self, file_paths):
        """
        Reads data set file from disk to memory using pandas
        """
        logging.info("loading and preprocessing data...")
        self.data = pd.read_csv(file_paths["data_file"], sep="\t")  ## reading data file
        self.data = Preprocess(file_paths["stpwds_file"]).perform_preprocessing(
            self.data
        )  ## performing text preprocessing
        logging.info("reading and preprocessing data completed...")

    def create_vocab(self):
        """
        Creates vocabulary over entire text data field.
        """
        logging.info("creating vocabulary...")
        self.vocab = sorted(
            list(
                self.data.clean_text.str.split(expand=True)
                .stack()
                .value_counts()
                .keys()
            )
        )
        self.word2index = {
            word: index for index, word in enumerate(self.vocab)
        }  ## map each word to index
        self.index2word = {
            index: word for index, word in enumerate(self.vocab)
        }  ## map each index to word
        logging.info("creating vocabulary completed...")

    def transform_labels(self):
        """
        Maps categorical string labels to {0, 1}
        """
        self.data["labels"] = self.data.task_1.map({"HOF": 1, "NOT": 0})

    def data2tensors(self):
        """
        Converts raw data sequences into vectorized sequences as tensors
        """
        self.transform_labels()
        vectorized_sequences, sequence_lengths, targets = [], [], []
        raw_data = list(self.data.clean_text.values)

        ## get the text sequence from dataframe
        for index, sentence in enumerate(raw_data):
            ## convert sentence into vectorized form replacing words with vocab indices
            vectorized_sequence = self.vectorize_sequence(sentence)
            sequence_length = len(
                vectorized_sequence
            )  ## computing sequence lengths for padding
            if sequence_length <= 0:
                continue

            vectorized_sequences.append(
                vectorized_sequence
            )  ## adding sequence vectors to train matrix
            sequence_lengths.append(sequence_length)
            targets.append(
                self.data.labels.values[index]
            )  ## fetching label for this example

        ## padding zeros at the end of tensor till max length tensor
        padded_sequence_tensor = self.pad_sequences(
            vectorized_sequences, torch.LongTensor(sequence_lengths)
        )
        length_tensor = torch.LongTensor(sequence_lengths)  ## casting to long
        target_tensor = torch.LongTensor(targets)  ## casting to long

        return (padded_sequence_tensor, target_tensor, length_tensor, raw_data)

    def get_data_loader(self, batch_size=8):
        (
            padded_sequence_tensor,
            target_tensor,
            length_tensor,
            raw_data,
        ) = self.data2tensors()
        self.hasoc_dataset = HASOCDataset(
            padded_sequence_tensor, target_tensor, length_tensor, raw_data
        )

        train_sampler, dev_sampler, test_sampler = self.train_dev_test_split()
        ## prepare dictionary for train and test PyTorch based dataloaders
        hasoc_dataloader = {
            "train_loader": torch.utils.data.DataLoader(
                self.hasoc_dataset, batch_size=batch_size, sampler=train_sampler
            ),
            "test_loader": torch.utils.data.DataLoader(
                self.hasoc_dataset, batch_size=batch_size, sampler=test_sampler
            ),
            "dev_loader": torch.utils.data.DataLoader(
                self.hasoc_dataset, batch_size=batch_size, sampler=dev_sampler
            ),
        }
        return hasoc_dataloader

    def train_dev_test_split(
        self, dev_size=0.2, test_size=0.2, shuffle_dataset=True, random_seed=42
    ):
        """
        Splits the data into train and test set using the SubsetSampler provided by PyTorch.
        """
        ## creating data indices for training and validation splits:
        dataset_size = len(self.hasoc_dataset)
        indices = list(range(dataset_size))
        train_test_split = int(np.floor(test_size * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)  ## shuffle row indices before split
        train_indices, test_indices = (
            indices[train_test_split:],
            indices[:train_test_split],
        )

        train_dev_split = int(
            np.floor(dev_size * len(train_indices))
        )  ## splitting train data further into train and dev
        train_indices, dev_indices = (
            train_indices[train_dev_split:],
            train_indices[:train_dev_split],
        )

        ## creating pytorch data samplers
        return (
            SubsetRandomSampler(train_indices),
            SubsetRandomSampler(dev_indices),
            SubsetRandomSampler(test_indices),
        )

    def sort_batch(self, batch, targets, lengths):
        """
        Sorts the data, lengths and target tensors based on the lengths
        of the sequences from longest to shortest in batch
        """
        sequence_lengths, perm_idx = lengths.sort(0, descending=True)
        sequence_tensor = batch[perm_idx]
        target_tensor = targets[perm_idx]
        return sequence_tensor.transpose(0, 1), target_tensor, sequence_lengths

    def vectorize_sequence(self, sentence):
        """
        Replaces tokens with their indices in vocabulary
        """
        return [self.word2index[token] for token in sentence.split()]

    def pad_sequences(self, vectorized_sequences, sequence_lengths):
        """
        Pads zeros at the end of each sequence in data tensor till max
        length of sequence in that batch
        """
        padded_sequence_tensor = torch.zeros(
            (len(vectorized_sequences), sequence_lengths.max())
        ).long()  ## init zeros tensor
        for idx, (seq, seqlen) in enumerate(
            zip(vectorized_sequences, sequence_lengths)
        ):  ## iterate over each sequence
            padded_sequence_tensor[idx, :seqlen] = torch.LongTensor(
                seq
            )  ## each sequence get padded by zeros until max length in that batch
        return padded_sequence_tensor  ## returns padded tensor
