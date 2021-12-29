from torch.utils.data import Dataset

"""
Standard Pytorch Dataset class for loading datasets.
"""


class HASOCDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, length_tensor, raw_data):
        """
        initializes  and populates the the length, data and target tensors, and raw texts list
        """
        assert data_tensor.size(0) == target_tensor.size(0) == length_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.length_tensor = length_tensor
        self.raw_data = raw_data

    def __getitem__(self, index):
        """
        returns the tuple of data tensor, targets, lengths of sequences tensor and raw texts list
        """
        return (
            self.data_tensor[index],
            self.target_tensor[index],
            self.length_tensor[index],
            self.raw_data[index],
        )

    def __len__(self):
        """
        returns the length of the data tensor.
        """
        return self.data_tensor.size(0)
