import torch.utils.data.dataset as Dataset


class libsvm_dataset(Dataset.Dataset):
    """
    The class for loading libsvm format data
    """
    def __init__(self, Data, label):
        """
        :param Data: feature data
        :param label: label data
        """
        super(libsvm_dataset, self).__init__()
        # initialize
        self.Data = Data
        self.label = label

    # return the size of dataset
    def __len__(self):
        return len(self.Data)

    # get data and label from dataset
    def __getitem__(self, item):
        data = self.Data[item]
        label = self.label[item]

        return data, label
