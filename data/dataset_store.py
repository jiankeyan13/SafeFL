from torch.utils.data import Dataset
from registry import dataset_builders
import numpy as np

class DatasetStore(Dataset):
    def __init__(self, name, split, dataset):
        self.dataset = dataset
        self.name = name
        self.split = split
    
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]
    
    def get_label(self):
        """
        统一获取所有样本标签的方法。
        返回: numpy array of shape (N,)
        """
        if hasattr(self.dataset, 'targets'):
            return np.array(self.dataset.targets)
        if hasattr(self.dataset, 'labels'):
            return np.array(self.dataset.labels)
        print(f"{self.name} has no labels/targets, Iterating to load labels (slow)...")
        labels = []
        for i in range(len(self.dataset)):
            labels.append(self.dataset[i][1])
        return np.array(labels)


def build_dataset(name: str, root:str, is_train: bool):
    return dataset_builders[name](root, is_train)

if __name__ == '__main__':
    train_dataset = build_dataset('cifar10', 'data', True)
    print(train_dataset.split)
    print(len(train_dataset))
    data = train_dataset[0]
    # print(data[0],data[1])
    test_dataset = build_dataset('cifar10', 'data', False)
    print(len(test_dataset))