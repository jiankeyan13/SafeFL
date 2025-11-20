from datasets import cifar10
from dataset_store import build_dataset

if __name__ == '__main__':
    dataset = build_dataset('cifar10', 'data', True)
    print(dataset.name)
    print(dataset.split)
    print(len(dataset))
    data = dataset[0]
    print(data[0],data[1])