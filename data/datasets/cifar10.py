from  registry import register_dataset
from dataset_store import DatasetStore
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

def foo(flag:bool):
    return "train" if flag else "test"
@register_dataset('cifar10')
def build_cifar10(root, is_train):    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    real_dataset = CIFAR10(root=root, train=is_train, download=True, transform = transform)
    return DatasetStore('cifar10', foo(is_train), real_dataset)