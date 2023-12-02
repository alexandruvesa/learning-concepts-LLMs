from typing import Tuple, Optional

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def load_cifar10_dataset(batch_size: int = 4, num_workers: int = 2, selected_class: Optional[int] = None) -> Tuple[
    DataLoader, DataLoader, Tuple[str, ...]]:
    """
    Load the CIFAR-10 dataset, optionally filtering the training set for a single class.

    Parameters:
    - batch_size: Number of samples per batch.
    - num_workers: Number of subprocesses to use for data loading.
    - selected_class: If specified, filter the training set to include only this class (0-9).

    Returns:
    - training_loader: DataLoader for the training set.
    - test_loader: DataLoader for the test set.
    - classes: List of class names.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if selected_class is not None:
        training_indices = [i for i, label in enumerate(training_set.targets) if label == selected_class]
        training_set = Subset(training_set, training_indices)

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return training_loader, test_loader, classes
