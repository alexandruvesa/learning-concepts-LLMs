import multiprocessing as mp

import torch
from torch import nn

from dataset import load_cifar10_dataset
from lora import freeze_non_lora_layers
from network import initialize_resnet18, train_model, test_model

mp.set_start_method('fork')
"""
If you are using macOS or any Unix-based OS, the default start method for multiprocessing is 'fork'. 
Python 3.8+ changed the default method to 'spawn' for macOS. Using 'fork' as the start method might solve the issue. 
"""


def train_with_lora():
    trainloader, testloader, classes = load_cifar10_dataset()

    resnet18 = initialize_resnet18(with_lora=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

    train_model(resnet18, trainloader, criterion, optimizer, epochs=1)
    test_model(resnet18, testloader)


def fine_tune_with_lora():
    trainloader, testloader, classes = load_cifar10_dataset(selected_class=3)
    resnet18 = initialize_resnet18(with_lora=True)
    freeze_non_lora_layers(resnet18)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)

    train_model(resnet18, trainloader, criterion, optimizer, epochs=1)
    test_model(resnet18, testloader)


train_with_lora()
fine_tune_with_lora()
