from typing import Dict

import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torchvision
import torchvision.models as models

from lora import linear_layer_parameterization


def initialize_resnet18(num_classes: int = 10, pretrained: bool = True, with_lora: bool = False, lora_rank: int = 1,
                        lora_alpha: int = 1, device: str = 'mps') -> torchvision.models:
    """
     Initialize a ResNet18 model and apply LoRA to the final fully connected layer.

     Parameters:
     - num_classes (int): Number of classes for the output layer.
     - pretrained (bool): If True, use a pre-trained ResNet18; otherwise, initialize from scratch.
     - device (str): The device to use, e.g., 'cpu' or 'cuda'.
     - lora_rank (int): The rank for the LoRA adaptation. Determines the rank of the low-rank matrices
                        (A and B). Controls the model's capacity to adapt with a lower value implying
                        fewer parameters and computational overhead, and a higher value implying more
                        capacity to adapt at the cost of more computational resources.
     - lora_alpha (float): The scaling factor for the LoRA adaptation. Controls the magnitude of
                           changes introduced by LoRA. Helps in maintaining the balance between
                           pre-trained weights and updates introduced by LoRA.

     Returns:
     - A ResNet18 model with LoRA applied to the final fully connected layer.
     """
    resnet18 = models.resnet18(pretrained=pretrained)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, num_classes)

    if with_lora:
        parametrize.register_parametrization(
            resnet18.fc, "weight",
            linear_layer_parameterization(resnet18.fc, 'cpu', rank=lora_rank, lora_alpha=lora_alpha)
        )
    return resnet18.to(device=device)


import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import optim


def train_model(model: torchvision.models, trainloader: DataLoader, criterion: CrossEntropyLoss, optimizer: optim,
                epochs=10, device='mps') -> None:
    # Move model to the specified device
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Move data to the specified device
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')


def test_model(model: torchvision.models, testloader: DataLoader) -> None:
    """
    The test_model function takes a model and testloader as input.
    It then iterates through the testloader, feeding each batch of images to the model.
    The output is compared with the labels, and if they match, it increments class_correct for that class by 1.
    Finally it prints out an accuracy score for each class.

    :param model: Pass the model to be tested
    :param testloader: Pass the test dataset to the function
    :return: The accuracy of each class in the test set
    :doc-author: Alex Vesa
    """
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = [0 for _ in range(len(classes))]
    class_total = [0 for _ in range(len(classes))]
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                if label < len(classes):  # Check if the label is within the range of defined classes
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    for i in range(len(classes)):
        if class_total[i] > 0:  # Only print accuracy for classes that are present in the test set
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


def store_original_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Stores the original weights of a PyTorch model before fine-tuning.

    Parameters:
    - model (nn.Module): The PyTorch model from which to store weights.

    Returns:
    - Dict[str, torch.Tensor]: A dictionary where keys are layer names and values are the corresponding original weight tensors.
    """
    original_weights = {}
    for name, param in model.named_parameters():
        if 'parametrizations' not in name:  # Exclude parametrized layers
            original_weights[name] = param.detach().clone()
    return original_weights


def verify_weights(model: nn.Module, original_weights: Dict[str, torch.Tensor], enable_disable_lora: callable) -> None:
    """
    Verifies that the original weights of non-LoRA layers are unchanged and that the LoRA layers are updated correctly.

    Parameters:
    - model (nn.Module): The PyTorch model to verify.
    - original_weights (Dict[str, torch.Tensor]): The dictionary of original weights, as returned by `store_original_weights`.
    - enable_disable_lora (callable): A function to enable or disable LoRA in the model.

    Returns:
    - None: The function asserts the validity of the model's weights and does not return anything.
    """
    # Check non-LoRA weights
    for name, param in model.named_parameters():
        if 'parametrizations' not in name:
            assert torch.all(param == original_weights[name]), f"Weights changed for {name}"

    # Check LoRA weights
    enable_disable_lora(model, enabled=True)
    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations'):
            lora_param = module.parametrizations.weight[0]
            updated_weight = lora_param.original + (lora_param.lora_B @ lora_param.lora_A) * lora_param.scale
            assert torch.equal(module.weight, updated_weight), f"LoRA updated weights do not match for {name}"

    enable_disable_lora(model, enabled=False)
    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations'):
            assert torch.equal(module.weight, original_weights[
                f'{name}.weight']), f"Original weights do not match when LoRA is disabled for {name}"


def enable_disable_lora(model: nn.Module, enabled: bool) -> None:
    """
    Enables or disables LoRA in a PyTorch model.

    Parameters:
    - model (nn.Module): The PyTorch model in which LoRA will be enabled or disabled.
    - enabled (bool): If True, LoRA is enabled. If False, LoRA is disabled.

    Returns:
    - None: The function directly modifies the model and does not return anything.
    """
    for module in model.modules():
        if hasattr(module, 'parametrizations'):
            lora_param = module.parametrizations.weight[0]
            lora_param.enabled = enabled
