import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torchvision.models as models

from lora import linear_layer_parameterization


def initialize_resnet18(num_classes=10, pretrained=True, with_lora=False, lora_rank=1, lora_alpha=1):
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
    return resnet18


def train_model(model, trainloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


def test_model(model, testloader):
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
