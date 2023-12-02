import torchvision


def get_network_weights(network: torchvision.models) -> dict:
    """
    The get_network_weights function takes a network as input and returns a dictionary of the original weights.
    The function is used to compare the original weights with those after training.

    :param network: torchvision.models: Specify the network to be used
    :return: A dictionary with the original weights of the network
    :doc-author: Alexandru Vesa
    """
    original_weights = {}
    for name, param in network.named_parameters():
        original_weights[name] = param.clone().detach()
    return original_weights


def count_network_parameters(net):
    total_parameters_lora = 0
    total_parameters_non_lora = 0

    # Count LoRA parameters in the final fully connected layer
    if hasattr(net.fc, 'parametrizations'):
        lora_layer = net.fc.parametrizations["weight"][0]
        total_parameters_lora += lora_layer.lora_A.nelement() + lora_layer.lora_B.nelement()
        print(
            f'LoRA Layer: W: {net.fc.weight.shape} + Lora_A: {lora_layer.lora_A.shape} + Lora_B: {lora_layer.lora_B.shape}')

    # Count non-LoRA parameters for the entire network
    for param in net.parameters():
        total_parameters_non_lora += param.nelement()

    # Print the parameter counts
    print(f'Total number of parameters (original): {total_parameters_non_lora:,}')
    print(f'Total number of parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}')
    print(f'Parameters introduced by LoRA: {total_parameters_lora:,}')

    # Calculate and print the parameter increment
    parameters_increment = (total_parameters_lora / total_parameters_non_lora) * 100
    print(f'Parameters increment: {parameters_increment:.3f}%')
