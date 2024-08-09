import torch
from layer import nodes
from torch.nn import CrossEntropyLoss, MSELoss

image_classifier_loss_function = CrossEntropyLoss()
generative_loss_function = MSELoss()

def bidirectional_network(feature_sizes: list, input_feature_size: int, output_feature: int, device: str):
    layers = []
    parameters = []

    first_layer_of_nodes, first_layer_weights, first_layer_bias = nodes(input_feature_size, feature_sizes[0], device)
    layers.append(first_layer_of_nodes)
    parameters.extend([first_layer_weights, first_layer_bias])

    number_of_layers = len(feature_sizes) - 1
    for each in range(number_of_layers):
        in_feature = feature_sizes[each]
        out_feature = feature_sizes[each+1]
        inner_layer_of_nodes, inner_weight, inner_bias = nodes(in_feature, out_feature, device)
        layers.append(inner_layer_of_nodes)
        parameters.extend([inner_weight, inner_bias])

    last_layer_of_nodes, last_layer_weights, last_layer_bias = nodes(feature_sizes[-1], output_feature, device)
    layers.append(last_layer_of_nodes)
    parameters.extend([last_layer_weights, last_layer_bias])

    def forward_layers_of_nodes(input_data: torch.Tensor, reverse_forward: bool):
        network_layers = layers[::-1] if reverse_forward else layers

        previous_layer_output = input_data
        neurons_activation = []
        for layer in network_layers:
            previous_layer_output = layer(previous_layer_output, reverse_forward)
            neurons_activation.append(previous_layer_output)

        return previous_layer_output, neurons_activation
    
    def bidirectional_forward(batched_image_data, batched_label_data):
        first_forward_pass_output, first_forward_pass_activations = forward_layers_of_nodes(input_data=batched_image_data, reverse_forward=False)
        second_forward_pass_output, second_forward_pass_activations = forward_layers_of_nodes(input_data=batched_label_data, reverse_forward=True)

        return first_forward_pass_output, first_forward_pass_activations, second_forward_pass_output, second_forward_pass_activations

    # TODO: Create a function that computes the distance of the two tensor. Return same shape with a value of the distance of two tensor correspond to it's position
    # TODO: Update the weights on how based on the distance of the tensor.

    return bidirectional_forward

x = torch.randn(1, 10, device="cuda")
y = torch.randn(1, 5, device="cuda")
network = bidirectional_network([10, 10, 10], 10, 5, "cuda")
print(network(x, y))
