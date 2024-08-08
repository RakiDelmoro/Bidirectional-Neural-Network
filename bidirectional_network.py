import torch
from layer import nodes

def bidirectional_network(feature_sizes: list, input_feature_size: int, output_feature: int, device: str):
    layers = []
    parameters = []

    first_layer_of_nodes, first_layer_weights, first_layer_bias = nodes(input_feature_size, feature_sizes[0], device)
    layers.append(first_layer_of_nodes)
    parameters.extend([first_layer_weights, first_layer_bias])

    number_of_layers = len(feature_sizes) - 1
    for each in range(number_of_layers):
        input_feature = feature_sizes[each]
        output_feature = feature_sizes[each+1]
        inner_layer_of_nodes, inner_weight, inner_bias = nodes(input_feature, output_feature, device)
        layers.append(inner_layer_of_nodes)
        parameters.extend([inner_weight, inner_bias])

    last_layer_of_nodes, last_layer_weights, last_layer_bias = nodes(feature_sizes[-1], output_feature, device)
    layers.append(last_layer_of_nodes)
    parameters.extend([last_layer_weights, last_layer_bias])

    def forward_layers_of_nodes(input_batch: torch.Tensor):
        previous_layer_output = input_batch
        for layer in layers:
            previous_layer_output = layer(previous_layer_output)

        return last_layer_of_nodes(previous_layer_output)

    # pass both image data and corresponding label
    def bidirectional_forward(input_image_batched, input_label_batched):
        # Forward image classifier.
        image_classifier_output = forward_layers_of_nodes(input_image_batched)
        # TODO: Function to forward the digit label and get the predicted image pixel.
        # TODO: Function to get the loss from image classifier and generative image
        # TODO: Get the activation value of both bidirectional passes.

