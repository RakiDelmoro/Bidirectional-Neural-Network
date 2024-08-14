import torch
import math
from torch.nn import Parameter
from layer import weight_and_bias_initialization

def network(feature_sizes: list, device: str):

    def nodes(weight: torch.Tensor, bias: torch.Tensor):
        # weight initialization
        w, b = weight_and_bias_initialization(weight, bias)

        def node_computation(input_feature: torch.Tensor):
            return (torch.matmul(input_feature, w)) + b

        return node_computation

    forward_layers = []
    backward_layers = []
    model_parameters = []

    for each in range(len(feature_sizes)):
        # Weights for each node
        weights = Parameter(torch.empty((feature_sizes[each], 1), device=device))
        # Bias for each node 
        bias = Parameter(torch.empty(1, device=device))
        layer_parameters = []
        forward_layer_nodes = []
        backward_layer_nodes = []

        node = nodes(weights, bias)
        for _ in range(feature_sizes[each]):
            forward_layer_nodes.append(node)
            backward_layer_nodes.append(node)

        forward_layers.append(forward_layer_nodes)
        backward_layers.insert(0, backward_layer_nodes)
        model_parameters.append(layer_parameters)

    def forward_in_bidirectional(image_data, label_data):
        forward_layers = forward_layers
        backward_layers = forward_layers[::-1]
        forward_pass_outputs = []
        backward_pass_outputs = []

        input_for_forward_pass = image_data
        input_for_backward_pass = label_data
        forward_pass_outputs.append(input_for_forward_pass)
        backward_pass_outputs.insert(0, input_for_backward_pass)

        for each_layer in range(len(forward_layers)):
            # Forward pass
            forward_output_nodes = []
            for forward_pass in forward_layers[each_layer]:
                forward_output_node = forward_pass(input_for_forward_pass)
                forward_output_nodes.append(forward_output_node)
            input_for_forward_pass = torch.stack(forward_output_nodes, dim=0)
            forward_pass_outputs.append(input_for_forward_pass)

            # Backward pass
            backward_output_nodes = []
            for backward_pass in forward_layers[each_layer]:
                backward_output_node = backward_pass(input_for_backward_pass)
                backward_output_nodes.append(backward_output_node)
            input_for_backward_pass = torch.stack(backward_output_nodes, dim=0)
            backward_pass_outputs.insert(0, input_for_backward_pass)

        return forward_pass_outputs, backward_pass_outputs

    return forward_in_bidirectional

def model_runner(model, number_of_epochs, model_parameters, lr):

    def nodes_discriminator(left_to_right_outputs, right_to_left_outputs):
        feature_nodes_loss = []
        for each_feature_nodes in range(len(left_to_right_outputs)):
            minimum_activation_per_nodes = torch.min(left_to_right_outputs[each_feature_nodes], right_to_left_outputs[each_feature_nodes])
            maximum_activation_per_nodes = torch.max(left_to_right_outputs[each_feature_nodes], right_to_left_outputs[each_feature_nodes])
            difference_per_nodes = minimum_activation_per_nodes / maximum_activation_per_nodes
            feature_nodes_loss.append(difference_per_nodes)

        return feature_nodes_loss

    def backpropagate_per_node(loss_per_nodes, model_parameters, learning_rate):
        for feature_layer_index, each_feature_nodes in enumerate(loss_per_nodes):
            weights_to_update_per_node = [[Parameter(model_parameters[feature_layer_index][0]), Parameter(model_parameters[feature_layer_index][1])] for _ in range(each_feature_nodes.shape[0])]
            for node_loss_index, each_node_loss in enumerate(each_feature_nodes):
                layer_optimizer = torch.optim.AdamW(weights_to_update_per_node[node_loss_index], learning_rate)
                layer_optimizer.zero_grad()
                each_node_loss.backward() 
                layer_optimizer.step()

        return weights_to_update_per_node

    def training(image_data, label_data):
        for epoch in range(number_of_epochs):
            forward_output, backward_output = model(image_data, label_data)
            feature_loss_per_node = nodes_discriminator(forward_output, backward_output)
            loss_per_nodes = backpropagate_per_node(feature_loss_per_node, model_parameters, lr)
            # TODO: Define the training objective of our model
            # TODO: Write a function about the training objective
            # TODO: Update the weights of the model based on training objective

    return training

x = torch.randn(10, device="cuda")
y = torch.randn(5, device="cuda")
model = network(feature_sizes=[10, 20, 20, 5], device="cuda")
print(model(x, y))
