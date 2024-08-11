import torch
import math
from layer import weight_and_bias_initialization

def network(feature_sizes: list, device: str):

    def nodes(weight: torch.Tensor, bias: torch.Tensor):
        # weight initialization
        w, b = weight_and_bias_initialization(weight, bias)

        def node_computation(input_feature: torch.Tensor):
            return (torch.matmul(input_feature, w)) + b

        return node_computation

    # Intialize network parameters
    weights = [torch.empty((feature_sizes[size], feature_sizes[size+1]), device=device) for size in range(len(feature_sizes)-1)]
    biases = [torch.empty(size, device=device) for size in feature_sizes]
    network_parameters = [[weights[each], biases[each+1]] for each in range(len(feature_sizes)-1)]

    forward_pass_layers = []
    backward_pass_layers = []

    for each in range(len(feature_sizes)-1):
        forward_layer = nodes(weights[each], biases[each+1])
        backward_layer = nodes(weights[each].t(), biases[each]) # Flip the weights

        forward_pass_layers.append(forward_layer)
        backward_pass_layers.insert(0, backward_layer)

    def forward_in_bidirectional(image_data, label_data):
        forward_pass_outputs = []
        backward_pass_outputs = []

        input_for_forward_pass = image_data
        input_for_backward_pass = label_data
        forward_pass_outputs.append(input_for_forward_pass)
        backward_pass_outputs.append(input_for_backward_pass)

        for each_layer in range(len(feature_sizes)-1):
            input_for_forward_pass = forward_pass_layers[each_layer](input_for_forward_pass)
            input_for_backward_pass = backward_pass_layers[each_layer](input_for_backward_pass)

            forward_pass_outputs.append(input_for_forward_pass)
            backward_pass_outputs.insert(0, input_for_backward_pass)

        return forward_pass_outputs, backward_pass_outputs
    
    return forward_in_bidirectional, network_parameters

def model_runner(model, number_of_epochs):

    def nodes_discriminator(left_to_right_outputs, right_to_left_outputs):
        feature_nodes_loss = []
        for each_feature_nodes in range(len(left_to_right_outputs)):
            nodes_loss = left_to_right_outputs[each_feature_nodes] - right_to_left_outputs[each_feature_nodes]
            feature_nodes_loss.append(nodes_loss)

        return feature_nodes_loss

    def training(image_data, label_data):
        forward_output, backward_output = model(image_data, label_data)
        feature_loss_per_node = nodes_discriminator(forward_output, backward_output)

    return training

x = torch.randn(10, device="cuda")
y = torch.randn(5, device="cuda")
model, model_parameters = network(feature_sizes=[10, 20, 20, 5], device="cuda")
forward_outputs, backward_outputs = model(x, y)
print(forward_outputs[-1], backward_outputs[-1])
