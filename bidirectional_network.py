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

    # Intialize network parameters
    weights = [Parameter(torch.empty((feature_sizes[size], feature_sizes[size+1]), device=device)) for size in range(len(feature_sizes)-1)]
    biases = [Parameter(torch.empty(size, device=device)) for size in feature_sizes]
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
        backward_pass_outputs.insert(0, input_for_backward_pass)

        for each_layer in range(len(feature_sizes)-1):
            input_for_forward_pass = forward_pass_layers[each_layer](input_for_forward_pass)
            input_for_backward_pass = backward_pass_layers[each_layer](input_for_backward_pass)

            forward_pass_outputs.append(input_for_forward_pass)
            backward_pass_outputs.insert(0, input_for_backward_pass)

        return forward_pass_outputs, backward_pass_outputs

    return forward_in_bidirectional, network_parameters

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
        each_loss_per_node = []
        for feature_layer_index, each_feature_nodes in enumerate(loss_per_nodes):
            for each_node_loss in each_feature_nodes:
                layer_parameters = torch.optim.AdamW(model_parameters[feature_layer_index], learning_rate)
                # backpropagation per node
                layer_parameters.zero_grad()
                each_node_loss.backward() # We get an error: 
                                            #Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
                # TODO: Find a way to backpropagate each node.
                layer_parameters.step()
                each_loss_per_node.append(each_node_loss)

        return each_loss_per_node

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
model, model_parameters = network(feature_sizes=[10, 20, 20, 5], device="cuda")
training_model = model_runner(model, 100, model_parameters, 0.001)
training_model(x, y)
