import torch
import math

def nodes(weight: torch.Tensor):
    # weight initialization
    torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    def node_computation(input_feature: torch.Tensor):
        return torch.matmul(input_feature, weight)

    return node_computation

def network(feature_sizes: list, device: str):
    # Intialize network layers and parameters
    forward_pass_layers = []
    backward_pass_layers = []
    network_weights = []

    for each in range(len(feature_sizes)-1):
        in_feature =  feature_sizes[each]
        out_feature = feature_sizes[each+1]
        weights = torch.empty((in_feature, out_feature), device=device)
        forward_layer = nodes(weights)
        backward_layer = nodes(weights.t()) # Flip the weights
        
        network_weights.append(weights)
        forward_pass_layers.append(forward_layer)
        backward_pass_layers.insert(0, backward_layer)

    def forward_in_bidirectional(image_data, label_data):
        forward_pass_outputs = []
        backward_pass_outputs = []

        input_for_forward_pass = image_data
        input_for_backward_pass = label_data
        for each_layer in range(len(feature_sizes)-1):
            input_for_forward_pass = forward_pass_layers[each_layer](input_for_forward_pass)
            input_for_backward_pass = backward_pass_layers[each_layer](input_for_backward_pass)
            
            forward_pass_outputs.append(input_for_forward_pass)
            backward_pass_outputs.append(input_for_backward_pass)

        return torch.concat(forward_pass_outputs, dim=-1), torch.concat(backward_pass_outputs, dim=-1)

    return forward_in_bidirectional

x = torch.randn(10, device="cuda")
y = torch.randn(5, device="cuda")
model = network(feature_sizes=[10, 20, 20, 5], device="cuda")
forward, backward = model(x, y)
print(forward.shape, backward.shape)

