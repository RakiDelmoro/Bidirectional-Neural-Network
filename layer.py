import math
import torch
import torch.nn as nn
from torch import empty
from torch.nn import Parameter
from torch.nn.functional import relu

def weight_and_bias_initialization(weight, bias):
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    return nn.init.uniform_(bias, -bound, bound)

def nodes(input_features: int, output_features: int, device: str='cuda'):
    weight = Parameter(empty((output_features, input_features), device=device))
    bias = Parameter(empty(output_features, device=device))

    # weight and bias initialization
    weight_and_bias_initialization(weight, bias)

    def node_computation(input_feature: torch.Tensor, reverse_computation: bool):
        return relu(torch.matmul((input_feature + bias), weight)) if reverse_computation else relu(torch.matmul(input_feature, weight.t()) + bias)

    return node_computation, weight, bias

# input_data_test = torch.randn(10, device='cuda')
# node, w, b = nodes(10, 10)
# print(node(input_data_test))
