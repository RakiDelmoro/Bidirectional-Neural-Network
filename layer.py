import math
import torch
import torch.nn as nn
from torch import empty
from torch.nn import Parameter
from torch.nn.functional import relu

def weight_and_bias_initialization(weight, bias):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    return nn.init.kaiming_uniform_(weight, a=math.sqrt(5)),  nn.init.uniform_(bias, -bound, bound)

def linear_layer(input_feature: int, output_feature: int, device: str):
    weight = Parameter(torch.empty((output_feature, input_feature), device=device))
    # bias = Parameter(torch.empty(output_feature, device=device))
    torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    # weight_and_bias_initialization(weight, bias)

    def layer_computation(x: torch.Tensor):
        return relu(torch.matmul(x, weight.t()))

    return layer_computation, weight
