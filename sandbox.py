import torch
from torch.nn.functional import linear

def micro_nets():
    # Input Data
    input_x = torch.tensor([0, 0], dtype=torch.float32, device="cuda")
    expected_y = torch.tensor([0, 1], dtype=torch.float32, device="cuda")

    # Parameters of input to hidden nodes
    input_to_hidden_weight_1 = torch.tensor([0.345], dtype=torch.float32, device="cuda")
    input_to_hidden_weight_2 = torch.tensor([1.839], dtype=torch.float32, device="cuda")
    input_to_hidden_weight_3 = torch.tensor([1.152], dtype=torch.float32, device="cuda")
    input_to_hidden_weight_4 = torch.tensor([0.946], dtype=torch.float32, device="cuda")
    hidden_bias = torch.tensor([[0.584], [0.325]], dtype=torch.float32, device="cuda")

    # Forward pass input to hidden nodes
    hidden_node_1 = (input_x[0]*input_to_hidden_weight_1) + (input_x[1]*input_to_hidden_weight_3)
    hidden_node_2 = (input_x[0]*input_to_hidden_weight_2) + (input_x[1]*input_to_hidden_weight_4)
    hidden_nodes = (hidden_node_1 + hidden_node_2) + hidden_bias

    # Parameters of hidden nodes to output nodes
    hidden_to_output_weight_5 = torch.tensor([0.873], dtype=torch.float32, device="cuda")
    hidden_to_output_weight_6 = torch.tensor([0.645], dtype=torch.float32, device="cuda")
    hidden_to_output_weight_7 = torch.tensor([1.053], dtype=torch.float32, device="cuda")
    hidden_to_output_weight_8 = torch.tensor([1.079], dtype=torch.float32, device="cuda")
    output_bias = torch.tensor([[1.064], [0.857]], dtype=torch.float32, device="cuda")

    # Forward pass hidden nodes to output nodes
    output_node_3 = (hidden_nodes[0]*hidden_to_output_weight_5) + (hidden_nodes[1]*hidden_to_output_weight_7)
    output_node_4 = (hidden_nodes[0]*hidden_to_output_weight_6) + (hidden_nodes[1]*hidden_to_output_weight_8)
    output_nodes = (output_node_3 + output_node_4) + output_bias

    # Loss of the network
    loss = sum([(output_nodes_i - expected_y_i)**2 for output_nodes_i, expected_y_i in zip(output_nodes, expected_y)])
   
    output_nodes_gradients = [2 * (output_nodes_i - expected_y_i) for output_nodes_i, expected_y_i in zip(output_nodes, expected_y)]
    # Gradient for each output nodes
    output_node_1_actual_data_and_gradient = (output_nodes[0], output_nodes_gradients[0])
    output_node_2_actual_data_and_gradient = (output_nodes[1], output_nodes_gradients[1])

    # Get the gradient of each parameter in output nodes
    weight_5_gradient = output_node_1_actual_data_and_gradient[1] * output_node_1_actual_data_and_gradient[0]
    weight_6_gradient = output_node_2_actual_data_and_gradient[1] * output_node_2_actual_data_and_gradient[0]
    weight_7_gradient = output_node_1_actual_data_and_gradient[1] * output_node_1_actual_data_and_gradient[0]
    weight_8_gradient = output_node_2_actual_data_and_gradient[1] * output_node_2_actual_data_and_gradient[0]

    pass

input_data = torch.tensor([[5, 15]], dtype=torch.float32, device="cuda")
expected = torch.tensor([[5]], dtype=torch.float32, device="cuda")
learning_rate = 0.001
    
def two_input_one_neuron(input_data, expected, lr):
    # W1 and W2
    weights = torch.tensor([[0.9], [0.2]], dtype=torch.float32, device="cuda")
    bias = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")

    while True:
        weighted_sum = torch.sum(torch.tensor([input_i * w_i for input_feature in input_data for input_i, w_i in zip(input_feature, weights)]))
        neuron = weighted_sum + bias
        neuron_loss = (neuron - expected)**2

        # Equavalent to optimizer.zero_grad() -> start with zero gradient
        weight_1_grad = 0
        weight_2_grad = 0
        bias_grad = 0
        neuron_grad = 0

        # Equavalent to loss.backward -> calculate local gradient for each parameter
        neuron_grad += 2 * (neuron - expected)
        weight_1_grad += input_data[0][0] * neuron_grad
        weight_2_grad += input_data[0][1] * neuron_grad
        bias_grad += neuron_grad

        # Equavalent to optimizer.step() -> update the parameters 
        new_weight1 = weights[0] - lr * weight_1_grad
        new_weight2 = weights[1] - lr * weight_2_grad
        new_bias = bias - lr * bias_grad

        weights[0] = new_weight1
        weights[1] = new_weight2
        bias = new_bias

        yield f"Loss: {neuron_loss.item()} Neuron activation: {neuron}"

def pytorch_two_input_one_neuron(input_data, expected, lr):
    weights = torch.tensor([[0.9], [0.2]], dtype=torch.float32, device="cuda", requires_grad=True)
    bias = torch.tensor([[1.0]], dtype=torch.float32, device="cuda", requires_grad=True)

    parameters = [weights, bias]

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(parameters, lr=lr)

    while True:
        neuron = linear(input_data, weights.t(), bias)
        neuron_loss = loss_func(neuron, expected)
        optimizer.zero_grad()
        neuron_loss.backward()
        optimizer.step()

        yield f"Loss: {neuron_loss.item()} Neuron activation: {neuron}"

for epoch in range(1, 10, 1):
    for py_result, custom_result in zip(pytorch_two_input_one_neuron(input_data, expected, learning_rate), two_input_one_neuron(input_data, expected, learning_rate)):
        print(f"Epoch: {epoch}")
        print(py_result)
        print(custom_result)
        epoch += 1