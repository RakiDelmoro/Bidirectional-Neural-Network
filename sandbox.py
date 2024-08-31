import torch
from torch.nn.functional import linear

def micro_nets(input_x, expected_y, learning_rate=0.001):
    '''
        Input shape = torch.Size(1, 2)
        Target shape = torch.Size(1, 2)
    '''

    # Parameters of input to hidden nodes
    input_to_hidden_weight_1 = torch.tensor([0.345], dtype=torch.float32, device="cuda")
    input_to_hidden_weight_2 = torch.tensor([1.839], dtype=torch.float32, device="cuda")
    input_to_hidden_weight_3 = torch.tensor([1.152], dtype=torch.float32, device="cuda")
    input_to_hidden_weight_4 = torch.tensor([0.946], dtype=torch.float32, device="cuda")
    hidden_bias = torch.tensor([[0.584, 0.325]], dtype=torch.float32, device="cuda")
    # Parameters of hidden nodes to output nodes
    hidden_to_output_weight_5 = torch.tensor([0.873], dtype=torch.float32, device="cuda")
    hidden_to_output_weight_6 = torch.tensor([0.645], dtype=torch.float32, device="cuda")
    hidden_to_output_weight_7 = torch.tensor([1.053], dtype=torch.float32, device="cuda")
    hidden_to_output_weight_8 = torch.tensor([1.079], dtype=torch.float32, device="cuda")
    output_bias = torch.tensor([[0, 0]], dtype=torch.float32, device="cuda")

    while True:
        # Forward pass input to hidden nodes
        hidden_node_1 = (input_x[0][0]*input_to_hidden_weight_1) + (input_x[0][1]*input_to_hidden_weight_3)
        hidden_node_2 = (input_x[0][0]*input_to_hidden_weight_2) + (input_x[0][1]*input_to_hidden_weight_4)
        hidden_nodes = torch.tensor([hidden_node_1, hidden_node_2], dtype=torch.float32, device="cuda") + hidden_bias

        # Forward pass hidden nodes to output nodes
        output_node_3 = (hidden_nodes[0][0]*hidden_to_output_weight_5) + (hidden_nodes[0][1]*hidden_to_output_weight_7)
        output_node_4 = (hidden_nodes[0][0]*hidden_to_output_weight_6) + (hidden_nodes[0][1]*hidden_to_output_weight_8)
        output_nodes = torch.tensor([output_node_3, output_node_4], dtype=torch.float32, device="cuda") + output_bias

        # Loss of the network
        loss = torch.mean(torch.tensor([(output_nodes_i - expected_y_i)**2 for output_nodes_i, expected_y_i in zip(output_nodes[0], expected_y[0])]))
        output_nodes_gradients = [(output_nodes_i - expected_y_i) for output_nodes_i, expected_y_i in zip(output_nodes[0], expected_y[0])]

        weight_5_gradient = 0
        weight_6_gradient = 0
        weight_7_gradient = 0
        weight_8_gradient = 0
        output_bias_gradient_1 = 0
        output_bias_gradient_2 = 0

        weight_1_gradient = 0
        weight_2_gradient = 0
        weight_3_gradient = 0
        weight_4_gradient = 0
        hidden_bias_gradient_1 = 0
        hidden_bias_gradient_2 = 0

        # Assign local gradient to output nodes to hidden nodes parameters
        weight_5_gradient += hidden_nodes[0][0] * output_nodes_gradients[0]
        weight_6_gradient += hidden_nodes[0][0] * output_nodes_gradients[1]
        weight_7_gradient += hidden_nodes[0][1] * output_nodes_gradients[0]
        weight_8_gradient += hidden_nodes[0][1] * output_nodes_gradients[1]
        output_bias_gradient_1 += output_nodes_gradients[0]
        output_bias_gradient_2 += output_nodes_gradients[1]

        # Assign local gradient to hidden nodes
        hidden_nodes_gradients = [
            (output_nodes_gradients[0] * hidden_to_output_weight_5) + (output_nodes_gradients[1] * hidden_to_output_weight_6),
            (output_nodes_gradients[0] * hidden_to_output_weight_7) + (output_nodes_gradients[1] * hidden_to_output_weight_8)
        ]

        # Assign local gradient to hidden nodes to input feature parameters
        weight_1_gradient += input_x[0][0] * hidden_nodes_gradients[0]
        weight_2_gradient += input_x[0][0] * hidden_nodes_gradients[1]
        weight_3_gradient += input_x[0][1] * hidden_nodes_gradients[0]
        weight_4_gradient += input_x[0][1] * hidden_nodes_gradients[1]
        hidden_bias_gradient_1 += hidden_nodes_gradients[0]
        hidden_bias_gradient_2 += hidden_nodes_gradients[1]

        # Update output nodes to hidden nodes parameters
        new_weight_5 = hidden_to_output_weight_5 - learning_rate * weight_5_gradient
        new_weight_6 = hidden_to_output_weight_6 - learning_rate * weight_6_gradient
        new_weight_7 = hidden_to_output_weight_7 - learning_rate * weight_7_gradient
        new_weight_8 = hidden_to_output_weight_8 - learning_rate * weight_8_gradient
        new_output_bias_1 = output_bias[0][0] - learning_rate * output_bias_gradient_1
        new_output_bias_2 = output_bias[0][1] - learning_rate * output_bias_gradient_2

        hidden_to_output_weight_5 = new_weight_5
        hidden_to_output_weight_6 = new_weight_6
        hidden_to_output_weight_7 = new_weight_7
        hidden_to_output_weight_8 = new_weight_8
        output_bias = torch.tensor([[new_output_bias_1, new_output_bias_2]], dtype=torch.float32, device="cuda")

        # Update hidden nodes to input feature parameters 
        new_weight_1 = input_to_hidden_weight_1 - learning_rate * weight_1_gradient
        new_weight_2 = input_to_hidden_weight_2 - learning_rate * weight_2_gradient
        new_weight_3 = input_to_hidden_weight_3 - learning_rate * weight_3_gradient
        new_weight_4 = input_to_hidden_weight_4 - learning_rate * weight_4_gradient
        new_hidden_bias_1 = hidden_bias[0][0] - learning_rate * hidden_bias_gradient_1
        new_hidden_bias_2 = hidden_bias[0][1] - learning_rate * hidden_bias_gradient_2 

        input_to_hidden_weight_1 = new_weight_1
        input_to_hidden_weight_2 = new_weight_2
        input_to_hidden_weight_3 = new_weight_3
        input_to_hidden_weight_4 = new_weight_4
        hidden_bias = torch.tensor([[new_hidden_bias_1, new_hidden_bias_2]], dtype=torch.float32, device="cuda")

        yield f"Loss: {loss.item()} Neuron activation: {output_nodes}"

def pytorch_micro_nets(input_x, expected_y, lr=0.001):
    input_to_hidden_weights = torch.tensor([[0.345, 1.152], [1.839, 0.946]], dtype=torch.float32, device="cuda", requires_grad=True)
    hidden_bias = torch.tensor([[0.584, 0.325]], dtype=torch.float32, device="cuda", requires_grad=True)
    hidden_to_output_weights = torch.tensor([[0.873, 1.053], [0.645, 1.079]], dtype=torch.float32, device="cuda", requires_grad=True)
    output_bias = torch.tensor([[0, 0]], dtype=torch.float32, device="cuda", requires_grad=True)

    parameters = [input_to_hidden_weights, hidden_bias, hidden_to_output_weights, output_bias]

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(parameters, lr=lr)

    while True:
        hidden_nodes = linear(input_x, input_to_hidden_weights, hidden_bias)
        output_nodes = linear(hidden_nodes, hidden_to_output_weights, output_bias)
        hidden_nodes.retain_grad()
        output_nodes.retain_grad()
        loss = loss_func(output_nodes, expected_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        yield f"Loss: {loss.item()} Neuron activation: {output_nodes}"

def two_input_one_neuron(input_data, expected, lr):
    # W1 and W2
    weights = torch.tensor([[0.5, 0.1]], dtype=torch.float32, device="cuda")
    bias = torch.tensor([[1]], dtype=torch.float32, device="cuda")

    for epoch in range(1, 10):
        print(f"Epoch: {epoch}")
        for each_data in range(input_data.shape[0]):
            batch_of_input = input_data[each_data].unsqueeze(0)
            expected_y = expected[each_data].unsqueeze(0)
            neuron = torch.sum(batch_of_input * weights) + bias
            neuron_loss = (neuron - expected_y)**2

            # Equivalent to optimizer.zero_grad() -> start with zero gradient
            weight_1_grad = 0
            weight_2_grad = 0
            bias_grad = 0
            neuron_grad = 0

            # Equivalent to loss.backward -> calculate local gradient for each parameter
            neuron_grad += 2 * (neuron - expected_y)
            weight_1_grad += input_data[each_data][0] * neuron_grad
            weight_2_grad += input_data[each_data][1] * neuron_grad
            bias_grad += neuron_grad

            # Equivalent to optimizer.step() -> update the parameters 
            new_weight1 = weights[0][0] - lr * weight_1_grad
            new_weight2 = weights[0][1] - lr * weight_2_grad
            new_bias = bias - lr * bias_grad

            weights[0][0] = new_weight1
            weights[0][1] = new_weight2
            bias = new_bias

            print(f"batch of {each_data+1} Loss: {neuron_loss.item()} Neuron activation: {neuron}")

def pytorch_two_input_one_neuron(input_data, expected, lr):
    weights = torch.tensor([[0.5]], dtype=torch.float32, device="cuda", requires_grad=True)
    bias = torch.tensor([[0]], dtype=torch.float32, device="cuda", requires_grad=True)

    parameters = [weights, bias]

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(parameters, lr=lr)

    for epoch in range(1, 10):
        print(f"Epoch: {epoch}")
        for each in range(input_data.shape[0]):
            expected_y = expected[each].unsqueeze(0)
            neuron = linear(input_data[each].unsqueeze(0), weights.t(), bias)
            neuron_loss = loss_func(neuron, expected_y)
            optimizer.zero_grad()
            neuron_loss.backward()
            optimizer.step()
            print(f"batch of {each+1} Loss: {neuron_loss.item()} Neuron activation: {neuron}")

def custom_one_to_two(input_data, expected, learning_rate):
    input_to_hidden_nodes_weights = torch.tensor([0.24, 0.53], dtype=torch.float32, device="cuda")
    input_to_hidden_bias = torch.tensor([2.01, 1.23], dtype=torch.float32, device="cuda")
    hidden_to_output_weights = torch.tensor([[0.1534, 0.1412], [0.881, 0.9541]], dtype=torch.float32, device="cuda")
    hidden_to_output_bias = torch.tensor([0.175, 0.231], dtype=torch.float32, device="cuda")

    for epoch in range(1, 10):
        print(f"Epoch: {epoch}")
        for each_data in range(input_data.shape[0]):
            batch_of_input = input_data[each_data].unsqueeze(0)
            expected_y = expected[each_data].unsqueeze(0)
            input_to_hidden_neuron = (batch_of_input * input_to_hidden_nodes_weights) + input_to_hidden_bias
            hidden_to_output_neuron = torch.matmul(input_to_hidden_neuron, hidden_to_output_weights) + hidden_to_output_bias

            neuron_loss = torch.mean((hidden_to_output_neuron - expected_y)**2)

            # zero gradients
            input_to_hidden_weight_1_grad = 0
            input_to_hidden_weight_2_grad = 0
            input_to_hidden_bias_1_grad = 0
            input_to_hidden_bias_2_grad = 0

            hidden_to_output_weight_1_grad = 0
            hidden_to_output_weight_2_grad = 0
            hidden_to_output_weight_3_grad = 0
            hidden_to_output_weight_4_grad = 0
            hidden_to_output_bias_grad_1 = 0
            hidden_to_output_bias_grad_2 = 0
            neuron_grad = 0

            # backpropagation
            neuron_grad += 2 * (hidden_to_output_neuron - expected_y)
            hidden_to_output_weight_1_grad += hidden_to_output_neuron[0][0] * neuron_grad[0][0]
            hidden_to_output_weight_2_grad += hidden_to_output_neuron[0][1] * neuron_grad[0][0]
            hidden_to_output_weight_3_grad += hidden_to_output_neuron[0][0] * neuron_grad[0][1]
            hidden_to_output_weight_4_grad += hidden_to_output_neuron[0][1] * neuron_grad[0][1]
            hidden_to_output_bias_grad_1 += neuron_grad[0][0]
            hidden_to_output_bias_grad_2 += neuron_grad[0][1]

            # TODO: Fix the calculation of hidden nodes gradient
            hidden_nodes_gradients = [
                (neuron_grad[0][0] * hidden_to_output_weights[0][0]) + (neuron_grad[0][1] * hidden_to_output_weights[1][0]),
                (neuron_grad[0][0] * hidden_to_output_weights[0][1]) + (neuron_grad[0][1] * hidden_to_output_weights[1][1])
            ]

            input_to_hidden_weight_1_grad += input_data[each_data][0] * hidden_nodes_gradients[0]
            input_to_hidden_weight_2_grad += input_data[each_data][0] * hidden_nodes_gradients[1]
            input_to_hidden_bias_1_grad += hidden_nodes_gradients[0]
            input_to_hidden_bias_2_grad += hidden_nodes_gradients[1] 

            # Update parameters
            new_hidden_to_output_weight_1 = hidden_to_output_weights[0][0] - learning_rate * hidden_to_output_weight_1_grad
            new_hidden_to_output_weight_2 = hidden_to_output_weights[1][0] - learning_rate * hidden_to_output_weight_2_grad
            new_hidden_to_output_weight_3 = hidden_to_output_weights[0][1] - learning_rate * hidden_to_output_weight_3_grad
            new_hidden_to_output_weight_4 = hidden_to_output_weights[1][1] - learning_rate * hidden_to_output_weight_4_grad
            new_hidden_to_output_bias_1 = hidden_to_output_bias[0] - learning_rate * hidden_to_output_bias_grad_1
            new_hidden_to_output_bias_2 = hidden_to_output_bias[1] - learning_rate * hidden_to_output_bias_grad_2

            new_input_to_hidden_weight_1 = input_to_hidden_nodes_weights[0] - learning_rate * input_to_hidden_weight_1_grad
            new_input_to_hidden_weight_2 = input_to_hidden_nodes_weights[1] - learning_rate * input_to_hidden_weight_2_grad
            new_input_to_hidden_bias_1 = input_to_hidden_bias[0] - learning_rate * input_to_hidden_bias_1_grad
            new_input_to_hidden_bias_2 = input_to_hidden_bias[1] - learning_rate * input_to_hidden_bias_2_grad

            hidden_to_output_weights[0][0] = new_hidden_to_output_weight_1
            hidden_to_output_weights[1][0] = new_hidden_to_output_weight_2
            hidden_to_output_weights[0][1] = new_hidden_to_output_weight_3
            hidden_to_output_weights[1][1] = new_hidden_to_output_weight_4
            hidden_to_output_bias[0] = new_hidden_to_output_bias_1
            hidden_to_output_bias[1] = new_hidden_to_output_bias_2

            input_to_hidden_nodes_weights[0] = new_input_to_hidden_weight_1
            input_to_hidden_nodes_weights[1] = new_input_to_hidden_weight_2
            input_to_hidden_bias[0] = new_input_to_hidden_bias_1
            input_to_hidden_bias[1] = new_input_to_hidden_bias_2

            print(f"batch of {each_data+1} Loss: {neuron_loss.item()} Neuron activation: {hidden_to_output_neuron}")


test_input_data = torch.tensor([[1], [2], [3]], dtype=torch.float32, device="cuda")
test_expected = torch.tensor([[4, 5],[6, 7],[8, 9]], dtype=torch.float32, device="cuda")
learning_rate = 0.001

def pytorch_one_to_two(input_data, expected, learning_rate):
    input_to_hidden_nodes_weights = torch.tensor([[0.24, 0.53]], dtype=torch.float32, device="cuda", requires_grad=True)
    input_to_hidden_bias = torch.tensor([2.01, 1.23], dtype=torch.float32, device="cuda", requires_grad=True)
    hidden_to_output_weights = torch.tensor([[0.1534, 0.1412], [0.881, 0.9541]], dtype=torch.float32, device="cuda", requires_grad=True)
    hidden_to_output_bias = torch.tensor([0.175, 0.231], dtype=torch.float32, device="cuda", requires_grad=True)

    parameters = [input_to_hidden_nodes_weights, input_to_hidden_bias, hidden_to_output_weights, hidden_to_output_bias]
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)

    for epoch in range(1, 10):
        print(f"Epoch: {epoch}")
        for each in range(input_data.shape[0]):
            expected_y = expected[each].unsqueeze(0)
            input_to_hidden_neuron = linear(input_data[each].unsqueeze(0), input_to_hidden_nodes_weights.t(), input_to_hidden_bias)
            hidden_to_output_neuron = linear(input_to_hidden_neuron, hidden_to_output_weights.t(), hidden_to_output_bias)
            hidden_to_output_neuron.retain_grad()
            neuron_loss = loss_func(hidden_to_output_neuron, expected_y)
            optimizer.zero_grad()
            neuron_loss.backward()
            optimizer.step()
            print(f"batch of {each+1} Loss: {neuron_loss.item()} Neuron activation: {hidden_to_output_neuron}")

custom_one_to_two(test_input_data, test_expected, learning_rate)
pytorch_one_to_two(test_input_data, test_expected, learning_rate)

# for epoch in range(1, 10, 1):
    # # for py_result, custom_result in zip(pytorch_two_input_one_neuron(input_data, expected, learning_rate), two_input_one_neuron(input_data, expected, learning_rate)):
    #   for py_result in pytorch_two_input_one_neuron(input_data, expected, learning_rate):
    #     print(f"Epoch: {epoch}")
    #     print(py_result)
    #     # print(custom_result)
    #     epoch += 1
