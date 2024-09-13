import torch
import random
import sys
from torch.nn.functional import linear

def generate_input_and_expected_pair(number_will_generated):
    # [[w1, w2], [w3, w4]]
    target_weights = torch.tensor([[3, 5], [7, 9]], dtype=torch.double, device="cuda").inverse()

    generated_pair_of_number = []
    for _ in range(number_will_generated):
        l2r_pair = torch.tensor([[random.randint(0, 2**32), random.randint(0, 2**32)],
                                 [random.randint(0, 2**32), random.randint(0, 2**32)],
                                 [random.randint(0, 2**32), random.randint(0, 2**32)]], dtype=torch.double, device="cuda")
        r2l_pair = (torch.matmul(l2r_pair, target_weights))
        generated_pair_of_number.append((l2r_pair, r2l_pair))

    return generated_pair_of_number

def neural_network_two_to_two(data):
    learning_rate=0.001
    """
    # - Input feature
    * - Neuron

    #      *
       ->
    #      *
    """

    # Define parameters - [[w1, w2], [w3, w4]]
    l2r_weights = torch.tensor([[0.1233, 0.5365], [0.1457, 0.1371]], dtype=torch.double, device="cuda")
    r2l_weights = l2r_weights.inverse()

    counter = 0
    # while True:
    # Use the inverse value of l2r weights for right to left weights
    # r2l_weights = l2r_weights.inverse()

    # Prepare Input data
    # data = generate_input_and_expected_pair(1)
    l2r_input_data = data[0][0]
    r2l_input_data = data[0][1]

    # forward pass
    l2r_output_nodes = (torch.matmul(l2r_input_data, l2r_weights))
    # backward pass
    r2l_output_nodes = (torch.matmul(r2l_input_data, r2l_weights))
    # Left to right nodes loss
    output_nodes_loss = (l2r_output_nodes - r2l_input_data)**2
    l2r_output_nodes_loss = (l2r_output_nodes - r2l_input_data)
    # Right to left nodes loss
    r2l_output_nodes_loss = 2 * (r2l_output_nodes - l2r_input_data)

    # Initialize gradients to ZERO
    l2r_weights_gradients = torch.zeros_like(l2r_weights)
    r2l_weights_gradients = torch.zeros_like(r2l_weights)
    
    # Left to right calculate local gradient
    l2r_weights_gradients[0, 0] = torch.sum(l2r_input_data[:, 0] * l2r_output_nodes_loss[:, 0])
    l2r_weights_gradients[0, 1] = torch.sum(l2r_input_data[:, 0] * l2r_output_nodes_loss[:, 1])
    l2r_weights_gradients[1, 0] = torch.sum(l2r_input_data[:, 1] * l2r_output_nodes_loss[:, 0])
    l2r_weights_gradients[1, 1] = torch.sum(l2r_input_data[:, 1] * l2r_output_nodes_loss[:, 1])
    # Right to left calculate local gradient
    r2l_weights_gradients[0, 0] = r2l_input_data[0, 0] * r2l_output_nodes_loss[0, 0]
    r2l_weights_gradients[0, 1] = r2l_input_data[0, 0] * r2l_output_nodes_loss[0, 1]
    r2l_weights_gradients[1, 0] = r2l_input_data[0, 1] * r2l_output_nodes_loss[0, 0]
    r2l_weights_gradients[1, 1] = r2l_input_data[0, 1] * r2l_output_nodes_loss[0, 1]

    # Normalize local gradients by batch size
    batch_size = l2r_input_data.shape[0]
    l2r_weights_gradients /= batch_size
    r2l_weights_gradients /= batch_size

    # Left to right calculate update parameters
    l2r_weights -= learning_rate * l2r_weights_gradients
    # Right to left calculate update parameters
    r2l_weights -= learning_rate * r2l_weights_gradients

    return f"Loss: {torch.mean(output_nodes_loss)} Weights: {l2r_weights.tolist()}"
    counter += 1

def pytorch_neural_network(data):
    weights = torch.tensor([[0.1233, 0.5365], [0.1457, 0.1371]], dtype=torch.double, device="cuda", requires_grad=True)

    parameters = [weights]

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(parameters, 0.001)

    # while True:
        # data = generate_input_and_expected_pair(1)
    l2r_input_data = data[0][0]
    r2l_input_data = data[0][1]

    output_nodes = linear(l2r_input_data, weights.t(), None)
    output_nodes.retain_grad()
    nodes_loss = loss_function(output_nodes, r2l_input_data)
    optimizer.zero_grad()
    nodes_loss.backward()
    optimizer.step()
    return f'Loss: {nodes_loss.item()} Weights: {weights.tolist()}'
        # yield f'Weights: {weights}'

epoch = 1
while True:
    data = generate_input_and_expected_pair(1)
    py_result = pytorch_neural_network(data)
    custom_result = neural_network_two_to_two(data)
    print(f"Epoch: {epoch}")
    print(f"Result from pytorch: {py_result}")
    print(f"Result from custom model: {custom_result}")

    if epoch == 10000:
        break

    epoch += 1

# neural_network_two_to_two()

# training_set = generate_input_pair(10)
# print(training_set)
