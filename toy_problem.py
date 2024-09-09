import random
import torch

def generate_input_and_expected_pair(number_will_generated):
    # [[w1, w2], [w3, w4]]
    # target_weights = torch.tensor([[-9/8, 5/8], [7/8, -3/8]], dtype=torch.float32, device="cuda")
    target_weights = torch.tensor([[3, 5], [7, 9]], dtype=torch.float32, device="cuda")

    generated_pair_of_number = []
    for _ in range(number_will_generated):
        l2r_pair = torch.tensor([[random.randint(0, 10),random.randint(0, 10)]], dtype=torch.float32, device="cuda")
        r2l_pair = torch.matmul(l2r_pair, target_weights)
        generated_pair_of_number.append((l2r_pair, r2l_pair))

    return generated_pair_of_number

def neural_network(learning_rate=0.001):
    # [[w1, w2], [w3, w4]]
    l2r_weights = torch.tensor([[0.1233, 0.5365], [0.1457, 0.1371]], dtype=torch.float32, device="cuda")
    r2l_weights = torch.tensor([[0.2334, 0.5345], [0.6456, 0.6588]], dtype=torch.float32, device="cuda")
    # r2l_weights = l2r_weights

    counter = 0

    while True:
        # Training AI MODEL
        data = generate_input_and_expected_pair(1)
        l2r_input_data = data[0][0]
        r2l_input_data = data[0][1]

        print(f"target for left to right: {r2l_input_data}")
        print(f"target for right to left: {l2r_input_data}")
        # forward pass
        l2r_output_nodes = torch.matmul(l2r_input_data, l2r_weights)
        # backward pass
        r2l_output_nodes = torch.matmul(l2r_output_nodes, r2l_weights)
        # Left to right nodes loss
        l2r_output_nodes_loss = 2 * (l2r_output_nodes - r2l_input_data)
        # Right to left nodes loss
        r2l_output_nodes_loss = 2 * (r2l_output_nodes - l2r_input_data)

        # If encountered nan value immediately stop training for debugging purposes
        if torch.isnan(r2l_output_nodes).any().item():
            print(f'Nan value encountered! for {counter} runs')
            break

        # Left to right calculate local gradient
        l2r_weight_1_gradient = l2r_input_data[0, 0] * l2r_output_nodes_loss[0, 0]
        l2r_weight_2_gradient = l2r_input_data[0, 0] * l2r_output_nodes_loss[0, 1]
        l2r_weight_3_gradient = l2r_input_data[0, 1] * l2r_output_nodes_loss[0, 0]
        l2r_weight_4_gradient = l2r_input_data[0, 1] * l2r_output_nodes_loss[0, 1]
        # Right to left calculate local gradient
        r2l_weight_1_gradient = r2l_input_data[0, 0] * r2l_output_nodes_loss[0, 0]
        r2l_weight_2_gradient = r2l_input_data[0, 0] * r2l_output_nodes_loss[0, 1]
        r2l_weight_3_gradient = r2l_input_data[0, 1] * r2l_output_nodes_loss[0, 0]
        r2l_weight_4_gradient = r2l_input_data[0, 1] * r2l_output_nodes_loss[0, 1]

        # Left to right calculate update parameters
        l2r_weight_1 = l2r_weights[0, 0] - learning_rate * l2r_weight_1_gradient
        l2r_weight_2 = l2r_weights[0, 1] - learning_rate * l2r_weight_2_gradient
        l2r_weight_3 = l2r_weights[1, 0] - learning_rate * l2r_weight_3_gradient
        l2r_weight_4 = l2r_weights[1, 1] - learning_rate * l2r_weight_4_gradient
        # Right to left calculate update parameters
        r2l_weight_1 = r2l_weights[0, 0] - learning_rate * r2l_weight_1_gradient
        r2l_weight_2 = r2l_weights[0, 1] - learning_rate * r2l_weight_2_gradient
        r2l_weight_3 = r2l_weights[1, 0] - learning_rate * r2l_weight_3_gradient
        r2l_weight_4 = r2l_weights[1, 1] - learning_rate * r2l_weight_4_gradient

        # Assign the new updated weights
        l2r_weights[0, 0] = l2r_weight_1
        l2r_weights[0, 1] = l2r_weight_2
        l2r_weights[1, 0] = l2r_weight_3
        l2r_weights[1, 1] = l2r_weight_4

        r2l_weights[0, 0] = r2l_weight_1
        r2l_weights[0, 1] = r2l_weight_2
        r2l_weights[1, 0] = r2l_weight_3
        r2l_weights[1, 1] = r2l_weight_4

        print(f"Left to right output node: {l2r_output_nodes}")
        print(f"Right to left output node: {r2l_output_nodes}")
        print(f"Left to right weights: [[{l2r_weight_1}, {l2r_weight_2}], [{l2r_weight_3}, {l2r_weight_4}]]")
        print(f"Right to left weights: [[{r2l_weight_1}, {r2l_weight_2}], [{r2l_weight_3}, {r2l_weight_4}]]")
        counter += 1

neural_network()

# training_set = generate_input_pair(10)
# print(training_set)
