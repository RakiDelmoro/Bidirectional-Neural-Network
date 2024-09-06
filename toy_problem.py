import random
import torch
import sys

def generate_input_and_expected_pair(number_will_generated):
    list_of_target_weights = [3, 5, 7, 9]
    generated_pair_of_number = []
    for _ in range(number_will_generated):
        l2r_pair = random.randint(0, 100), random.randint(0, 100)

        r2l_int_1 = (l2r_pair[0] * list_of_target_weights[0]) + (l2r_pair[1] * list_of_target_weights[1])
        r2l_int_2 = (l2r_pair[0] * list_of_target_weights[2]) + (l2r_pair[1] * list_of_target_weights[3])
        r2l_pair = r2l_int_1, r2l_int_2

        generated_pair_of_number.append((torch.tensor(l2r_pair, dtype=torch.float32, device='cuda'), torch.tensor(r2l_pair, dtype=torch.float32, device='cuda')))

    return generated_pair_of_number

def calculate_local_gradient():
    pass

def neural_network(input_data, learning_rate=0.0001):    
    l2r_weight_1 = torch.tensor([0.1233], dtype=torch.float32, device="cuda")
    l2r_weight_2 = torch.tensor([0.5365], dtype=torch.float32, device="cuda")
    l2r_weight_3 = torch.tensor([0.1457], dtype=torch.float32, device="cuda")
    l2r_weight_4 = torch.tensor([0.1371], dtype=torch.float32, device="cuda")

    r2l_weight_1 = torch.tensor([0.2334], dtype=torch.float32, device="cuda")
    r2l_weight_2 = torch.tensor([0.5345], dtype=torch.float32, device="cuda")
    r2l_weight_3 = torch.tensor([0.6456], dtype=torch.float32, device="cuda")
    r2l_weight_4 = torch.tensor([0.6588], dtype=torch.float32, device="cuda")

    while True:
        l2r_input = input_data[0][0]
        r2l_input = input_data[0][1]

        print(f"target for forward pass output: {r2l_input}")
        print(f"target for backward pass output: {l2r_input}")
        # Forward pass
        l2r_output_1 = (l2r_input[0] * l2r_weight_1) + (l2r_input[1] * l2r_weight_2)
        l2r_output_2 = (l2r_input[0] * l2r_weight_3) + (l2r_input[1] * l2r_weight_4)
        # # backward pass
        r2l_output_1 = (r2l_input[0] * r2l_weight_1) + (r2l_input[1] * r2l_weight_2)
        r2l_output_2 = (r2l_input[0] * r2l_weight_3) + (r2l_input[1] * r2l_weight_4)

        # Left to right nodes loss
        l2r_output_node_1_loss = l2r_output_1 - r2l_input[0]
        l2r_output_node_2_loss = l2r_output_2 - r2l_input[1]

        # Right to left nodes loss
        r2l_output_node_1_loss = r2l_output_1 - l2r_input[0]
        r2l_output_node_2_loss = r2l_output_2 - l2r_input[1]

        # Left to right calculate local gradient
        l2r_weight_1_gradient = l2r_input[0] * l2r_output_node_1_loss
        l2r_weight_2_gradient = l2r_input[1] * l2r_output_node_1_loss
        l2r_weight_3_gradient = l2r_input[0] * l2r_output_node_2_loss
        l2r_weight_4_gradient = l2r_input[1] * l2r_output_node_2_loss
        # Right to left calculate local gradient
        r2l_weight_1_gradient = r2l_input[0] * r2l_output_node_1_loss
        r2l_weight_2_gradient = r2l_input[1] * r2l_output_node_1_loss
        r2l_weight_3_gradient = r2l_input[0] * r2l_output_node_2_loss
        r2l_weight_4_gradient = r2l_input[1] * r2l_output_node_2_loss

        # Left to right calculate update parameters
        l2r_weight_1 = l2r_weight_1 - learning_rate * l2r_weight_1_gradient
        l2r_weight_2 = l2r_weight_2 - learning_rate * l2r_weight_2_gradient
        l2r_weight_3 = l2r_weight_3 - learning_rate * l2r_weight_3_gradient
        l2r_weight_4 = l2r_weight_4 - learning_rate * l2r_weight_4_gradient
        # Right to left calculate update parameters
        r2l_weight_1 = r2l_weight_1 - learning_rate * r2l_weight_1_gradient
        r2l_weight_2 = r2l_weight_2 - learning_rate * r2l_weight_2_gradient
        r2l_weight_3 = r2l_weight_3 - learning_rate * r2l_weight_3_gradient
        r2l_weight_4 = r2l_weight_4 - learning_rate * r2l_weight_4_gradient

        print(f"Left to right output node: {l2r_output_1, l2r_output_2}")
        print(f"Right to left output node: {r2l_output_1, r2l_output_2}")
        print(f"Weight 1: {l2r_weight_1}, {l2r_weight_2}, {l2r_weight_3}, {l2r_weight_4}")

neural_network(generate_input_and_expected_pair(1))

# training_set = generate_input_pair(10)
# print(training_set)
