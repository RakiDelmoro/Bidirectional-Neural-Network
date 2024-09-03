import random
import torch
import sys

def generate_input_and_expected_pair(number_will_generated):
    list_of_target_weights = [3, 5, 7, 9]
    generated_pair_of_number = []
    for _ in range(number_will_generated):
        left_to_right_pair = random.randint(0, sys.maxsize), random.randint(0, sys.maxsize)

        right_to_left_int_1 = (left_to_right_pair[0] * list_of_target_weights[0]) + (left_to_right_pair[1] * list_of_target_weights[1])
        right_to_left_int_2 = (left_to_right_pair[0] * list_of_target_weights[2]) + (left_to_right_pair[1] * list_of_target_weights[3])
        right_to_left_pair = right_to_left_int_1, right_to_left_int_2
        
        generated_pair_of_number.append((torch.tensor(left_to_right_pair, dtype=torch.float32, device='cuda'), torch.tensor(right_to_left_pair, dtype=torch.float32, device='cuda')))

    return generated_pair_of_number

def neural_network(input_data, learning_rate=0.01):    
    weight_1 = torch.tensor([0.1233], dtype=torch.float32, device="cuda")
    weight_2 = torch.tensor([0.5365], dtype=torch.float32, device="cuda")
    weight_3 = torch.tensor([0.1457], dtype=torch.float32, device="cuda")
    weight_4 = torch.tensor([0.1371], dtype=torch.float32, device="cuda")

    while True:
        left_to_right_input = input_data[0][0]
        right_to_left_input = input_data[0][1]

        print(f"target for forward pass output: {right_to_left_input}")
        print(f"target for backward pass output: {left_to_right_input}")
        # Forward pass
        left_to_right_output_1 = (left_to_right_input[0] * weight_1) + (left_to_right_input[1] * weight_2)
        left_to_right_output_2 = (left_to_right_input[0] * weight_3) + (left_to_right_input[1] * weight_4)
        # # backward pass
        right_to_left_output_1 = (right_to_left_input[0] * weight_1) + (right_to_left_input[1] * weight_3)
        right_to_left_output_2 = (right_to_left_input[0] * weight_2) + (right_to_left_input[1] * weight_4)

        weight_1_forward_gradient = left_to_right_input[0] * left_to_right_output_1
        weight_2_forward_gradient = left_to_right_input[0] * left_to_right_output_1
        weight_3_forward_gradient = left_to_right_input[1] * left_to_right_output_2
        weight_4_forward_gradient = left_to_right_input[1] * left_to_right_output_2

        weight_1_backward_gradient = right_to_left_input[0] * right_to_left_output_1
        weight_2_backward_gradient = right_to_left_input[1] * right_to_left_output_2
        weight_3_backward_gradient = right_to_left_input[0] * right_to_left_output_1
        weight_4_backward_gradient = right_to_left_input[1] * right_to_left_output_2

        weight_1_avg_gradient = torch.mean(torch.tensor([weight_1_forward_gradient, weight_1_backward_gradient]))
        weight_2_avg_gradient = torch.mean(torch.tensor([weight_2_forward_gradient, weight_2_backward_gradient]))
        weight_3_avg_gradient = torch.mean(torch.tensor([weight_3_forward_gradient, weight_3_backward_gradient]))
        weight_4_avg_gradient = torch.mean(torch.tensor([weight_4_forward_gradient, weight_4_backward_gradient]))

        weight_1 = weight_1 - learning_rate * weight_1_avg_gradient
        weight_2 = weight_2 - learning_rate * weight_2_avg_gradient
        weight_3 = weight_3 - learning_rate * weight_3_avg_gradient
        weight_4 = weight_4 - learning_rate * weight_4_avg_gradient

        print(f"Left to right output node: {left_to_right_output_1, left_to_right_output_2}")
        print(f"Right to left output node: {right_to_left_output_1, right_to_left_output_2}")
        print(f"Weight 1: {weight_1}, {weight_2}, {weight_3}, {weight_4}")

neural_network(generate_input_and_expected_pair(1))

# training_set = generate_input_pair(10)
# print(training_set)
