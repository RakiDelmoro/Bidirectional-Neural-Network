import random
import torch

def generate_input_and_expected_pair(number_will_generated):
    generated_pair_of_number = []
    for _ in range(number_will_generated):
        tensor_int_pair = torch.tensor([random.randint(0, 100), random.randint(0, 100)], device="cuda", dtype=torch.float32), torch.tensor([random.randint(0, 100), random.randint(0, 100)], device="cuda", dtype=torch.float32)
        generated_pair_of_number.append(tensor_int_pair)

    return generated_pair_of_number

def neural_network(input_data, learning_rate=0.01):    
    weight_1 = torch.tensor([0.123], dtype=torch.float32, device="cuda")
    weight_2 = torch.tensor([0.536], dtype=torch.float32, device="cuda")
    weight_3 = torch.tensor([0.145], dtype=torch.float32, device="cuda")
    weight_4 = torch.tensor([0.137], dtype=torch.float32, device="cuda")

    while True:
        input_for_forward_pass = input_data[0][0]
        input_for_backward_pass = input_data[0][1]
        
        print(f"target for forward pass output: {input_for_backward_pass}")
        print(f"target for backward pass output: {input_for_forward_pass}")
        # Forward pass
        output_node_1 = (input_for_forward_pass[0] * weight_1) + (input_for_forward_pass[1] * weight_2)
        output_node_2 = (input_for_forward_pass[0] * weight_3) + (input_for_forward_pass[1] * weight_4)
        # backward pass
        input_node_1 = (input_for_backward_pass[0] * weight_1) + (input_for_backward_pass[1] * weight_3)
        input_node_2 = (input_for_backward_pass[0] * weight_2) + (input_for_backward_pass[1] * weight_4)

        weight_1_forward_gradient = input_for_forward_pass[0] * output_node_1
        weight_2_forward_gradient = input_for_forward_pass[1] * output_node_1
        weight_3_forward_gradient = input_for_forward_pass[0] * output_node_2
        weight_4_forward_gradient = input_for_forward_pass[1] * output_node_2

        weight_1_backward_gradient = input_for_backward_pass[0] * input_node_1
        weight_2_backward_gradient = input_for_backward_pass[1] * input_node_1
        weight_3_backward_gradient = input_for_backward_pass[0] * input_node_2
        weight_4_backward_gradient = input_for_backward_pass[1] * input_node_2

        weight_1_avg_gradient = torch.mean(torch.tensor([weight_1_forward_gradient, weight_1_backward_gradient]))
        weight_2_avg_gradient = torch.mean(torch.tensor([weight_2_forward_gradient, weight_2_backward_gradient]))
        weight_3_avg_gradient = torch.mean(torch.tensor([weight_3_forward_gradient, weight_3_backward_gradient]))
        weight_4_avg_gradient = torch.mean(torch.tensor([weight_4_forward_gradient, weight_4_backward_gradient]))

        weight_1 = weight_1 - learning_rate * weight_1_avg_gradient
        weight_2 = weight_2 - learning_rate * weight_2_avg_gradient
        weight_3 = weight_3 - learning_rate * weight_3_avg_gradient
        weight_4 = weight_4 - learning_rate * weight_4_avg_gradient

        print(output_node_1, output_node_2)
        print(input_node_1, input_node_2)

neural_network(generate_input_and_expected_pair(1))

# training_set = generate_input_pair(10)
# print(training_set)
