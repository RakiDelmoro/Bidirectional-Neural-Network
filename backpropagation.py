import numpy as np
    
def mlp_network(input_feature, hidden_feature, output_feature):
    # initialize weights
    def weight_and_bias_initialization():
        input_to_hidden_weights = np.random.randn(input_feature, hidden_feature)
        hidden_to_output_weights = np.random.randn(hidden_feature, output_feature)

        bias_hidden = np.zeros((1, hidden_feature))
        bias_output = np.zeros((1, output_feature))

        return [input_to_hidden_weights, bias_hidden], [hidden_to_output_weights, bias_output]

    sigmoid_activation_not_derivate = lambda input_x: 1 / (1 + np.exp(-input_x))
    sigmoid_activation_derivative = lambda input_x: input_x * (1 - input_x)

    def forward_in_network(input_data):
        input_to_hidden_weight_and_bias, hidden_to_output_weight_and_bias = weight_and_bias_initialization()

        # Input to Hidden layer
        hidden_activation = np.dot(input_data, input_to_hidden_weight_and_bias[0]) + input_to_hidden_weight_and_bias[1]
        hidden_output_nodes = sigmoid_activation_not_derivate(hidden_activation)

        # Hidden to output layer
        output_activation = np.dot(hidden_output_nodes, hidden_to_output_weight_and_bias[0]) + hidden_to_output_weight_and_bias[1]
        network_prediction = sigmoid_activation_not_derivate(output_activation)

        return network_prediction, hidden_output_nodes

    def backward_in_network(input_data, network_prediction, hidden_output_nodes, network_error, learning_rate):
        input_to_hidden_weight_and_bias, hidden_to_output_weight_and_bias = weight_and_bias_initialization()

        # Compute the output layer error
        network_last_layer_delta = network_error * sigmoid_activation_derivative(network_prediction)

        # Compute the hidden layer error
        network_hidden_layer_error = np.dot(network_last_layer_delta, hidden_to_output_weight_and_bias[0].T)
        network_hidden_layer_delta = network_hidden_layer_error * sigmoid_activation_derivative(hidden_output_nodes)

        # output to hidden weight update
        hidden_to_output_weight_and_bias[0] += np.dot(hidden_output_nodes.T, network_last_layer_delta) * learning_rate
        # output bias update
        hidden_to_output_weight_and_bias[1] += np.sum(network_last_layer_delta, axis=0, keepdims=True) * learning_rate

        # hidden to input weight update
        input_to_hidden_weight_and_bias[0] += np.dot(input_data.T, network_hidden_layer_delta) * learning_rate
        # hidden bias update 
        input_to_hidden_weight_and_bias[1] += np.sum(network_hidden_layer_delta, axis=0, keepdims=True) * learning_rate


    def train(input_data, expected, learning_rate):
        # Forward pass
        network_prediction, hidden_output_nodes = forward_in_network(input_data)
        # Compute the error 
        network_error = network_prediction - expected
        backward_in_network(input_data, network_prediction, hidden_output_nodes, network_error, learning_rate)

        return network_prediction

    return train

model = mlp_network(10, 20, 5)
input_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
expected_data = np.array([[6, 7, 8, 9, 10]])
print(model(input_data, expected_data, 0.001))
