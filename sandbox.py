def one_neural_network(input_data=2, desired_output=10, lr=0.01):
    # Initialize weights and bias
    weight = 0.1
    bias = 0.1

    while True:
        node1 = (input_data*weight) + bias
        # loss calculation to match the desired output we squared so that we can't have a negative number.
        loss = (node1 - desired_output)**2
        
        print(f"Neuron: {node1} Weight: {weight} Bias: {bias} Loss: {loss}")

        if int(node1) == desired_output:
            print(node1)
            break

        # How much steep were need to toward the local minima
        derivative = 2*(node1 - desired_output)

        # We step based on the derivative
        gradient_weight = node1*derivative
        gradient_bias = bias*derivative

        # weight and bias update
        weight += weight - lr * gradient_weight
        bias += bias - lr * gradient_bias

one_neural_network()
