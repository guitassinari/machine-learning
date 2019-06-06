# coding: utf-8

from models.NeuralNetwork import NeuralNetwork
from models.NeuralNetworkMath import NeuralNetworkMath
from data.Dataset import Dataset

inputs = 1
outputs = 1
hidden_layer_neurons = 3
parameters = {
    "layers_structure": [inputs, hidden_layer_neurons, outputs]
}

def test_weight_matrices():
    nn = NeuralNetwork(parameters, Dataset([]), debug=True)
    assert len(nn.weight_matrices) == 2  # input -> hidden, hidden -> output
    assert len(nn.weight_matrices[0]) == inputs
    assert len(nn.weight_matrices[1]) == hidden_layer_neurons
    assert len(nn.weight_matrices[0][0]) == hidden_layer_neurons
    assert len(nn.weight_matrices[1][0]) == outputs


def test_prediction():
    nn = NeuralNetwork(parameters, Dataset([]), debug=True)
    first_line = nn.predict([1])[0]
    assert first_line == [0.9744787489988975]


def test_hidden_activations():
    nn = NeuralNetwork(parameters, Dataset([]), debug=True)
    first_line = nn.hidden_activation([1])[0]
    expected_result = NeuralNetworkMath.sigmoid(2)
    expected_line = [expected_result, expected_result, expected_result]
    print(first_line, expected_line)
    assert all([a == b for a, b in zip(first_line, expected_line)])
