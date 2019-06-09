# coding: utf-8

from models.NeuralNetwork import NeuralNetwork
from models.NeuralNetworkMath import NeuralNetworkMath
from data.Dataset import Dataset
from data.Example import Example

inputs = 1
outputs = 1
hidden_layer_neurons = 3
example = Example([1, 2], [1, 1])
dataset = Dataset([example])
parameters = {
    "layers_structure": [inputs, hidden_layer_neurons, outputs],
    "lambda": 0.1
}


def test_weight_matrices():
    nn = NeuralNetwork(parameters, dataset, debug=True)
    assert len(nn.weight_matrices) == 2  # input -> hidden, hidden -> output
    assert len(nn.weight_matrices[0]) == hidden_layer_neurons
    assert len(nn.weight_matrices[1]) == outputs
    assert len(nn.weight_matrices[0][0]) == inputs
    assert len(nn.weight_matrices[1][0]) == hidden_layer_neurons


def test_prediction():
    nn = NeuralNetwork(parameters, dataset, debug=True)
    first_line = nn.predict([1])[0]
    assert first_line == [0.9744787489988975]


def test_hidden_activations():
    nn = NeuralNetwork(parameters, dataset, debug=True)
    first_line = nn.hidden_activation([1])[0]
    expected_result = NeuralNetworkMath.sigmoid(2)
    expected_line = [expected_result, expected_result, expected_result]
    assert all([a == b for a, b in zip(first_line, expected_line)])


def test_backpropagate():
    nn = NeuralNetwork(parameters, dataset, debug=True)

    nn.back_propagate([0])
    assert nn.deltas[2][0] == [0.9605766589005886]
