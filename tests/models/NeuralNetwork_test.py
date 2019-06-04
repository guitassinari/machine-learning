# coding: utf-8

from models.NeuralNetwork import NeuralNetwork


def test_weight_matrices():
    inputs = 1
    outputs = 1
    hidden_layer_neurons = 3
    nn = NeuralNetwork(layers_n_neurons=[inputs, hidden_layer_neurons, outputs])
    assert len(nn.weight_matrices) == 2  # input -> hidden, hidden -> output
    assert len(nn.weight_matrices[0]) == inputs
    assert len(nn.weight_matrices[1]) == hidden_layer_neurons
    assert len(nn.weight_matrices[0][0]) == hidden_layer_neurons
    assert len(nn.weight_matrices[1][0]) == outputs


def test_prediction():
    inputs = 1
    outputs = 1
    hidden_layer_neurons = 3
    nn = NeuralNetwork(layers_n_neurons=[inputs, hidden_layer_neurons, outputs],
                       debug=True)
    first_line = nn.predict([1])[0]
    assert first_line == [0.9990889488055994]


def test_hidden_activations():
    inputs = 1
    outputs = 1
    hidden_layer_neurons = 3
    nn = NeuralNetwork(layers_n_neurons=[inputs, hidden_layer_neurons, outputs],
                       debug=True)
    first_line = nn.hidden_activation([1])[0]
    assert all([a == b for a, b in zip(first_line, [2, 2, 2])])
