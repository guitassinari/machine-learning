# coding: utf-8

from models.NeuralNetwork import NeuralNetwork

def test_weight_matrices():
    n_hidden_layers = 1
    inputs = 1
    outputs = 1
    n_layer_neuron = 3
    nn = NeuralNetwork(n_inputs=inputs,
                       n_outputs=outputs,
                       n_hidden_layers=n_hidden_layers,
                       layers_n_neurons=[n_layer_neuron])
    assert len(nn.weight_matrices) == n_hidden_layers + 1
    assert len(nn.weight_matrices[0]) == inputs
    assert len(nn.weight_matrices[1]) == n_layer_neuron
    assert len(nn.weight_matrices[0][0]) == n_layer_neuron
    assert len(nn.weight_matrices[1][0]) == outputs

def test_prediction():
    n_hidden_layers = 1
    inputs = 1
    outputs = 1
    n_layer_neuron = 3
    nn = NeuralNetwork(n_inputs=inputs,
                       n_outputs=outputs,
                       n_hidden_layers=n_hidden_layers,
                       layers_n_neurons=[n_layer_neuron],
                       debug=True)
    assert nn.predict([1]) == [0.9525741268224334]
