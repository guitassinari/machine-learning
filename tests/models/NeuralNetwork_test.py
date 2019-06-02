# coding: utf-8

from models.NeuralNetwork import NeuralNetwork

def test_weight_matrices():
    nn = NeuralNetwork(n_inputs=1,
                       n_outputs=1,
                       n_hidden_layers=1,
                       layers_n_neurons=[3])
    assert len(nn.weight_matrices) == 2
