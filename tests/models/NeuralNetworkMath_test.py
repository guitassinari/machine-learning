# coding: utf-8

from models.NeuralNetworkMath import NeuralNetworkMath


def test_loss_function():
    expected = [0, 1, 0]
    predicted = [0.5, 0.2, 0.3]
    assert NeuralNetworkMath.loss(predicted, expected) == 1.6094379124341003


def test_regularization():
    weights = [[[1, 2, 3],
                [1, 2, 3],
                [1, 2, 3]]]
    _lambda = 1
    n_samples = 2
    assert NeuralNetworkMath.loss_regularization(weights,
                                                 _lambda=_lambda,
                                                 n_examples=n_samples) == 10.5


def test_delta():
    weights = [[1, 1, 1],
               [1, 1, 1]]
    activations = [.5, .5, .5]
    next_deltas = [1, 1]
    deltas = NeuralNetworkMath.delta(activations, weights, next_deltas)
    assert all([a == b for a, b in zip(deltas, [.5, .5, .5])])


def test_output_delta():
    real_output = [0, 1, 0]
    expected_output = [1, 0, 0]
    deltas = NeuralNetworkMath.output_delta(real_output, expected_output)
    assert all([a == b for a, b in zip(deltas, [-1, 1, 0])])
