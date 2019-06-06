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
