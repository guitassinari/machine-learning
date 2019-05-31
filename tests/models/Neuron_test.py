from models.Neuron import Neuron

def test_prediction():
    neuron = Neuron([1, 2, 3])
    assert neuron.predict([3, 2, 1]) == 0.9999546021312976