"""
Module for importing MNIST dataset.
"""

import pickle
import gzip

import numpy as np

def load_data():
    """unzip and unpickle archive."""
    file = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = (
        pickle.load(file, encoding='latin1'))
    file.close()

    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Reorganize the data."""
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return (list(training_data), list(validation_data), list(test_data))

def vectorized_result(j):
    """return vectorized form of j."""
    vec = np.zeros((10, 1))
    vec[j] = 1.0
    return vec
