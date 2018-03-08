from random import randint

import numpy as np

# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(0, n_unique-1) for _ in range(length)]

# one hot encode sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return np.array(encoding)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]

# prepare data for the LSTM
def get_pair(n_in, n_out, n_unique):
    # generate random sequence
    sequence_in = generate_sequence(n_in, n_unique)
    sequence_out = sequence_in[1:] + [0] # sequence_in[:n_out] + [0 for _ in range(n_in-n_out)]
    # one hot encode
    X = one_hot_encode(sequence_in, n_unique)
    y = one_hot_encode(sequence_out, n_unique)
    # reshape as 3D
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X,y



