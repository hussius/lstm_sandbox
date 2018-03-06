# Working through attention for seq2seq models starting from
# https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/

import numpy as np

from data import generate_sequence, one_hot_encode, one_hot_decode, get_pair
from model import seq2seq, lstm, attention


# generate random sequence
sequence = generate_sequence(5, 50)
print(sequence)

oh = one_hot_encode(sequence, 50)
print(oh)

d = one_hot_decode(oh)
print(d)

X, y = get_pair(5, 2, 50)
print(X.shape, y.shape)
print('X=%s, y=%s' % (one_hot_decode(X[0]), one_hot_decode(y[0])))

# Baseline without attention
# configure problem
n_features = 50
n_timesteps_in = 5
n_timesteps_out = 2

# Create different models & compare
simple_lstm = lstm(lstm_cells=150, n_timesteps_in=n_timesteps_in, n_features=n_features)
seq2seq_model = seq2seq(lstm_cells=150, n_timesteps_in=n_timesteps_in, n_features=n_features)
attention_model = attention(lstm_cells=150, n_timesteps_in=n_timesteps_in, n_features=n_features)


for model in simple_lstm, seq2seq_model, attention_model:

    # train
    for epoch in range(5000):
        # generate new random sequence
        X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
        # fit model for one epoch on this sequence
        model.fit(X, y, epochs=1, verbose=0)

    # evaluate
    total, correct = 100, 0
    for _ in range(total):
        X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
        yhat = model.predict(X, verbose=0)
        if np.array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
            correct += 1
    print('Accuracy for {}: {}'.format(model.name, float(correct)/float(total)*100.0))

    # spot check examples
    # for _ in range(10):
    #    X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    #    yhat = model.predict(X, verbose=0)
    #    print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))

