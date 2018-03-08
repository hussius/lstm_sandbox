from keras.models import Sequential
from keras.layers import TimeDistributed, RepeatVector, \
                         LSTM, Dense

from attention_decoder import AttentionDecoder


def lstm(lstm_cells, n_timesteps_in, n_features):
    model = Sequential(name='SimpleLSTM')
    model.add(LSTM(lstm_cells, input_shape=(n_timesteps_in, n_features),
              return_sequences=True))
    model.add(TimeDistributed(Dense(n_features, activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print(model.summary())
    return(model)


def seq2seq(lstm_cells, n_timesteps_in, n_features):
    model = Sequential(name="Seq2Seq")
    model.add(LSTM(lstm_cells, input_shape=(n_timesteps_in, n_features)))
    # The final state is repeated five times to match the number of time steps
    # in the output
    model.add(RepeatVector(n_timesteps_in))
    model.add(LSTM(lstm_cells, return_sequences=True))
    model.add(TimeDistributed(Dense(n_features, activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print(model.summary())
    return(model)


def attention(lstm_cells, n_timesteps_in, n_features):
    model = Sequential(name="AttentionLSTM")
    model.add(LSTM(lstm_cells, input_shape=(n_timesteps_in, n_features),
                   return_sequences=True))
    model.add(AttentionDecoder(lstm_cells, n_features))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    print(model.summary())
    return(model)
