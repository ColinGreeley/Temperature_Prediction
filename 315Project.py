import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Dropout
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data(t, start_index, end_index, history_size, target_size, step):
    path = 'climate-change-earth-surface-temperature-data/'
    global_temperatures_path = path + 'GlobalTemperatures.csv'
    data = pd.read_csv(global_temperatures_path)[1200:].to_numpy()[:,1:]
    #print(data.shape)
    for i in range(0,data.shape[1]):
        mean = data[:,i].mean(axis=0)
        data[:,i] -= mean
        std = data[:,i].std(axis=0)
        data[:,i] /= std
    x_train = []
    y_train = []
    target = data[:,t]
    print(target.shape)
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(data) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        x_train.append(data[indices])
        y_train.append(target[i:i+target_size])

    return np.array(x_train).astype('float32'), np.array(y_train).astype('float32')

def accuracy(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.equal(tf.keras.backend.round(y_true), tf.keras.backend.round(y_pred)))

def make_model(data, n_future):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(None, data.shape[-1])))
    model.add(Dropout(0.1))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(n_future, activation='linear'))
    model.compile(optimizer='adam', loss='mae', metrics=[accuracy])
    model.summary()
    return model

def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(t, history, true_future, prediction, xlabel, ylabel):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    plt.plot(num_in, np.array(history[:, t]), label='History')
    plt.plot(np.arange(num_out), np.array(true_future), 'g-',
            label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out), np.array(prediction), 'r--',
                label='Predicted Future')
    plt.xlabel('Time {}'.format(xlabel))
    plt.ylabel('{} Temperateure (C)'.format(ylabel))
    plt.plot([num_out-1], np.array(true_future[-1]), 'bo')
    plt.plot([num_out-1], np.array(prediction[-1]), 'ro')
    print('Difference:', history[-1, t]-prediction[-1])
    plt.title('{} Temperature Prediction for 3 years (beginning 1/1/2000)'.format(ylabel))
    plt.legend(loc='upper left')
    plt.savefig('{}Temp.png'.format(ylabel))
    plt.show()

def test1(t, lookback, target, batch_size, step):
    name = {0:'Average', 2:'Maximum', 4:'Minumum'}
    x_train, y_train = get_data(t, 0, 1800, lookback, target, step)
    x_test, y_test = get_data(t, 1800, None, lookback, target, step)
    model = make_model(x_train, target)
    history = model.fit(x_train, y_train, epochs=20).history
    model.evaluate(x_test, y_test)
    multi_step_plot(t, x_test[0], y_test[0], model.predict(x_test)[0], '(Months)', name[t])
    loss = history['loss']
    accuracy = history['accuracy']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, '-', label='Training loss')
    plt.title('Training loss for {}'.format(name[t]))
    plt.legend()
    plt.savefig('{}_Loss.png'.format(name[t]))
    plt.show()    
    plt.figure()
    plt.plot(epochs, accuracy, '-', label='Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training accuracy for {}'.format(name[t]))
    plt.legend()
    plt.savefig('{}_Acc.png'.format(name[t]))
    plt.show()

if __name__ == "__main__":
    
    test1(0, lookback=60, target=36, batch_size=32, step=1)
    test1(2, lookback=60, target=36, batch_size=32, step=1)
    test1(4, lookback=60, target=36, batch_size=32, step=1)