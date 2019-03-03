from keras.datasets import mnist
import numpy as np

import plotter
import cnnfactory as factory


def load_dataset():
    # Fase di input
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    # Fase di normalizzazione
    max = np.max(x_train)
    x_train, x_test = x_train / max, x_test / max

    mean = np.std(x_train)
    x_train, x_test = x_train - mean, x_test - mean

    return (x_train, y_train), (x_test, y_test)


def create_trained_model():
    (x_train, y_train), (x_test, y_test) = load_dataset()

    # Otteniamo il modello
    cnn_model = factory.create_lenet_5_model(input_shape=(28, 28, 1))

    cnn_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # Fase di allenamento
    history = cnn_model.fit(x_train, y_train,
                            validation_split=0.33, epochs=4)

    # Plot training & validation accuracy values
    plotter.plot_history(history)

    # Fase di valutazione
    loss, metric = cnn_model.evaluate(x_test, y_test)
    print('loss: ', loss)
    print('accuracy: ', metric)
    return cnn_model

