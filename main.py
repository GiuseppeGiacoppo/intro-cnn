from lenet5 import create_lenet_5_model
from keras.datasets import mnist

import matplotlib.pyplot as plt
import numpy as np

# Fase di input
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Fase di normalizzazione
max = np.max(x_train)
x_train, x_test = x_train / max, x_test / max

mean = np.std(x_train)
x_train, x_test = x_train - mean, x_test - mean

# Otteniamo il modello
model = create_lenet_5_model(input_shape=(28, 28, 1))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fase di allenamento
history = model.fit(x_train, y_train,
                    validation_split=0.33, epochs=5)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Fase di valutazione
loss, metric = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('accuracy: ', metric)
