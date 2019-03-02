from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def create_alexnet_model():
    model = Sequential()

    model.add(Conv2D(filters=48, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=192, kernel_size=(3, 3), activation='relu'))

    model.add(Conv2D(filters=192, kernel_size=(3, 3), activation='relu'))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(2048, activation='relu'))

    model.add(Dense(2048, activation='relu'))

    model.add(Dense(1000, activation='relu'))
    return model
