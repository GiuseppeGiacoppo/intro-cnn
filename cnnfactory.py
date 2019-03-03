from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout


def create_lenet_5_model(input_shape=(32, 32, 1), output=10):
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=output, activation='softmax'))

    return model


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


def create_modified_lenet_5_model(input_shape=(32, 32, 1), output=10):
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D())

    model.add(Dropout(0.25))
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=output, activation='softmax'))

    return model

