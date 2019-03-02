from keras import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense


# http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf


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
