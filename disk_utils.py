from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np


def save_model(model, file_name):
    model_json = model.to_json()
    with open(file_name + '.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_name + '.h5')
    return


def load_model(file_name):
    json_file = open(file_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(file_name + '.h5')
    return loaded_model


def load_image_for_prediction(file_name, input_shape):
    value_to_predict = image.load_img(file_name, color_mode='grayscale', target_size=(input_shape[1], input_shape[0]))
    value_to_predict = image.img_to_array(value_to_predict)
    value_to_predict = np.reshape(value_to_predict, input_shape)
    return value_to_predict
