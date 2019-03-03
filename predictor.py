import numpy as np


def predict(model, input_value):
    value_to_predict = np.reshape(input_value, (1, 28, 28, 1))
    return model.predict_classes(value_to_predict)
