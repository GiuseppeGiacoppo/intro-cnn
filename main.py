import mnist
import plotter
import predictor
import disk_utils

#model = mnist.create_trained_model()
#disk_utils.save_model(model, 'model')

model = disk_utils.load_model('model')
value_to_predict = disk_utils.load_image_for_prediction('target.jpg', (28, 28, 1))
plotter.plot_matrix(value_to_predict)

value_predicted = predictor.predict(model, value_to_predict)
print("Il valore numerico è:", value_predicted)
