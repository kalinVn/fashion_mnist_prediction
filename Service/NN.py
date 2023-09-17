import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from factory import Model as ModelFactory
import cv2
import os
import numpy as np

class NN:

    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.model_factory = ModelFactory()

    def fit(self):
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = fashion_mnist.load_data()
        train_data_scaled = self.train_data / 255.0
        test_data_scaled=  self.test_data / 255.0

        self.model = self.model_factory.get_keras_model()



        if (os.path.exists('store/models/model_SaveModel_format_test')):
            self.model = tf.keras.models.load_model("store/models/model_SaveModel_format_test");
        else:
            self.model.fit(train_data_scaled, self.train_labels,
                           epochs=10, validation_data=(test_data_scaled, self.test_labels))
            self.model.save('store/models/model_SaveModel_format_test')

    def predict(self, image_path):
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                       "Ankle boot"]
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (28, 28))
        prediction = self.model.predict(img_resized.reshape(1, 28, 28))
        # print(self.test_data[1])
        input_predict_label = np.argmax(prediction)

        print("The article is: ", class_names[input_predict_label])


