
#Organização dos imports
import glob

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

model = load_model("chestxray.model")
#Constantes e variáveis
#test_images = cv2.imread("../dataset/chest_xray/validacao/BACTERIA/0-BACTERIA.jpeg")
PATH_VALIDATION = "../dataset/chest_xray/validacao/BACTERIA/0-BACTERIA.jpeg"
IMAGE_SIZE = 224
image = load_img(PATH_VALIDATION,
                 target_size=(IMAGE_SIZE, IMAGE_SIZE))  # Loads an image into PIL format. (PIL it is a Python library)
image = img_to_array(image)
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)
print("#####:  {}".format(image.shape))

print("[INFO] computing ChestXRay detections...")
#probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


predictions = model.predict(image)
#predictions = probability_model.predict(image)

print("#####:  {}".format(predictions[0]))