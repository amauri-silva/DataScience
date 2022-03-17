
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
#::: Categorias das imagens
NORMAL = "NORMAL"
PNEUMONIA_BACTERIA = "PNEUMONIA_BACTERIA"
PNEUMONIA_VIRAL = "PNEUMONIA_VIRAL"

PATH_VALIDATION = "../dataset/chest_xray/validacao/BACTERIA/0-BACTERIA.jpeg"
#[INFO] computing ChestXRay detections...
#####:  [0.9960813  0.00391868]

PATH_VALIDATION = "../dataset/chest_xray/validacao/NORMAL/0-NORMAL.jpeg"
#####:  (1, 224, 224, 3)
#[INFO] computing ChestXRay detections...
#####:  [2.6151352e-04 9.9973851e-01]

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
