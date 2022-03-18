
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

print("[INFO] Carregando o modelo para realizar a predição ...")
model = load_model("chestxray_3C.model")
#Constantes e variáveis
PATH_VALIDATION = "../dataset/chest_xray/validacao"
NORMAL = "NORMAL"
PNEUMONIA_BACTERIA = "PNEUMONIA_BACTERIA"
PNEUMONIA_VIRAL = "PNEUMONIA_VIRAL"

LABELS = ['BACTERIA','NORMAL','VIRAL']

PATH_VALIDATION = "../dataset/chest_xray/validacao/BACTERIA/0-BACTERIA.jpeg"

IMAGE_SIZE = 224
image = load_img(PATH_VALIDATION,
                 target_size=(IMAGE_SIZE, IMAGE_SIZE))  # Loads an image into PIL format. (PIL it is a Python library)
image = img_to_array(image)
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)
print("#####:  {}".format(image.shape))

print("[INFO] Execuando a detecção na imagem (Raio-X)...")
predictions = model.predict(image)
#predictions = probability_model.predict(image)

print("#####:  {}".format(predictions[0]))
