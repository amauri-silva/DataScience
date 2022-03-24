import operator
import glob as glb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
from PIL import Image, ImageFont, ImageDraw

def classification(image_path, dir_name):
    IMAGE_SIZE = 224
    image = load_img(image_path,
                     target_size=(
                     IMAGE_SIZE, IMAGE_SIZE))  # Loads an image into PIL format. (PIL it is a Python library)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    index = np.argmax(predictions)
    #max_value = predictions[0][index]
    #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$: {}".format(index))
    #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$: {}".format(max_value))
    print("[INFO] Classificando as imagens do diretório: {}".format(dir_name))
    if(index == 0):
        print("::RaioX Analisado: {}".format(dir_name))
        print("Classificado como: {}".format(LABELS[index]))
        print(":::::::::Acurácia: %.2f" % predictions[0][index])
        print("--------------------------------------------------------------------")
    elif(index == 1):
        print("::RaioX Analisado: {}".format(dir_name))
        print("Classificado como: {}".format(LABELS[index]))
        print(":::::::::Acurácia: %.2f" % predictions[0][index])
        print("--------------------------------------------------------------------")
    else:
        print("::RaioX Analisado: {}".format(dir_name))
        print("Classificado como: {}".format(LABELS[index]))
        print(":::::::::Acurácia: %.2f" % predictions[0][index])
        print("--------------------------------------------------------------------")


def get_images_validation(dir):
    path = PATH_VALIDATION + dir + FORMAT

    for image in glb.glob(path):
        classification(image, dir)

def get_dir_validation():
    for dir in DIRETORIOS:
        get_images_validation(dir)

if __name__ == '__main__':
    print("[INFO] Carregando o modelo para realizar a predição ...")
    model = load_model("../model/chestxray_3C.model")
    # Constantes e variáveis
    PATH_VALIDATION = "../dataset/chest_xray/validacao"
    NORMAL = "NORMAL"
    PNEUMONIA_BACTERIA = "PNEUMONIA_BACTERIA"
    PNEUMONIA_VIRAL = "PNEUMONIA_VIRAL"

    LABELS = ['BACTERIA', 'NORMAL', 'VIRAL']
    PATH_VALIDATION = "../dataset/chest_xray/validacao/"
    DIRETORIOS = ['BACTERIA/', 'NORMAL/', 'VIRAL/']
    FORMAT = "*.jpeg"

    # PATH_VALIDATION = "../dataset/chest_xray/validacao/BACTERIA/0-BACTERIA.jpeg"

    get_dir_validation()