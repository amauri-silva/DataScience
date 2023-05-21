
from keras.models import load_model
from keras.utils import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import load_img
import os
import glob as glb
import numpy as np
import operator
from PIL import Image, ImageFont, ImageDraw


def classification(image_path, dir_name):
    """ This method is responsible to show the style of metrics while the model prediction is running.

    Keyword arguments:
    image_path -- Full path of wich imagens + name's image.
    dir_name   -- Dir name where wich classe of image it is.
    """

    IMAGE_SIZE = 224
    image = load_img(image_path,
                     target_size=(
                     IMAGE_SIZE, IMAGE_SIZE))  # Loads an image into PIL format. (PIL it is a Python library)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)
    index = np.argmax(predictions)
    
    print("[INFO] Classifying the directory's imagens: {}".format(dir_name))
    if(index == 0):
        print("::Analized X-Ray: {}".format(dir_name))
        print("Classifying as: {}".format(LABELS[index]))
        print(":::::::::Accuracy: %.2f" % predictions[0][index])
        print("--------------------------------------------------------------------")
    elif(index == 1):
        print("::Analized X-Ray: {}".format(dir_name))
        print("Classifying as: {}".format(LABELS[index]))
        print(":::::::::Accuracy: %.2f" % predictions[0][index])
        print("--------------------------------------------------------------------")
    else:
        print("::Analized X-Ray: {}".format(dir_name))
        print("Classifying as: {}".format(LABELS[index]))
        print(":::::::::Accuracy: %.2f" % predictions[0][index])
        print("--------------------------------------------------------------------")


def get_images_validation(dir):
    """ Method that do loop over each class of pneumonia by dictory to validate its imagens.

    Keyword arguments:
    dir        -- Directory of each pneumonia class.
    """

    path = PATH_VALIDATION + dir + FORMAT

    for image in glb.glob(path):
        classification(image, dir)

def get_dir_validation():
    for dir in DIRETORIOS:
        get_images_validation(dir)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    print("[INFO] Loaging the model to start the prectidion ...")
    model = load_model("../model/chestxray_3C.model")
    
    # Constants and variables
    PATH_VALIDATION = "../dataset/chest_xray/validation"
    NORMAL = "NORMAL"
    PNEUMONIA_BACTERIA = "PNEUMONIA_BACTERIA"
    PNEUMONIA_VIRAL = "PNEUMONIA_VIRAL"

    LABELS = ['BACTERIA', 'NORMAL', 'VIRAL']
    PATH_VALIDATION = "../dataset/chest_xray/validation/"
    DIRETORIOS = ['BACTERIA/', 'NORMAL/', 'VIRAL/']
    FORMAT = "*.jpeg"

    get_dir_validation()