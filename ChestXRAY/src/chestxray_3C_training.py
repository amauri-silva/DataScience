import cv2
from imutils import paths
#import numpy as np
#import cv2 as cv
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import os
import time
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import argparse
import os
import time

data = []
labels = []
# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4 #Coeficiente inicial para calcular do Stochastic Gradient Descendent 1e-4 = 0.0001
EPOCHS = 2 #Quantidade de treinamento da rede neural para
BS = 40 #Valor maior que 1 e divisivel pelo tamanho total do dataset


#Metodos
def resizeImages(imagePaths, imageSize, data, labels):
    print("[INFO] loading images...")

    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        #print("####: {}".format(label))

        # load the input image (224x224) and preprocess it
        # Ao definir o tamanho fixo das imagens distorce porem ainda sim da para indentificar os objetos na imagens
        # essa abordagem devem ser levada em consideracao quanto ao tipo de tratamento e resultado esperado no processamento
        image = load_img(imagePath,
                         target_size=(imageSize, imageSize))  # Loads an image into PIL format. (PIL it is a Pythin library)
        image = img_to_array(image)
        image = preprocess_input(image)  # Pre-processing steps include resizing to 224×224 pixels, conversion to array format,
        # and scaling the pixel intensities in the input image to the range [-1, 1]
        # (via the preprocess_input convenience function)

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    lb = LabelEncoder()
    labels = lb.fit_transform(labels)
    #-------------------------------------------------------------------------------------
    # perform one-hot encoding on the labels
    #lb = LabelBinarizer()
    # transformação binaria das classes em labels binarias (0,1)
    #labels = lb.fit_transform(labels)
    # cria uma matriz das labels binarias
    labels = to_categorical(labels)
    print("### Categories:  {}".format(labels))

    #[5059, 5062]
    # https://datascience.stackexchange.com/questions/20199/train-test-split-error-found-input-variables-with-inconsistent-numbers-of-sam

    # 2 - Separa os dados para treino e test, nesse caso a porcentagem é 80(treino)/20(teste)
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20, stratify=labels, random_state=42)

    # ======================================================================================================================
    # Data Augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    # ======================================================================================================================
    # 4 - Cria a arquitetura base do modelo (Rede mobileNet2 entrega uma melhor performance para processamento em dispositivos Móveis em temos de processamento)
    # load the MobileNetV2 network, ensuring the head FC layer sets are
    # left off
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)))

    # ======================================================================================================================
    # 5 - Define/Carrega a arquitetura(parametros) base da rede (OBS: O mobileNet2 entrega uma melhor performance para processamento em dispositivos Móveis)
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)

    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)

    # ======================================================================================================================
    # 6 - Criando o Modelo
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    # ======================================================================================================================
    # 7 - Compilação do modelo(Ainda não esta claro po que fazer essa compilação)
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    # plot_model(model, show_shapes=True)
    plot_model(model, show_shapes=True, show_layer_names=True, rankdir="LR", expand_nested=True, dpi=True)

    # ======================================================================================================================
    # 8 - Treinamento do Modelo (train the head of the network)
    print("[INFO] training head...")
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)

    # ======================================================================================================================
    # 9 - Executa Testes de Predição
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BS)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    
    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), predIdxs,
                                target_names=lb.classes_))

    # ======================================================================================================================
    # 10 - Grava o modelo no formato .H5
    print("[INFO] saving mask detector model...")
    model.save("chestxray_3C.model", save_format="h5")

    # ======================================================================================================================
    # 11 - plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

if __name__ == '__main__':
    # Objetos e constantes
    PATH_TRAIN = "../dataset/chest_xray/treinamento"
    PATH_VALIDATION = "../dataset/chest_xray/validacao"
    #::: Categorias das imagens
    NORMAL = "NORMAL"
    PNEUMONIA_BACTERIA = "PNEUMONIA_BACTERIA"
    PNEUMONIA_VIRAL = "PNEUMONIA_VIRAL"

    imagePaths = list(paths.list_images(PATH_TRAIN))
    imageSize = 224
    LABELS = ['BACTERIA','NORMAL','VIRAL']
    labels = []
    # Hiperâmetros

    resizeImages(imagePaths, imageSize, data, labels)

