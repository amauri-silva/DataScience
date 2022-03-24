
#:::BLOCO-1 ======================================================================================================================
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
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sb
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import os
from sklearn.metrics import confusion_matrix





#:::BLOCO-2 ======================================================================================================================
def dataAugmentatio():
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    return aug

#:::BLOCO-3 ======================================================================================================================
def baseModelArc():
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)))

    # Define/Carrega a arquitetura(parametros) base da rede (OBS: O mobileNet2 entrega uma melhor performance para processamento em dispositivos Móveis)
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel)
    headModel = Dropout(0.2)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)

    # Criando o Modelo ( Loop sobre todos os  layers no modelo base onde eles não iram ser atualizados durante o primeiro treinamento )
    model = Model(inputs=baseModel.input, outputs=headModel)

    return model, baseModel

#:::BLOCO-4 ======================================================================================================================
def trainingModel(aug, model, trainX, trainY, testX, testY):
    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)
    return H

#:::BLOCO-5 ======================================================================================================================
def plotMetrics(H):
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
    plt.savefig("acuracia_final.png")

#:::BLOCO-6 ======================================================================================================================
def confusionMatrix(testY, predIdxs):

    test_Y = np.argmax(testY, axis=1)
    cf_matrix = confusion_matrix(test_Y, predIdxs)
    heat_map = sb.heatmap(cf_matrix, annot=True, cmap="Blues", fmt="d")
    heat_map.set_title('Matriz de confusão\n');
    heat_map.set_xlabel('Predição das categorias\n\n')
    heat_map.xaxis.set_ticklabels(['BACTERIA', 'NORMAL', 'VIRAL'])
    heat_map.yaxis.set_ticklabels(['BACTERIA', 'NORMAL', 'VIRAL'])
    plt.savefig("matrix_de_confusao.png")


#:::BLOCO-7 ======================================================================================================================
def training(imagePaths, imageSize, data, labels):
    print("[INFO] Carregando as imagens para processamento ...")

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
    # cria uma matriz das labels binarias
    print("### Categorias/classes em forma de matriz:  {}".format(labels))
    labels = to_categorical(labels)

    # Separa os dados para treino e test, nesse caso a porcentagem é 80(treino)/20(teste)
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20, stratify=labels, random_state=42)
    # Data Augmentation
    aug = dataAugmentatio()

    # Cria a arquitetura base do modelo
    model, baseModel = baseModelArc()

    for layer in baseModel.layers:
        layer.trainable = False

    # Compilação do modelo
    print("[INFO] Compilando o modelo...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    plot_model(model, show_shapes=True, show_layer_names=True, rankdir="LR", expand_nested=True, dpi=True)

    # Treinamento do Modelo (train the head of the network)
    print("[INFO] Utilizando data augmetation no treinamento do modelo ...")
    H = trainingModel(aug, model, trainX, trainY, testX, testY)

    # Executa Testes de Predição
    print("[INFO] Avaliando a rede (executando os testes de predição) ...")
    predIdxs = model.predict(testX, batch_size=BS)

    # Para cada imagem no set de teste procuramos o index da label no qual contem a maio probabilidade de predição
    predIdxs = np.argmax(predIdxs, axis=1)

    print("[INFO] Gerando a Matrix de confusão")
    confusionMatrix(testY, predIdxs)

    print("[INFO] Exibindo os dados da classificação formatados ...")
    print(classification_report(testY.argmax(axis=1), predIdxs,
                                target_names=lb.classes_))
    # Grava o modelo no formato .H5
    print("[INFO] Salvando o modelo de detecção de Raio-X ...")
    model.save("chestxray_3C.model", save_format="h5")

    # Plotagem dos dados de treinamento
    plotMetrics(H)

#:::BLOCO-8 ======================================================================================================================
if __name__ == '__main__':

    # Objetos e constantes
    PATH_TRAIN = "../dataset/chest_xray/treinamento"
    imagePaths = list(paths.list_images(PATH_TRAIN))
    data = []
    labels = []

    # Hiperâmetros
    imageSize = 224
    INIT_LR = 1e-4  # Coeficiente inicial para calcular do Stochastic Gradient Descendent 1e-4 = 0.0001
    EPOCHS = 20  # Quantidade de treinamento da rede neural
    BS = 40  # Valor maior que 1 e divisivel pelo tamanho total do dataset

    training(imagePaths, imageSize, data, labels)

