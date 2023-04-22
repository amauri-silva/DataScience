
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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from imutils import paths
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb


def dataAugmentatio():
    """ Method responsable for difine the dataAugmentation properties.
    """

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest")
    return aug

def baseModelArc():
    """ Method responsible for create the core architecture of the network and its parameters.
    """

    #Transfer learning approach is used (imagenet wights)
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
                            input_tensor=Input(shape=(224, 224, 3)))

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(128, activation="relu")(headModel) # Up to 128 neuron on the hidden layer
    headModel = Dropout(0.2)(headModel) # All neurons that achive less than 20% of wight, doesn't pass for the next layer
    headModel = Dense(3, activation="softmax")(headModel) #Outpout with 3 layers

    # Creating the model ( Loop over all layers on the base model where they not will be update within the first traning step.)
    model = Model(inputs=baseModel.input, outputs=headModel)

    return model, baseModel

def trainingModel(aug, model, trainX, trainY, testX, testY):
    """ Method where the model is training and where the hyperparameters is used.
        
    Keyword arguments:
    aug     -- Object of pre augmentation set up
    model   -- Core of model
    trainX  -- Traing objects(compiled images)
    trainY  -- Traing objects to predic (compiled images)
    testX   -- Test objects(compiled images)
    testY   -- Test objects to predig (compiled images)
    """

    H = model.fit(
        aug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS)
    return H

def plotMetrics(H):
    """ Method that compile the training metrics over the traing processes

    Keyword arguments:
    H   -- Model object
    """

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

def confusionMatrix(testY, predIdxs):
    """ Method responsible for create the confusion Matrix image.
    """

    test_Y = np.argmax(testY, axis=1)
    cf_matrix = confusion_matrix(test_Y, predIdxs)
    heat_map = sb.heatmap(cf_matrix, annot=True, cmap="Blues", fmt="d")
    heat_map.set_title('Confusion Matrix\n');
    heat_map.set_xlabel('Category predictions\n\n')
    heat_map.xaxis.set_ticklabels(['BACTERIA', 'NORMAL', 'VIRAL'])
    heat_map.yaxis.set_ticklabels(['BACTERIA', 'NORMAL', 'VIRAL'])
    plt.savefig("matrix_de_confusao.png")

def training(imagePaths, imageSize, data, labels):
    """ Main method that execute the training model.

    Keywords arguments:
    imagePaths  -- Absolute path where the images is located
    imageSize   -- Defaul size of each image to process.
    data        -- An empyt list
    labels      -- Class of each Pneumonia that is classified
    """

    print("[INFO] Loading the imagens to process than ...")

    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
        #print("####: {}".format(label))

        # load the input image (224x224) and preprocess it
        image = load_img(imagePath,
                         target_size=(imageSize, imageSize))  # Loads an image into PIL format. (PIL it is a Pythin library)
        image = img_to_array(image)
        image = preprocess_input(image)  # Pre-processing steps include resizing to 224×224 pixels, conversion to array format,
        # and scaling the pixel intensities in the input image to the range [-1, 1]
        # (via the preprocess_input convenience function)

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # Convert the data and labels to NumPy arrays
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    lb = LabelEncoder()
    labels = lb.fit_transform(labels)
    
    # Creates a matrix of binary labels
    print("### Categories/classes in matrix format:  {}".format(labels))
    labels = to_categorical(labels)

    # Separete the data to train and test, the percentage seted was 80%(train) - 20%(test)
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20, stratify=labels, random_state=42)
    # Data Augmentation
    aug = dataAugmentatio()

    # Creats the base architecture of the model
    model, baseModel = baseModelArc()

    for layer in baseModel.layers:
        layer.trainable = False

    # Model compilation
    print("[INFO] Compiling the model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    plot_model(model, show_shapes=True, show_layer_names=True, rankdir="LR", expand_nested=True, dpi=True)

    # Model training (train the head of the network)
    print("[INFO] Using data augmetation on model training ...")
    H = trainingModel(aug, model, trainX, trainY, testX, testY)

    # Executing prediction tests
    print("[INFO] Checking the network (running the prediction tests) ...")
    predIdxs = model.predict(testX, batch_size=BS)

    # For each image on test set, it is finding the index label that containg the high prediction probability.
    predIdxs = np.argmax(predIdxs, axis=1)

    print("[INFO] Build the confusion matrix")
    confusionMatrix(testY, predIdxs)

    print("[INFO] Showing the formated dada of classification ...")
    print(classification_report(testY.argmax(axis=1), predIdxs,
                                target_names=lb.classes_))
    # Save the model on H5 format
    print("[INFO] Saving the X-Ray model detector ...")
    model.save("chestxray_3C.model", save_format="h5")

    # Ploting the training data
    plotMetrics(H)

if __name__ == '__main__':

    # Object and constants
    PATH_TRAIN = "../dataset/chest_xray/treinamento"
    imagePaths = list(paths.list_images(PATH_TRAIN))
    data = []
    labels = []

    # Hyperparameters
    imageSize = 224
    INIT_LR = 1e-4  # Initial value of coefficient to calculate the Stochastic Gradient Descendent 1e-4 = 0.0001
    EPOCHS = 20  # Total neuralnetwork training per steps
    BS = 40  # Value bigger than one and divided by the size of dataset.

    training(imagePaths, imageSize, data, labels)

