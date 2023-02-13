#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""
Module providing neural network implementation functions for
dog breed predictions, dog detection and human detection.
"""

__version__ = "1.0.0"
__author__ = "Sergio Badillo"


import numpy
import cv2
import imutils
from imutils import face_utils
import keras
from keras.applications import xception, resnet
from dlib import cnn_face_detection_model_v1, get_frontal_face_detector
import dog_names


def path_to_tensor(img_path):
    """Loads RGB image as PIL.Image.Image type and
    convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)

    Args:
        img_path (str): path to image file

    Returns:
        array : tensor with shape (1, 224, 224, 3)
    """
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    img_arr = keras.utils.img_to_array(img)
    return numpy.expand_dims(img_arr, axis=0)


def find_human(img, algorithm="dlib-cnn"):
    """Uses CNN-MMOD algorithm to detect human faces in image.
    Returns face position if a human face is found, otherwise returns False
    Extracts only first face for simplicity.

    Args:
        img (numpy.array): image numpy array.
            If path is provided, will try to generate an array
        algorithm (str, optional): _description_. Defaults to "dlib-cnn".

    Returns:
        _type_: _description_
    """

    # transform to ndarray if needed
    if not isinstance(img, numpy.ndarray):
        try:
            print("trying to convert")
            img = cv2.imread(img)
        except TypeError():
            print(
                "Oops!  That was not a valid file. Please check that you are providing an image in array form."
            )

    if algorithm == "dlib-cnn":
        # more accurate, but slow, this detector is used by default

        face_cnn = cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
        rects = face_cnn(img, 1)

        if len(rects) > 0:

            face = face_utils.rect_to_bb(rects[0].rect)
            confidence = rects[0].confidence
            return face, confidence
        else:
            return None

    if algorithm == "hog":
        # algorithm HoG, this one is faster but has higher error.

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_hog = get_frontal_face_detector()
        rects = face_hog(gray, 1)

        if rects:
            face = face_utils.rect_to_bb(rects[0])
            confidence = 1.0
            return face, confidence

        else:
            return None


def find_dog(img_path):
    """Returns True if a dog is detected in the image stored at img_path."""

    img = resnet.preprocess_input(path_to_tensor(img_path))

    # define ResNet50 model
    ResNet50_model = resnet.ResNet50(weights="imagenet")

    # predict
    prediction = numpy.argmax(ResNet50_model.predict(img, verbose=0))

    return (prediction <= 268) & (prediction >= 151)


def find_dogbreed(img_path):
    """Predict dog breed from image in img path.

    Extracts features using built-in keras Xception module
    Then do the prediction with our model and return dog breed using
    dog_names labels.

    Input shape corresponds to pre-processed bottleneck features.

    Args:
        img_path (str): path to image file

    Returns:
        str: label describing dog breed
    """

    # load labels
    labels = dog_names.dog_names

    # Extract bottleneck features using Keras' Xception module.
    tensor = path_to_tensor(img_path)
    base_model = xception.Xception(weights="imagenet", include_top=False)
    bottleneck_features = base_model.predict(xception.preprocess_input(tensor))

    # Create top model layers to predict on bottleneck features.
    # Input shape corresponds to pre-processed feature shape.
    model = keras.Sequential(name="Xception-final")
    model.add(keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    model.add(keras.layers.Dense(133, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )

    # load top layers trained weights
    model.load_weights("models/weights.best.Xception-final.hdf5")

    # predict
    predicted_vector = model.predict(bottleneck_features, verbose=0)

    return labels[numpy.argmax(predicted_vector)]


def draw_bow(face, img_arr, text, color):
    """draw boxes using faces coordinates and annotates text in image"""

    face_x, face_y, face_w, face_h = face

    cv2.rectangle(
        img_arr,
        (face_x, face_y),
        (face_x + face_w, face_y + face_h),
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img_arr,
        text,
        (face_x, face_y - 3),
        fontFace=cv2.FONT_HERSHEY_DUPLEX,
        fontScale=0.7,
        color=color,
        lineType=cv2.LINE_AA,
    )
    return img_arr


def predict_final(img_path):
    """Accepts a file path to an image and first determines whether
    the image contains a human, dog, or neither.

    - if a dog is detected, return the predicted breed.
    - if a human is detected, return the resembling dog breed.
    - if neither is detected, provide output that indicates an error.
    """

    print("Analyzing image: {img_path}")

    IMG_MAX_WIDTH = 600
    HUMAN_COLOR = (0, 0, 255)

    # initialize results dict
    results = {
        "is_dog": False,
        "is_human": False,
        "breed": None,  # used for both dogs and humans lol
    }

    # import to np array and resize if size is exceeded.
    img_arr = cv2.imread(img_path)
    if img_arr.shape[0] > IMG_MAX_WIDTH:
        img_arr = imutils.resize(img_arr, width=IMG_MAX_WIDTH)

    # main prediction logic

    # check if dog
    results["is_dog"] = find_dog(img_path)
    if results["is_dog"]:
        print("Doggo 🐕 found !")
        results["breed"] = find_dogbreed(img_path)
        print(f"Looks like a {results['breed']}")

        return (results, img_arr)

    # check if human face
    human_pos = find_human(img_arr, algorithm="hog")
    results["is_human"] = human_pos is not None
    if results["is_human"]:
        print("Human 👩👨detected !")
        img_arr = draw_bow(human_pos[0], img_arr, text="human !?", color=HUMAN_COLOR)
        # convert BGR image to RGB
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        results["breed"] = find_dogbreed(img_path)
        print(f"Anyway, he/she looks like a {results['breed']}.")

        return (results, img_arr)

    # elif results["is_human"] and not results['is_dog']:
    #     print("Human 👩👨detected !")
    #     img_arr = draw_bow(human_pos[0], img_arr, text="human !?", color=HUMAN_COLOR)
    #     results["breed"] = find_dogbreed(img_path)
    #     print(f"Anyway, he/she looks like a {results['breed']}.")

    # elif is_dog and results["is_human"]:
    #     print("❌ Not sure if I see a a dog or a human!, predicting breed anyway !")

    # fallback : not dog, neither dog
    print("❌ I wasn't able to find any faces in the picture :'(")

    return (results, img_arr)
