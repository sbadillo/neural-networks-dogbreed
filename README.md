TODO

- [ ] Web Application
	- [x]  There is a web application that utilizes data to inform how the web application works.
- [ ] Code
  - [ ] Code is formatted neatly with comments and uses DRY principles.
- [ ] README.md file that communicates : 
  - [ ] TOC (updated)
  - [x] the libraries used, 
  - [x] the files in the repository with a small description of each, 
  - [ ] and necessary acknowledgements.
  - [x] Instruction to run the application on a local.
  - [ ] 1. Project Definition, 
    - [ ] 1.1 Motivation for the project
  - [ ] 2. Analysis
    - [ ] 2.2 a summary of the results of the analysis, 
  - [ ] 3. Conclusion.
  - [ ] Aknowledgements.
- [ ] Git repository and submission

---

**Contents**

- [Introduction](#introduction)
  - [User interface](#user-interface)
  - [Prediction algorithms](#prediction-algorithms)
- [Quickstart](#quickstart)
  - [Installation](#installation)
  - [Running the app](#running-the-app)
- [Project Structure and Libraries](#project-structure-and-libraries)
- [Details](#details)
  - [Project Definition and Motivation](#project-definition-and-motivation)
  - [Analysis](#analysis)
    - [Machine Learning techniques](#machine-learning-techniques)
      - [**Dog vs Human differentiation**](#dog-vs-human-differentiation)
      - [**Dog's breed prediction**](#dogs-breed-prediction)
      - [**Imagenet**](#imagenet)
  - [Conclusions](#conclusions)
    - [Aknowledgements](#aknowledgements)

# Introduction

An application that predicts a dog's breed from a user-provided image by using artificial neural networks.

![Capture](capture2.gif)

##  User interface
The web app follows this simple user story: 

* User uploads an image to the main page.
* if a dog is detected in the image, we display the predicted breed.
* if a human is detected in the image, we return the resembling dog breed.
* if neither is detected in the image, provide output that indicates an error and some suggestions.

## Prediction algorithms
The project's backend demonstrates the concept of [**Transfer Learning**](https://en.wikipedia.org/wiki/Transfer_learning) by leveraging existing models and building on top of pre-trained **convolutional neural networks**. These models are used to identify dogs or humans, and later to predict a dog's breed.


# Quickstart
## Installation
Clone this repository, then cd into it

    $ git clone https://github.com/sbadillo/neural-networks-dogbreed.git
    $ cd neural-networks-dogbreed

It is recommended to set-up a python virtual environment using the `conda` package manager from the Anaconda distribution ([available here](https://www.anaconda.com/products/distribution)).

Run this command from a conda-enabled shell. This will create a new environment called **`dog-env`** and install all dependencies.

    $ conda env create -f environment.yml

Activate the newly created `dog-env` environment

    $ conda activate dog-env

## Running the app

To start the web app in production, navigate to the /app directory and run the main app run.py

    $ cd app
    $ python run.py

alternatively, start a debug server with flask
    
    $ cd app
    $ flask --debug --app=run.py run --host=0.0.0.0

That's it! The app should be up and running. Open your browser and navigate to http://127.0.0.1:8080 to try it.

(Or http://127.0.0.1:5000 if running the debug server).

# Project Structure and Libraries
The project follows a simple structure. All necessary files to run the web server are found in the `app/` directory. For more insight into the thought process of the project go to the `notebook/` directory.

    neural-networks-dogbreed
    ├───app/
    │   ├───run.py          # Main application that launches the web server.
    │   ├───predict.py      # Module containing prediction functions called by run.py.
    │   ├───models/         # Contains pre-trained model files and weight files.
    │   ├───static/         # Static assets for our web app: images and CSS.
    │   ├───templates/      # html templates of our index and results pages.
    │   └───uploads/        # holds the user's uploaded image during runtime
    └───notebook/
        ├───dog_app.html    # Project notebook
        └───dog_app.ipynb   # Project notebook (jupyter file)

This project is built in python3, the main libraries used are described below:

- **TensorFlow**: end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources. https://www.tensorflow.org/
- **Keras**: deep learning API running on top of TensorFlow. It was developed with a focus on enabling fast experimentation. https://keras.io/
- **Dlib**: toolkit containing machine learning algorithms http://dlib.net/
- **Open-CV**: Open Source Computer Vision Library. Includes several hundreds of computer vision algorithms and image processing functionality. https://opencv.org/
- **Numpy**: fundamental package for numerical computing with Python. https://numpy.org/
- **Flask**: lightweight framework for building web applications. https://flask.palletsprojects.com/
- **Waitress**: production-quality pure-Python WSGI server with "very acceptable" performance. https://github.com/Pylons/waitress

# Details

## Project Definition and Motivation

## Analysis

### Machine Learning techniques

#### **Dog vs Human differentiation**

We start by defining whether our image contains a dog or not. Dog's detection is done by implementing the pre-trained Resnet50 model. This model has been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.

If a dog is not detected, the algoritm attemps to find a human face. This is done by using dlib's mmod cnn model, a fairly accurate model for front-facing images. (Alternatively, the hog-svm model is used for faster results.)

#### **Dog's breed prediction**
The implementation builds upon the [Xception model](https://arxiv.org/abs/1610.02357) (CVPR 2017). By using Xception as a base model, we can leverage it's pre-trained features on our particular problem. 

#### **Imagenet**
ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories. Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


## Conclusions

### Aknowledgements




