TODO

- [x] Web Application
	- [x]  There is a web application that utilizes data to inform how the web application works.
  - [x] Code
    - [x] Code is formatted neatly with comments and uses DRY principles.
    - [x] reproducibility and environment.yml file
- [ ] README.md file that communicates : 
  - [ ] TOC (updated)
  - [x] Instruction to run app.
  - [x] Explain files/structure of the repository
  - [x] Acknowledge main libraries used, 
  - [ ] Details
    - [x] 1. Project Definition, 
      - [x] 1.1 Motivation for the project
    - [ ] 2. Analysis
      - [ ] 2.2 a summary of the results of the analysis, 
    - [ ] 3. Conclusion.
    - [ ] Acknowledgements.
- [x] Git repository init
- [ ] submit

---

**Contents**

- [Introduction](#introduction)
  - [User interface](#user-interface)
  - [Finding dog's breed with artificial intelligence](#finding-dogs-breed-with-artificial-intelligence)
- [Quick start](#quick-start)
  - [Installation](#installation)
  - [Running the app](#running-the-app)
- [Project Structure and Libraries](#project-structure-and-libraries)
  - [Main Libraries](#main-libraries)
- [Details](#details)
  - [Project Definition and Motivation](#project-definition-and-motivation)
  - [Analysis](#analysis)
    - [Machine Learning techniques](#machine-learning-techniques)
      - [**Dog vs Human differentiation**](#dog-vs-human-differentiation)
      - [**Dog's breed prediction**](#dogs-breed-prediction)
      - [**Imagenet**](#imagenet)
  - [Conclusions](#conclusions)
    - [Acknowledgements](#acknowledgements)


# Introduction

An application that predicts a dog's breed from a user-provided image by using artificial neural networks.

![Capture](capture2.gif)

##  User interface
The web app follows this simple user story: 

* User uploads an image to the main page.
* if a dog is detected in the image, we display the predicted breed.
* for comparison, the results page provides some examples of the training set.
* as a fun twist, if a human is detected in the image, we return the resembling dog breed.
* if neither is detected in the image, we provide an error and some suggestions.

## Finding dog's breed with artificial intelligence
The project's backend demonstrates the concept of [**Transfer Learning**](https://en.wikipedia.org/wiki/Transfer_learning) by leveraging existing models and building on top of pre-trained **convolutional neural networks**. These models are used to identify dogs or humans, and later to predict the dog's breed.


# Quick start

## Installation
Clone this repository, then navigate into it

    $ git clone https://github.com/sbadillo/neural-networks-dogbreed.git
    $ cd neural-networks-dogbreed

It is recommended to set-up a python virtual environment using the `conda` package manager from the Anaconda distribution ([available here](https://www.anaconda.com/products/distribution)).

Run this command from a conda-enabled shell. This will create a new environment called **`dog-env`** and install all dependencies. Alternatively, you can install the [packages manually](/environment.yml).

    ## note: this command might take a minute or two.

    $ conda env create -f environment.yml

Activate the newly created `dog-env` environment

    $ conda activate dog-env


## Running the app

To start the web app in production, navigate to the `/app` directory and start the main app `run.py`

    $ cd app
    $ python run.py

alternatively, start a debug server with flask
    
    $ cd app
    $ flask --debug --app=run.py run --host=0.0.0.0

That's it! The app should be up and running. Open your browser and navigate to http://127.0.0.1:8080 to try it.

(Or http://127.0.0.1:5000 if running the debug server).


# Project Structure and Libraries
The project follows a simple structure. All necessary files to run the application are found in the `app/` directory. For more insight into the thought process of the project go to the `notebook/` directory.

    neural-networks-dogbreed
    ├───app/
    │   ├───run.py          # Main application module.
    │   ├───predict.py      # Module containing prediction functions.
    │   ├───models/         # Contains pre-trained model files and weight files.
    │   ├───static/         # Static assets for our web app: images and CSS.
    │   ├───templates/      # html templates of our index and results pages.
    │   └───uploads/        # holds the user's uploaded image during runtime
    └───notebook**/
        ├───dog_app.html    # Project notebook in rendered html format.
        └───dog_app.ipynb   # Project notebook in jupyter format.

**Please note that no dependencies are included to reproduce the jupyter notebook (.ipynb). For notebook reproducibility please refer to the [original repo](https://github.com/udacity/dog-project). 

Alternatively, the html notebook can be viewed [here](https://nbviewer.org/github/sbadillo/neural-networks-dogbreed/blob/master/notebook/dog_app.html).

For sake of keeping this repo tidy, the complete set of training/test/validation images used in training are NOT provided in this repository. As these are not necessary in runtime and are openly available in the [original project](https://github.com/udacity/dog-project) repository : 

  - [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
  - [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)


## Main Libraries

This project is built on python3, the main libraries are described here.

- **TensorFlow**: End-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries, and community resources. https://www.tensorflow.org
- **Keras**: Deep learning API running on top of TensorFlow. It was developed with a focus on enabling fast experimentation. https://keras.io
- **Dlib**: Toolkit containing machine learning algorithms http://dlib.net
- **Open-CV**: Open Source Computer Vision Library. Includes several hundreds of computer vision algorithms and image processing functionality. https://opencv.org
- **Numpy**: Fundamental package for numerical computing with Python. https://numpy.org
- **Flask**: Lightweight framework for building web applications in python. https://flask.palletsprojects.com
- **Waitress**: production-quality pure-Python WSGI server with "very acceptable" performance. https://github.com/Pylons/waitress

# Details

## Project Definition and Motivation

This project implements state-of-the art Convolutional Neural Network (CNN) models for object classification in images. Specifically, it aims to identify dogs and estimate the breed from a provided image. We build this application to demonstrate how modern deep learning techniques have evolved to solve real-world problems historically achievable only by humans.


The final product can be divided into two sections:

- A front-end user interface providing a simple workflow for the a user to upload an image.
- An python back-end which implements machine learning algorithms, making all computations available to the frontend through a RESTful API.

Limits:

Even though our main purpose is that of breed identification for the human's best friend, we let the our implementation apply prediction directly to human faces as well. We expand then our algorithm to perform human face identification, adding an amusing functionality to the user's experience.

Having set up the limits of our problem we can split our detection challenge into three main tasks

1. Object detection for dogs.
2. Object detection for humans (or human faces).
3. Object detection for dog breed. 

Motivation:

Artificial vision have fascinated humans for millennia. Throughout history, we have dreamed of artificial forms that are capable of making autonomous decisions and drastically increase our productivity. 

Up until contemporary history, visual object identification has been something that only intelligent life has been capable of doing. The human interest for vision systems continue today and have lead to an incredible research and advancement since the 60's [1]. From autonomous driving to security applications, computer vision has become a key field of artificial intelligence. 

This project aims to demonstrate state-of-the-art techniques in computer vision. Applying modern models into a common-life task. Dog breed identification is difficult even in some cases for humans. We will see how computers are able to tackle this problem in a programmatically and objective way.


## Analysis

Object detection is a common computer vision task. As the name implies, the task is to identify instances of objects in digital images (or video frames). 

### Deep Learning

To answer the question *what objects are where*, a computational model needs to run through pixels of an image in order to identify any patterns or "features". The extracted features allow the model to hypothesize over the presence of an object. We might call each extraction step as a single transformation. These transformations might also come in the form of image processing, such as color mapping and scaling.

To achieve complex object detection, these computational models attempt to create high-level abstractions, using architectures that support multiple and iterative transformations. This particular approach is known as **Deep Learning**, characterized by a layered and often non-linear architecture.

In this context, each detection task needed for our final project will need an algorithm performing a specific model. For instance, the model detecting a human's face will be different from the model that infers dog breed.

### **Dog vs Human differentiation**

We implement two different models in order to infer whether we are looking into a dogs or a human faces. Since our main subject is the former, we look for dogs first and fallback to human detection only if necessary. Dog-human identification will be mutually exclusive for sake of simplicity.

### Dataset
Deep learning models are characterized by their need for very large datasets to be trained and tested with. 
In addition, data (images in this case) must be expressed in matrix or tensor forms.

We provide our datasets for human and dogs detection

  - [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
  - [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

#### Detecting dogs
In order to fullfil the main case of our app, we need to ensure that a dog exists in the image. To achieve this, we will leverage existing models.

##### ImageNet and ResNet

ImageNet is an incredible effort to provide researchers around the world with image data for training large-scale object detection models. The project compiles over 1.2 million images from the public domain. In a very expensive labeling effort, images are organized into 1000 categories according to the WordNet hierarchy. 

https://www.image-net.org/

CITE 
Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015. paper | bibtex | paper content on arxiv | attribute annotations

However, when referring to ImageNet, we often refer to their annual ImageNet Large Scale Visual Recognition Challenge (ILSVRC). A challenge that serve as a benchmark (and have motivated) some of most sophisticated object detection algorithms to date.

Out of the [1000 labels](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) used in ImageNet, 133 correspond to breeds of dogs. This makes ImageNet baseline ideal for our project. 

#### ResNet-50

The resnet 50 model ...






#### Detecting humans


We start by defining whether our image contains a dog or not. Dog's detection is done by implementing the pre-trained Resnet50 model. This model has been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.

If a dog is not detected, the algorithm attempts to find a human face. This is done by using dlib's mmod cnn model, a fairly accurate model for front-facing images. (Alternatively, the hog-svm model is used for faster results.)

#### **Dog's breed prediction**
The implementation builds upon the [Xception model](https://arxiv.org/abs/1610.02357) (CVPR 2017). By using Xception as a base model, we can leverage it's pre-trained features on our particular problem. 

#### **Imagenet**
ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories. Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


## Conclusions

### Acknowledgements


	[1] The earliest applications were pattern recognition systems for character recognition in office automation related tasks : L. G. Roberts, Pattern Recognition With An Adaptive Network, in: Proc. IRE International Convention Record, 66–70, 1960 and J. T. Tippett, D. A. Borkowitz, L. C. Clapp, C. J. Koester, A. J. Vanderburgh (Eds.), Optical and Electro-Optical Information Processing,
	MIT Press, 1965.



