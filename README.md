# Dog's Breed Prediction Project

This application helps the user find a dog's breed from a provided image file. 

![Capture](.\capture2.gif)

## User interface
The prediction flow goes as follows: 

* if a dog is detected in the image, return the predicted breed.
* if a human is detected in the image, return the resembling dog breed.
* if neither is detected in the image, provide output that indicates an error.


### Dog vs Human differentiation

We start by defining whether our image contains a dog or not. Dog's detection is done by implementing the pre-trained Resnet50 model. This model has been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.

If a dog is not detected, the algoritm attemps to find a human face. This is done by using dlib's mmod cnn model, a fairly accurate model for front-facing images. (Alternatively, the hog-svm model is used for faster results.)


## Transfer learning
### Dog's breed prediction
The implementation builds upon the [Xception model](https://arxiv.org/abs/1610.02357) (CVPR 2017). By using Xception as a base model, we can leverage it's pre-trained features on our particular problem. 



### Imagenet
ImageNet contains over 10 million URLs, each linking to an image containing an object from one of 1000 categories. Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.


# Installation and setup

### Installation

Clone this repository, then cd into it

    $ git clone https://github.com/sbadillo/neural-networks-dogbreed.git
    $ cd neural-networks-dogbreed

It is recommended to set-up a virtual environment before installing all dependencies. 

To do so using using conda run:

    $ conda create --name dog-env --file --requirements.txt

Or (alternatively)  using venv & pip

    $ python3 -m venv env
    $ source env/bin/activate
    $ pip install -r requirements.txt

### Model files
These two files are expected on the /models directory

    # dlibs' mmod cnn model for face detection:
    \models\mmod_human_face_detector.dat
    # Dog breed prediction top layers weight file:  
    \models\weights.best.Xception-final.hdf5

## Running the app

To start the web app in production, navigate to the /app directory and run the main app run.py

    $ cd app
    $ python run.py

alternatively, start a debug server with flask
    
    cd app
    flask --debug --app=run.py run --host=0.0.0.0

Ready! The app should be running. Open your browser and navigate to http://127.0.0.1:8080

(or http://127.0.0.1:5000 if running the debug server).



# General TODO : 

* [x] Get shit working
    * [x] set development server and run.py
    * [x] set basic input template
    * [x] get upload file
    * [x] get upload stream
    * [x] implement models api

* create results page
    * [x] implement prediction results
    * [x] add database of example images
    * [x] complete predict.html
    * [x] handle dog prediction 
    * [x] and human results
    * [x] handle no prediction

* [ ] Improve UX/UI
    * [x] styling results on index.html
    * [ ] try bootstrap progress bar
