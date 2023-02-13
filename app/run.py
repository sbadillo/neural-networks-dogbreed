#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""
Breed prediction's app main module

Set's up flask application, calls predict.py and performs html rendering.
"""

__version__ = "1.0.0"
__author__ = "Sergio Badillo"

import os
import shutil
import cv2

from flask import Flask
from flask import (
    render_template,
    url_for,
    request,
    abort,
    send_from_directory,
)

from werkzeug.utils import secure_filename
import predict

UPLOAD_FOLDER = "uploads"
UPLOAD_EXTENSIONS = ["jpg", "jpeg", "png", "gif"]
MAX_CONTENT_LENGTH = 1024 * 1024 * 14  # 14 MB
EXAMPLE_DIR = "./static/example_images"


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_EXTENSIONS"] = UPLOAD_EXTENSIONS


def allowed_file(filename):
    """_summary_
    Args:
        filename (string): name of file
    Returns:
        boolean: whether file extension is in ALLOWED_EXTENSIONS or not
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["UPLOAD_EXTENSIONS"]
    )


def clean_dir(directory):
    """Erases user images in directory.

    Args:
        directory (str): directory path
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if "user_img" in filename:
                os.unlink(file_path)

        except KeyError as ex:
            print(f"Failed to delete {file_path}. Reason: {ex}")


def find_examples(breed):
    """finds the directory of example images for breed.

    Args:
        breed (str): dog breed, according to Imagenet labels

    Returns:
        list: paths to image files
    """
    folder_name = next(d for d in os.listdir(EXAMPLE_DIR) if d[4:] == breed)
    dir_breed = os.path.join(EXAMPLE_DIR, folder_name)
    return [os.path.join(dir_breed, i) for i in os.listdir(dir_breed)]


# index webpage receives user input for model
@app.route("/")
def index():
    """Render the app index page.

    Returns:
        flask's render_template instance of master.html
    """

    # this will clean all files present in the uploads folder
    clean_dir(UPLOAD_FOLDER)

    return render_template(
        "index.html",
    )


@app.route("/predict", methods=["GET", "POST"])
def call_predict():
    """summary"""

    if request.method == "POST":
        file = request.files["file"]

        # store image in a secure way to ./uploads
        filename = secure_filename(file.filename)
        if filename != "":
            if not allowed_file(filename):
                abort(400)
            extension = filename.rsplit(".", 1)[1].lower()
            file_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "user_img" + "." + extension
            )
            file.save(file_path)

        # call prediction function on img
        img_path = file_path

        results = predict.predict_final(img_path)
        print(results[0])
        breed = results[0]["breed"]

        # Find the directory for example images.
        examples = None
        if breed:
            examples = find_examples(breed)

            return render_template(
                "predict.html",
                user_img_path=img_path,
                is_dog=results[0]["is_dog"],
                is_human=results[0]["is_human"],
                breed=breed,
                examples=examples,
            )

        return render_template("no_predict.html", user_img_path=img_path)

    return ""


# ? Note: this function is run from template to retrieve the image path
@app.route("/uploads/<filename>")
def upload(filename):
    """summary"""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.errorhandler(413)
def too_large(error):
    """helper"""
    return "File is too large", 413


@app.errorhandler(400)
def invalid_type(error):
    """helper"""
    return "Not a valid image file", 400


def main():
    """runs app"""
    # app.run()     # run app with flask

    # using Waitress as WSGI production server
    waitress.serve(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    import waitress

    main()
