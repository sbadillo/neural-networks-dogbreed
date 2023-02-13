#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""
Breed prediction's app main module

Set's up flask application, calls predict.py and performs html rendering.
"""

__version__ = "1.0.0"
__author__ = "Sergio Badillo"

import os
from flask import Flask
from flask import (  # pylint: disable=unused-import
    render_template,
    url_for,
    request,
    abort,
    send_from_directory,
)
from werkzeug.utils import secure_filename
import predict


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # disable tensorflow compile warnings


UPLOAD_FOLDER = "uploads"
UPLOAD_EXTENSIONS = ["jpg", "jpeg", "png", "gif", "webp"]
MAX_CONTENT_LENGTH = 1024 * 1024 * 14  # 14 MB
EXAMPLE_DIR = "./static/example_images"


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_EXTENSIONS"] = UPLOAD_EXTENSIONS


def allowed_file(filename):
    """Helper function returns whether a file is allowed or not.
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
        directory (str): directory path.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if "user_img" in filename:
                os.unlink(file_path)

        except KeyError as ex:
            print(f"Failed to delete {file_path}. Reason: {ex}")


def find_examples(breed):
    """Finds and returns the path to a directory containing
    example dog images for breed.

    Args:
        breed (str): dog breed, according to Imagenet labels

    Returns:
        list: paths to image files
    """
    folder_name = next(d for d in os.listdir(EXAMPLE_DIR) if d[4:] == breed)
    dir_breed = os.path.join(EXAMPLE_DIR, folder_name)
    return [os.path.join(dir_breed, i) for i in os.listdir(dir_breed)]


@app.route("/")
def index():
    """Render the app's index page.

    Returns:
        flask's render_template instance of index.html
    """

    clean_dir(UPLOAD_FOLDER)  # clean all user images upon page load

    return render_template(
        "index.html",
    )


@app.route("/predict", methods=["GET", "POST"])
def call_predict():
    """Calls for a prediction on posted image file.

    This functions digests the POST request and intercepts the uploaded file.
    If the file is safely secured io disk, the predict.predict_final() function is called.
    The returned html rendered page depend on the results of the prediction:
        - predict.html for succesful human or dog detections
        - no_predict.html for unsuccesful detections.

    Returns:
        render: rendered html-jinja page depending on output
    """

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
    """returns user message for error case"""
    print(error)
    return "File is too large (Max file size 14 MB).", 413


@app.errorhandler(400)
def invalid_type(error):
    """returns user message for error case"""
    print(error)
    return "Not a valid image file, please try again.", 400


def main():
    """Runs the application's WSGI server on local:8080

    NOTE: to start debug server with flask run:
        $ flask --debug --app=run.py run --host=0.0.0.0
    """

    # start waitress production server
    print("🐕🏃Starting production server on http://127.0.0.1:8080")
    waitress.serve(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    import waitress

    main()
