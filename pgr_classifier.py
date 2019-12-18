from flask import Flask, request, redirect, flash, url_for, send_from_directory
from flask import render_template_string
# from werkzeug.utils import secure_filename
from img_clustering import guess_image
from keras import Input
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# import os

app = Flask(__name__)

UPLOAD_FOLDER = '/workspace/uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

vgg_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(512, 512, 3)))
model = Sequential(name="VGG16_Preprocessor")
for layer in vgg_model.layers[:]:
    model.add(layer)
model.add(Flatten())
model.add(Dense(4096, input_shape=(131072,)))


def vectorize(filename):
    image = load_img(filename, target_size=(512, 512))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    # print(model.predict(image)[0])
    image_vector = model.predict(image)[0]
    return image_vector


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(filename)
            return redirect(url_for('uploaded_file', filename=filename))
    return '''
    <!doctype html>
    <title>Classify Image</title>
    <h1>Choose Image</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Classify>
    </form>
    '''


@app.route('/images/<filename>', strict_slashes=False)
def serve_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    imageFile = "%s" % (filename)
    imageVector = vectorize(imageFile)
    guesses = guess_image([imageVector])[0]
    garbage_prob = "Garbage Problem: " + str(round(guesses[0]*100, 2)) + "%"
    stl_prob = "Streetlight Problem: " + str(round(guesses[1]*100, 2)) + "%"
    tree_prob = "Tree Problem: " + str(round(guesses[2]*100, 2)) + "%"
    toilet_prob = "Toilet Problem: " + str(round(guesses[3]*100, 2)) + "%"
    pothole_prob = "Pothole Problem: " + str(round(guesses[4]*100, 2)) + "%"
    nota_prob = "None of the Above: " + str(round(guesses[5]*100, 2)) + "%"
    return render_template_string(
        '''
    <!doctype html>
    <title>Classify Image</title>
    <h1>Choose Image</h1>
    <form method=post enctype=multipart/form-data action="/">
      <p><input type=file name=file>
         <input type=submit value=Classify>
    </form>
    <h2>{{garbage_prob}}</h2>
    <h2>{{streetlight_prob}}</h2>
    <h2>{{tree_prob}}</h2>
    <h2>{{toilet_prob}}</h2>
    <h2>{{pothole_prob}}</h2>
    <h2>{{nota_prob}}</h2>
    <img height="600px" src={{url_for('uploaded_file', filename=filename)}}>
    ''',
        filename=filename,
        garbage_prob=garbage_prob,
        streetlight_prob=stl_prob,
        tree_prob=tree_prob,
        toilet_prob=toilet_prob,
        pothole_prob=pothole_prob,
        nota_prob=nota_prob)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
