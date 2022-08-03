from __future__ import division, print_function


import os


import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template



# Define a flask app
app = Flask(__name__,static_url_path='/static')

# Model saved with Keras model.save()
MODEL_PATH ='model_inc.h5'

# Load your trained model
model = load_model(MODEL_PATH)


print("yes")

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(200, 200))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==1:
        preds="cataract"
    elif preds==2:
        preds="Glaucoma"
    elif preds==0:
        preds="Normal"
    else:
        preds="Retina Diseases"
      
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index1.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        #basepath = os.path.dirname(__file__)
        #file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        #f.save(file_path)

        # Make prediction
        preds = model_predict(f, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

