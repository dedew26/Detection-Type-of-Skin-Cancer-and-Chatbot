from flask import Flask, render_template, request
import os
from keras.models import load_model
import numpy as np
from PIL import Image
from process import preparation, generate_response

# download nltk
preparation()

# from tensorflow.keras.models import load_model
# from tensorflow import keras
# from tf.keras.saving import load_model
# import tensorflow as tf
# tf.keras.models.load_model

app = Flask(__name__)
app.static_folder= 'static'

app.config['UPLOAD_FOLDER'] = 'static/upload/'  # Folder untuk menyimpan file gambar
model = load_model('Mymodel.h5')  # Ubah 'model.h5' sesuai dengan nama file model Anda

@app.route('/Detection', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        filename.replace(" ", "%20")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocessing gambar
        img = Image.open(filepath)
        img = img.resize((220, 220)) 
        img = np.array(img) # Ubah ukuran gambar sesuai kebutuhan model Anda
        img = img/255
        img = np.expand_dims(img, axis=0)

        # Prediksi dengan model
        prediction = model.predict(img)
        labels = ['acticnis keratosis', 'basal cell carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'serborrheric keratosis', 'vascular lesion'] 
        predicted_label = labels[np.argmax(prediction)]
        image_src = f"/static/upload/{filename}"
        print("ImageSRC: ",image_src)
        # Lakukan sesuatu dengan hasil prediksi
        return render_template('Detection.html', hasil=predicted_label, image_src=image_src)
    return render_template('Detection.html')

@app.route("/")
def Index():
    return render_template("Index.html")

@app.route("/Home")
def Home():
    return render_template("Home.html")

@app.route("/Detection")
def detection():
    return render_template("Detection.html")

@app.route("/Q-and-A")
def qa():
    return render_template("Q-and-A.html")

@app.route("/About-Us")
def about():
    return render_template("About-Us.html")

@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result

if __name__ == "__main__":
    app.run(debug=True)