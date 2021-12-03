# keras imports
from flask import Flask, render_template, request, send_from_directory
from keras.preprocessing.image import img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from sklearn.linear_model import LogisticRegression
import h5py
import numpy as np
import pickle as pk
import os
app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"


first_check = VGG16(weights='imagenet')
second_check = pk.load(open("phone_nophone_model/classifier.pickle", 'rb')) #phone vs nophone
third_check = pk.load(open("damage_model/classifier.pickle", 'rb'))# damage vs no damage
fourth_check = pk.load(open("Severity_model/classifier.pickle", 'rb')) #high vs low


def prepare_img_224(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def phone_nophone_check(classifier,x):
    base_model = first_check
    train_labels = ['not a phone', 'phone']
    
    model = Model(base_model.input, base_model.get_layer('fc1').output)
    feature = model.predict(x)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    preds = classifier.predict(flat)
    prediction = train_labels[preds[0]]
    
    if train_labels[preds[0]] == 'phone':
        return "Validation complete ,It's a phone." 
    else:
        return "no-phn"
    
    
def damage_nodamage_check(classifier,x):
    base_model = first_check
    train_labels = ['damage', 'no-damage']
    
    model = Model(base_model.input, base_model.get_layer('fc1').output)
    feature = model.predict(x)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    preds = classifier.predict(flat)
    prediction = train_labels[preds[0]]
    
    if train_labels[preds[0]] == 'damage':
        return "Validation complete , Phone is damaged"
    else:
        return "no-damage"  
    
def cleandir(directo):
    for i in os.listdir(f'{directo}'):
        print(i)
        os.remove(f'{directo}/{i}')
    
def high_low_check(classifier,x):
    base_model = first_check
    train_labels = ['Severe', 'Minor']
    
    model = Model(base_model.input, base_model.get_layer('fc1').output)
    feature = model.predict(x)
    flat = feature.flatten()
    flat = np.expand_dims(flat, axis=0)
    preds = classifier.predict(flat)
    prediction = train_labels[preds[0]]
    return 'Severity assesment complete. Your phone damage impact is -' + train_labels[preds[0]]
#######################################################################################################################

def classify(img):
    try:
        img_224 = prepare_img_224(img)
        res = []
        g1 = phone_nophone_check(second_check,img_224)
        res.append(g1)
        if g1 == "no-phn":
            res = []
            res = ["Are you sure this is a picture of your phone? Please submit another picture of your phone."]
            return res
        
        g2 = damage_nodamage_check(third_check,img_224)
        res.append(g2)

        if g2 == "no-damage":
            return res

        g3 = high_low_check(fourth_check,img_224)
        res.append(g3)
        return res
    except:
        return ("Image not accessible. Please try again.")

# Predict & classify image

# home page
@app.route("/")
def home():
    cleandir("uploads")
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():

    if request.method == "GET":
        return render_template("home.html")

    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print(upload_image_path)
        file.save(upload_image_path)

        label = classify(upload_image_path)
        print("++++++++++++++++")
        print(label)
    return render_template(
        "classify.html", image_file_name=file.filename, label=label,
    )


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    #port = int(os.environ.get("PORT", 5000))
    #app.run(host='0.0.0.0', port=port)
    app.run()