from flask import Flask,render_template,request
from PIL import Image
import io
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
import timeit
import base64





# load json and create model
from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


#Instantiating the class Flask 
#app is the object that holds our project
app=Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")
@app.route('/links')
def showLinks():
    return render_template("links.html")

@app.route('/devs')
def showDevs():
    return render_template("devs.html")

@app.route('/predict',method=['POST'])
def predict():
    #Catching the data received 
    data=(request.form)
    #Slicing out only the required portion of the data (BASE64 string)
    #converting the clipped string into byte format as it is the supported format by the method decodebytes()
    data=base64.decodebytes(bytes((data['value'][22:]),'utf-8'))
    image = Image.open(io.BytesIO(data))
    #converting the image to greyscale
    image=image.convert('L')
    #resizing with smoothing (ANTIALIAS)
    image=image.resize((28,28),Image.ANTIALIAS)
    #converting the image to array
    image = np.asarray(image)
    #dividing each pixel intensity by 255 to apply MINMAX scaling
    image=image.astype('float32')/255
    #converting the image shape to that of the training data as it is what the model accepts
    image=image.reshape(1,28,28,1)
    #storing the index of the output array which has the greatest probabilistic value
    number=np.argmax(loaded_model.predict(image))
    #returning predicted number as a response
    if((type(number)!=None)):
        return ("PREDICTED NUMBER : "+str(number))
    else:
        return ("UNABLE TO PREDICT")

if __name__=='__main__':
    app.run()

