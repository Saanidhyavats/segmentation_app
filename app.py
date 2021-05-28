# Importing essential libraries and modules
from flask import Flask, render_template, request, Markup
import numpy as np
import io
import cv2
import os
import tensorflow as tf
import base64
from PIL import Image
import matplotlib.pyplot as plt
#sm.set_framework('tf.keras')
from tensorflow.keras.models import load_model
os.environ["SM_FRAMEWORK"]='tf.keras'
import segmentation_models as sm

def modelarch():
 BACKBONE = 'resnet34'
 preprocess_input = sm.get_preprocessing(BACKBONE)
 model = sm.Unet('resnet34', classes=3, activation='softmax')
 model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
 )
 return model


# loading the model
print('opening model')
nxModel=modelarch()
nxModel.load_weights('model_seg1.h5')
"""x1=cv2.cvtColor(cv2.resize(cv2.imread("A_000009.png"),(320,320)),cv2.COLOR_BGR2RGB)
x2=cv2.cvtColor(cv2.resize(cv2.imread("A_000009y.png"),(320,320)),cv2.COLOR_BGR2RGB)
print(plt.imshow(x1))
print(plt.imshow(x2))
x1=x1.reshape(1,320,320,3).astype(np.float32)
z=model2.predict(x1)
z=z.reshape(320,320,3)
print(plt.imshow(z))"""
print('model loaded')


# =========================================================================================

# Custom functions for calculations




# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def disease_prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "nofile"
        file = request.files['file']
        if not file:
            return "image not received"
        try:
            print('y')
            print('h')
            img = Image.open(request.files['file'])
            image = img
            if image:
                print('yes')
            else:
                print('no')
            print('myimage')
            print(image)
            image = image.resize((320,320))
            print('s')
            image = np.asarray(image)
            print('u')
            n_image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            print('c')
            x1=cv2.cvtColor(n_image,cv2.COLOR_BGR2RGB)
            x1=x1.reshape(1,320,320,3).astype(np.float32)
            z=nxModel.predict(x1)
            z=z.reshape(320,320,3)
            myimage = cv2.cvtColor(z,cv2.COLOR_RGB2BGR)
            imagebytes = cv2.imencode('.jpg',myimage*255)[1].tostring()
            imagebytes1 = cv2.imencode('.jpg',image)[1].tostring()
            print(imagebytes)
            encoded_img_data = base64.b64encode(imagebytes)
            print("done")
            return render_template('diseaseresult.html',img_data = encoded_img_data.decode('utf-8'))
        except:
            pass
    return render_template('mytemplate.html')


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False, host='localhost',port=8000)    


    
