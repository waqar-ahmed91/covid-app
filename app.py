import streamlit as st
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image

st.image('omdena-logo.png',use_column_width=True)
st.markdown('<h1 style="color:black;">COVID X-Ray Classification Model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">This classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> COVID and Non-COVID</h3>', unsafe_allow_html=True)

img_height = 299
img_width = 299


# Load the model
model = load_model('2023-09-07_xception.h5', compile=False)

upload= st.file_uploader('Upload X-ray image of a patient for classification', type=["png", "jpg", "jpeg"])
c1, c2= st.columns(2)
c1.header('Uploaded Image')
c2.header('Predicted Class')


if upload is not None:
    img = Image.open(upload)
    img = img.resize((299, 299))
    img = img.convert("RGB")
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predict_coef = model.predict(img)
    predict_coef = np.round(predict_coef)
    img = load_img(upload, target_size=(img_width,img_height))
  # image = img_to_array(img)
  # image = np.expand_dims(image, axis=0)
    c1.image(img)
  # prediction = model.predict(image)
  # prediction = (prediction > 0.5).astype(int)
  # print(prediction)
    if predict_coef == 0:
      c2.write('COVID Not Detected')
    else:
      c2.write('COVID Detected')

