import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests

# Model download link
MODEL_URL = "https://your-model-link/lung_disease_model.h5"
MODEL_PATH = "lung_disease_model.h5"

# Download model if not present
@st.cache_resource
def load_model():

    try:
        with open(MODEL_PATH, "rb"):
            pass
    except:
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

st.title("Lung Disease Detection")

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg","png","jpeg"])

def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image)

    if st.button("Predict"):

        img = preprocess(image)

        prediction = model.predict(img)

        if prediction[0][0] > 0.5:
            st.success("Pneumonia Detected")
        else:
            st.success("Normal Lung")
