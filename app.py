import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("lung_disease_model.h5")

st.title("Lung Disease Detection")

st.write("Upload a Chest X-ray image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

def preprocess(image):
    image = image.resize((224,224))
    img = np.array(image)/255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):

        img = preprocess(image)

        prediction = model.predict(img)

        if prediction[0][0] > 0.5:
            result = "Pneumonia Detected"
        else:
            result = "Normal Lung"

        st.success(result)
