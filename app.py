import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("lung_disease_model.h5")

# Title
st.title("Lung Disease Detection using AI")

st.write("Upload a Chest X-ray image to detect Pneumonia.")

# Upload image
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","png","jpeg"])

def preprocess_image(image):
    image = image.resize((224,224))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("Predict Disease"):

        img_array = preprocess_image(image)

        prediction = model.predict(img_array)

        confidence = prediction[0][0]

        if confidence > 0.5:
            result = "Pneumonia Detected"
        else:
            result = "Normal Lung"

        st.subheader("Prediction Result")
        st.success(result)

        st.write("Confidence Score:", round(float(confidence)*100,2), "%")

        # Simple report
        report = f"""
        AI Medical Report

        Prediction: {result}
        Confidence: {round(float(confidence)*100,2)}%

        Recommendation:
        Please consult a medical professional for confirmation.
        """

        st.download_button(
            "Download Report",
            report,
            file_name="AI_Xray_Report.txt"
        )
