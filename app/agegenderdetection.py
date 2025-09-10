import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os


import requests

def download_model():
    url = "https://www.dropbox.com/scl/fi/1jhjbbxrhyp1kk2lmkh00/model_11.h5?rlkey=ub1c1rehfljsonl9fmhqwtxx5&st=ninkel24&dl=1"
    output = "model_11.h5"
    if not os.path.exists(output):
        r = requests.get(url, allow_redirects=True)
        with open(output, "wb") as f:
            f.write(r.content)
    return output



# Load the trained model
@st.cache_resource
def load_model():
    model_path = download_model()
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

st.title("👤 Age & Gender Prediction")
st.write("Upload a face image, and the model will predict **Gender** and **Age**.")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess (resize to model input shape)
    img = image.convert("L").resize((128, 128))   # "L" means grayscale
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)   # add channel dimension
    img_array = np.expand_dims(img_array, axis=0)    # add batch dimension

    # Make predictions
    gender_pred, age_pred = model.predict(img_array)

    # Decode Gender
    gender_label = "Female" if gender_pred[0][0] > 0.5 else "Male"

    # Decode Age (rounded since it’s regression)
    predicted_age = int(age_pred[0][0])

    st.success(f"Predicted Gender: **{gender_label}**")
    st.info(f"Predicted Age: **{predicted_age} years**")



