import streamlit as st
import numpy as np
import cv2
import joblib
import tensorflow as tf
from PIL import Image

# Page configuration
st.set_page_config(page_title="Paddy Disease Detection", layout="wide")

st.title("🌾 Early Detection of Paddy Crop Diseases")
st.subheader("Hybrid AI Model (CNN + SVM)")

st.write("""
This dashboard demonstrates **Objective-1** of the research work:
Implementation of a **Hybrid Deep Learning Model (CNN + SVM)** 
for accurate classification of paddy crop diseases.
""")

# Load models
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("models/cnn_feature_model.h5")
    svm_model = joblib.load("models/svm_model.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return cnn_model, svm_model, encoder

cnn_model, svm_model, encoder = load_models()

# Image preprocessing
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Sidebar
st.sidebar.title("Upload Paddy Leaf Image")
uploaded_file = st.sidebar.file_uploader("Choose Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # preprocessing
    img = preprocess_image(image)

    # CNN feature extraction
    features = cnn_model.predict(img)

    # SVM classification
    prediction = svm_model.predict(features)
    probability = svm_model.predict_proba(features)

    # Decode predicted label
    disease = encoder.inverse_transform(prediction)[0]
    confidence = np.max(probability) * 100

    st.success(f"Predicted Disease : {disease}")
    st.info(f"Confidence : {confidence:.2f}%")

    st.write("Prediction Probability")

    prob_dict = {}
    class_names = encoder.classes_

    for i, name in enumerate(class_names):
        prob_dict[name] = probability[0][i]

    st.bar_chart(prob_dict)

else:
    st.warning("Please upload a paddy leaf image.")