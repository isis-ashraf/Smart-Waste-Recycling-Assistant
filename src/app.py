import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

# =========================
# Load models
# =========================
cnn_model = tf.keras.models.load_model("models/baseline_cnn_trashnet.h5")
vgg_model = tf.keras.models.load_model("models/vgg16_trashnet.h5")

class_names = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

st.title("♻️ Smart Waste Classification Demo")
st.write("Upload an image of waste to see predictions from CNN and VGG16 models.")

# =========================
# Image upload
# =========================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -------------------------
    # Preprocess for CNN
    # -------------------------
    img_cnn = image.resize((224, 224))
    img_array_cnn = np.array(img_cnn) / 255.0
    img_array_cnn = np.expand_dims(img_array_cnn, axis=0)

    # -------------------------
    # Preprocess for VGG16
    # -------------------------
    img_vgg = image.resize((224, 224))
    img_array_vgg = np.array(img_vgg)
    img_array_vgg = np.expand_dims(img_array_vgg, axis=0)
    img_array_vgg = preprocess_input(img_array_vgg)

    # -------------------------
    # Predict
    # -------------------------
    preds_cnn = cnn_model.predict(img_array_cnn)[0]
    preds_vgg = vgg_model.predict(img_array_vgg)[0]

    # -------------------------
    # Display top 3 predictions
    # -------------------------
    st.subheader("CNN Model Predictions")
    top_idx_cnn = preds_cnn.argsort()[-3:][::-1]
    for i in top_idx_cnn:
        st.write(f"{class_names[i]}: {preds_cnn[i]*100:.2f}%")

    st.subheader("VGG16 Model Predictions")
    top_idx_vgg = preds_vgg.argsort()[-3:][::-1]
    for i in top_idx_vgg:
        st.write(f"{class_names[i]}: {preds_vgg[i]*100:.2f}%")
