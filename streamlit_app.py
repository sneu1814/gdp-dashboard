import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# -----------------------------
# Build CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -----------------------------
# UI
# -----------------------------
st.title("Deepfake Detection System")
st.write("Upload an image to detect whether it is Real or Fake")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    score = float(prediction[0][0])

    st.write("### Detection Result")

    if score < 0.5:
        st.success("Real Image")
    else:
        st.error("Fake Image")

    st.write("Confidence Score:", round(score, 2))