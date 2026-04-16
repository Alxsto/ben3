
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Seitentitel
st.set_page_config(page_title="Digitales Fundbüro", layout="centered")

st.title("🔍 Digitales Fundbüro")
st.write("Lade ein Bild hoch und die KI erkennt den Gegenstand.")

# Modell laden
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/keras_model.h5", compile=False)
    return model

model = load_model()

# Labels laden
def load_labels():
    with open("model/labels.txt", "r") as f:
        labels = f.readlines()
    return [label.strip() for label in labels]

labels = load_labels()

# Bild vorbereiten
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Standard bei Teachable Machine
    img_array = np.array(image)
    img_array = (img_array / 127.5) - 1  # Normalisierung
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Upload Bereich
uploaded_file = st.file_uploader("📸 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    st.write("🔎 Analysiere Bild...")

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    predicted_label = labels[index]

    st.success(f"🧾 Gegenstand erkannt: **{predicted_label}**")
    st.info(f"📊 Sicherheit: **{confidence * 100:.2f}%**")

    # Optional: alle Wahrscheinlichkeiten anzeigen
    st.subheader("Weitere mögliche Treffer:")
    for i, label in enumerate(labels):
        st.write(f"{label}: {prediction[0][i] * 100:.2f}%")
