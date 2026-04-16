import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Titel der App
st.title("🔍 Digitales Fundbüro")
st.write("Lade ein Bild hoch und die KI erkennt den Gegenstand.")

# Modell laden
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    return model

model = load_model()

# Labels laden
def load_labels():
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

labels = load_labels()

# Bild hochladen
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten (Teachable Machine nutzt meist 224x224)
    size = (224, 224)
    image = image.resize(size)
    image_array = np.asarray(image)

    # Normalisierung (wichtig für Teachable Machine Modelle)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Batch Dimension hinzufügen
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = labels[index]
    confidence_score = prediction[0][index]

    # Ergebnis anzeigen
    st.subheader("📦 Ergebnis")
    st.write(f"**Gegenstand:** {class_name}")
    st.write(f"**Sicherheit:** {confidence_score * 100:.2f}%")

    # Optional: alle Wahrscheinlichkeiten anzeigen
    if st.checkbox("Alle Wahrscheinlichkeiten anzeigen"):
        for i, label in enumerate(labels):
            st.write(f"{label}: {prediction[0][i] * 100:.2f}%")
