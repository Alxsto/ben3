import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Titel der App
st.title("YOLO Objekterkennung mit Streamlit")

# Modell laden (erstes Mal wird es automatisch heruntergeladen)
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # leichtes Modell
    return model

model = load_model()

# Datei-Upload
uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file)
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # In NumPy konvertieren
    img_array = np.array(image)

    # YOLO Vorhersage
    results = model(img_array)

    # Ergebnisbild erzeugen
    annotated_frame = results[0].plot()

    # BGR -> RGB
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Ergebnis anzeigen
    st.image(annotated_frame, caption="Erkannte Objekte", use_column_width=True)

    # Klassen + Confidence anzeigen
    st.subheader("Erkannte Objekte:")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        st.write(f"{label}: {conf:.2f}")
