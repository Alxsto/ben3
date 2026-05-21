import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import json
import os

# -----------------------------
# Seitenkonfiguration
# -----------------------------
st.set_page_config(
    page_title="Fish AI Detector",
    page_icon="🐟",
    layout="centered"
)

st.title("🐟 Fish AI Detector")
st.write("Lade ein Bild hoch oder mache ein Foto eines Fisches.")

# -----------------------------
# Modell laden
# -----------------------------
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

# Prüfen ob Modell existiert
if not os.path.exists(MODEL_PATH):
    st.error("YOLO-Modell nicht gefunden! Bitte best.pt in /model ablegen.")
    st.stop()

model = load_model()

# -----------------------------
# Fischdaten laden
# -----------------------------
with open("fish_info.json", "r", encoding="utf-8") as f:
    fish_info = json.load(f)

# -----------------------------
# Bildquelle
# -----------------------------
option = st.radio(
    "Bildquelle wählen:",
    ["Bild hochladen", "Kamera verwenden"]
)

image = None

if option == "Bild hochladen":
    uploaded_file = st.file_uploader(
        "Bild auswählen",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

else:
    camera_image = st.camera_input("Foto aufnehmen")

    if camera_image:
        image = Image.open(camera_image)

# -----------------------------
# KI-Erkennung
# -----------------------------
if image is not None:

    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    if st.button("🔍 Fisch erkennen"):

        with st.spinner("KI analysiert Bild..."):

            img_array = np.array(image)

            results = model.predict(
                source=img_array,
                conf=0.4
            )

            result = results[0]

            boxes = result.boxes

            if len(boxes) == 0:
                st.warning("Kein Fisch erkannt.")
            else:

                detected_classes = []

                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    detected_classes.append(class_name)

                # Duplikate entfernen
                detected_classes = list(set(detected_classes))

                st.success("Fisch erkannt!")

                # Bild mit Bounding Boxes anzeigen
                plotted = result.plot()
                st.image(plotted, caption="Erkennung", use_column_width=True)

                # Infos anzeigen
                for fish in detected_classes:

                    st.subheader(f"🐠 {fish}")

                    if fish in fish_info:

                        info = fish_info[fish]

                        st.write(f"**Lateinischer Name:** {info['latin_name']}")
                        st.write(f"**Schonzeit:** {info['closed_season']}")
                        st.write(f"**Mindestmaß:** {info['min_size']}")
                        st.write(f"**Lebensraum:** {info['habitat']}")
                        st.write(f"**Info:** {info['fact']}")

                    else:
                        st.info("Keine zusätzlichen Informationen gefunden.")
