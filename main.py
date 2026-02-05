# ======================
# ENVIRONMENT FIXES (MUST BE FIRST)
# ======================
import os
import tempfile
os.environ["DEEPFACE_HOME"] = os.path.join(os.path.expanduser("~"), ".deepface")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TF noise

# ======================
# IMPORTS
# ======================
import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

from deepface import DeepFace

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="Face Recognition POC", layout="wide")
st.title("🔍 Face Recognition Demo (DeepFace – Facenet512)")

KNOWN_FACES_DIR = "known_faces"

# ======================
# LOAD FACENET MODEL (OFFLINE, CACHED)
# ======================
@st.cache_resource
def load_facenet_model():
    """
    Loads Facenet512 once.
    No downloads. Uses local .h5 only.
    """
    model = DeepFace.build_model("Facenet512")
    return model

try:
    facenet_model = load_facenet_model()
    deepface_available = True
    st.success("✅ Facenet512 model loaded locally (offline)")
except Exception as e:
    deepface_available = False
    st.error("❌ Facenet512 could not be loaded locally")
    st.error(str(e))

# ======================
# FALLBACK FACE FEATURES
# ======================
def detect_faces_cascade(image_bgr):
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)

def extract_face_features_fallback(image_bgr, face_rect):
    x, y, w, h = face_rect
    face = image_bgr[y:y+h, x:x+w]
    face = cv2.resize(face, (128, 128))
    features = []
    for ch in cv2.split(face):
        hist = cv2.calcHist([ch], [0], None, [32], [0, 256])
        features.extend(hist.flatten())
    return np.array(features, dtype=np.float32)

# ======================
# LOAD KNOWN FACES
# ======================
@st.cache_data(show_spinner=False)
def load_known_faces():
    known_features = []
    known_names = []

    base = Path(KNOWN_FACES_DIR)
    if not base.exists():
        return known_features, known_names

    for person_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        name = person_dir.name

        image_file = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
            candidate = person_dir / f"{name}{ext}"
            if candidate.exists():
                image_file = candidate
                break

        if image_file is None:
            continue

        img = cv2.imread(str(image_file))
        if img is None:
            continue

        try:
            if deepface_available:
                emb = DeepFace.represent(
                    img_path=str(image_file),
                    model_name="Facenet512",
                    model=facenet_model,
                    enforce_detection=False
                )
                if emb:
                    known_features.append(np.array(emb[0]["embedding"], dtype=np.float32))
                    known_names.append(name)
            else:
                faces = detect_faces_cascade(img)
                if len(faces) > 0:
                    known_features.append(
                        extract_face_features_fallback(img, faces[0])
                    )
                    known_names.append(name)
        except:
            continue

    return known_features, known_names

known_features, known_names = load_known_faces()

# ======================
# UI – DATABASE STATUS
# ======================
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.subheader("📊 Database Status")
    if known_names:
        st.success(f"Loaded {len(known_names)} identities")
        st.write(", ".join(sorted(known_names)))
    else:
        st.error("No known faces loaded")

with col2:
    st.subheader("ℹ️ Info")
    st.write("Model: Facenet512 (offline)")
    st.write("Backend: TensorFlow")

with col3:
    st.subheader("🔄 Reload")
    if st.button("Reload Faces"):
        load_known_faces.clear()
        st.experimental_rerun()

st.divider()

# ======================
# UPLOAD & RECOGNITION
# ======================
st.subheader("🖼️ Upload Image")
uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image_pil.save(tmp.name)
        tmp_path = tmp.name

    try:
        emb = DeepFace.represent(
            img_path=tmp_path,
            model_name="Facenet512",
            model=facenet_model,
            enforce_detection=False
        )

        if not emb:
            st.error("No face detected")
        else:
            test_vec = np.array(emb[0]["embedding"], dtype=np.float32)

            distances = [np.linalg.norm(test_vec - k) for k in known_features]
            best_idx = int(np.argmin(distances))
            best_dist = distances[best_idx]
            best_name = known_names[best_idx]

            st.divider()
            st.subheader("🎯 Result")

            if best_dist < 0.6:
                st.success(f"Match: {best_name}")
                st.write(f"Distance: {best_dist:.4f}")
            else:
                st.warning("No confident match")
                st.write(f"Closest: {best_name}")
                st.write(f"Distance: {best_dist:.4f}")

    except Exception as e:
        st.error(str(e))

st.divider()
st.caption("Offline Face Recognition POC – Corporate-safe")
