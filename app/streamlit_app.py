"""
Streamlit web application for Skin Cancer Detection.

Launch:
    streamlit run app/streamlit_app.py
    python main.py app
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.hybrid_model import HybridSkinCancerModel
from utils.config import load_config

# Page configuration
st.set_page_config(
    page_title="Skin Cancer Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_LABELS = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}


@st.cache_resource
def load_model():
    """Load model once and cache."""
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    hybrid = HybridSkinCancerModel(config)
    checkpoint = config.checkpoint_dir / config.get("classification.checkpoint_name", "classifier_best.keras")
    if checkpoint.exists():
        import tensorflow as tf
        hybrid.classification_model = tf.keras.models.load_model(checkpoint, compile=False)
    return hybrid, config


def render_sidebar(config) -> dict:
    """Render sidebar with clinical metadata inputs."""
    st.sidebar.header("Clinical Metadata")
    age = st.sidebar.slider("Age", min_value=1, max_value=100, value=50)
    gender = st.sidebar.selectbox("Gender", config.get("metadata.gender_classes", ["male", "female", "unknown"]))
    location = st.sidebar.selectbox(
        "Lesion Location",
        config.get("metadata.location_classes", ["unknown"]),
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This system is for research purposes only. "
        "Always consult a qualified dermatologist for medical diagnosis."
    )
    return {"age": age, "gender": gender, "location": location}


def render_risk_badge(risk_level: str) -> None:
    """Display color-coded risk level."""
    if "HIGH" in risk_level.upper():
        st.error(f"⚠️ {risk_level}")
    elif "MODERATE" in risk_level.upper():
        st.warning(f"⚡ {risk_level}")
    else:
        st.success(f"✅ {risk_level}")


def main() -> None:
    st.title("🔬 Explainable Skin Cancer Detection System")
    st.markdown(
        "*An Explainable Hybrid Deep Learning Framework for Early Skin Cancer Detection "
        "Using Image Enhancement, Lesion Segmentation, Clinical Metadata Fusion, "
        "and Attention-Based Classification*"
    )

    try:
        hybrid, config = load_model()
    except Exception as exc:
        st.warning(f"Model not loaded ({exc}). Running in demo mode with untrained architecture.")
        config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
        hybrid = HybridSkinCancerModel(config)
        try:
            hybrid.build_classifier()
        except Exception:
            pass

    metadata = render_sidebar(config)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Skin Lesion Image")
        uploaded_file = st.file_uploader(
            "Choose a dermoscopic image (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            image_array = np.array(image)
        else:
            st.info("Please upload a skin lesion image to begin analysis.")
            image_array = None

    with col2:
        st.subheader("Prediction Results")

        if image_array is not None and st.button("🔍 Predict", type="primary", use_container_width=True):
            with st.spinner("Analyzing lesion... Running hybrid deep learning pipeline..."):
                try:
                    result = hybrid.predict(
                        image=image_array,
                        age=metadata["age"],
                        gender=metadata["gender"],
                        location=metadata["location"],
                    )

                    disease = result["disease"]
                    st.metric(
                        label="Predicted Disease",
                        value=disease.upper(),
                        delta=f"{CLASS_LABELS.get(disease, disease)}",
                    )
                    st.metric(label="Confidence Score", value=f"{result['confidence']:.1%}")

                    render_risk_badge(result["risk_level"])

                    st.subheader("Class Probabilities")
                    prob_data = result.get("probabilities", {})
                    for cls, prob in sorted(prob_data.items(), key=lambda x: -x[1]):
                        st.progress(prob, text=f"{cls.upper()}: {prob:.1%}")

                    if result.get("gradcam_path") and Path(result["gradcam_path"]).exists():
                        st.subheader("Grad-CAM Explainability")
                        st.image(result["gradcam_path"], caption="Grad-CAM Heatmap Overlay", use_container_width=True)

                except FileNotFoundError:
                    st.error(
                        "No trained model found. Please train the model first:\n"
                        "`python train.py` or `python main.py train`"
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    st.markdown("---")
    st.subheader("Pipeline Architecture")
    st.code(
        """
Input Image
    ↓
Hair Removal (Morphological Operations)
    ↓
CLAHE Image Enhancement
    ↓
Noise Removal & Normalization
    ↓
U-Net Lesion Segmentation
    ↓
EfficientNetV2 Feature Extraction
    ↓
Clinical Metadata Fusion (Age, Gender, Location)
    ↓
CBAM / SE Attention Layer
    ↓
Classification (7 HAM10000 Classes)
    ↓
Grad-CAM Explainability
    ↓
Prediction + Risk Assessment
        """,
        language="text",
    )


if __name__ == "__main__":
    main()
