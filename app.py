import os, time
import numpy as np
import pandas as pd
import onnxruntime as ort
import streamlit as st
from PIL import Image

LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
          "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
          "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "effb0_fp16.onnx")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
THRESHOLD = 0.30


@st.cache_resource
def load_session():
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


def predict(img):
    im = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    x = ((x - MEAN) / STD)[None].astype(np.float32)
    sess = load_session()
    t = time.perf_counter()
    logits = sess.run(None, {"input": x})[0][0]
    ms = (time.perf_counter() - t) * 1000
    return 1 / (1 + np.exp(-logits)), ms


def bar_with_tick(score):
    return (
        '<div style="position:relative;height:8px;border-radius:4px;'
        'background:rgba(151,166,195,0.25);margin:2px 0 14px 0">'
        f'<div style="width:{score * 100:.1f}%;height:8px;border-radius:4px;background:#ff4b4b"></div>'
        f'<div style="position:absolute;left:{THRESHOLD * 100:.0f}%;top:-4px;width:2px;height:16px;'
        'background:currentColor;opacity:.6"></div></div>'
    )


st.set_page_config(page_title="Chest X-ray CPU Classifier", layout="centered")
st.title("CPU-Efficient Chest X-ray Classifier")
st.caption("EfficientNet-B0 on NIH ChestX-ray14, ONNX FP16, running on CPU only. "
           "Research prototype: not a medical device and not for diagnosis. "
           "Please do not upload real patient images.")

file = st.file_uploader("Upload a frontal chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"])

if file is not None:
    img = Image.open(file)
    left, right = st.columns(2)
    left.image(img, caption="Uploaded image")

    scores, ms = predict(img)
    order = np.argsort(scores)[::-1]

    right.subheader("Top 5 finding scores")
    for i in order[:5]:
        right.write(f"{LABELS[i]}: {scores[i]:.2f}")
        right.markdown(bar_with_tick(float(scores[i])), unsafe_allow_html=True)
    right.caption(f"The vertical tick on each bar marks the {THRESHOLD:.2f} threshold.")
    right.caption(f"Inference time on CPU: {ms:.1f} ms")

    with st.expander("All 14 finding scores"):
        st.dataframe(pd.DataFrame({"Finding": [LABELS[i] for i in order],
                                   "Score": [round(float(scores[i]), 3) for i in order]}),
                     hide_index=True)
    st.caption("Scores are relative model outputs, not calibrated probabilities.")
else:
    st.info("Upload a chest X-ray to see the model's scores.")
