import base64
import io
import os
import time

import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
          "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
          "Emphysema", "Fibrosis", "Pleural thickening", "Hernia"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "effb0_fp16.onnx")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

INK, SLATE, TEAL = "#0F1B2D", "#55657A", "#0E7C86"
HIGH, MOD, LOW = "#B3361B", "#B7791F", "#7C97A6"

st.set_page_config(page_title="Chest X-ray CPU Classifier", page_icon="🫁", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp { font-family: 'IBM Plex Sans', sans-serif; color: #0F1B2D; }
.stApp { background: #F6F8FA; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }

.hero h1 { font-family: 'Source Serif 4', serif; font-weight: 600; font-size: 2.6rem;
  line-height: 1.15; letter-spacing: -0.01em; margin: 0 0 .6rem 0; color: #0F1B2D; }
.hero p { font-size: 1.08rem; color: #55657A; max-width: 640px; line-height: 1.6; margin: 0 0 1.4rem 0; }

.stats { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 1.6rem; }
.stat { background: #fff; border: 1px solid #DDE3EA; border-radius: 10px; padding: 14px 18px; min-width: 190px; flex: 1; }
.stat .v { font-weight: 600; font-size: 1.05rem; }
.stat .k { font-size: .82rem; color: #55657A; margin-top: 2px; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #DDE3EA; }
.stTabs [data-baseweb="tab"] { height: 44px; padding: 0 16px; font-weight: 500; color: #55657A; }
.stTabs [aria-selected="true"] { color: #0E7C86 !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #0E7C86; }

[data-testid="stFileUploader"] section { background: #fff; border: 1.5px dashed #9FB3C2; border-radius: 10px; padding: 22px; }
[data-testid="stFileUploader"] section:hover { border-color: #0E7C86; }
.stButton button, .stDownloadButton button { border-radius: 8px; border: 1px solid #0E7C86; color: #0E7C86; background: #fff; font-weight: 500; }
.stButton button:hover, .stDownloadButton button:hover { background: #0E7C86; color: #fff; border-color: #0E7C86; }

.viewer { background: #0B1420; border-radius: 10px; padding: 14px; position: relative; }
.viewer img { width: 100%; border-radius: 6px; display: block; }
.viewer .tag { position: absolute; top: 22px; left: 24px; font-size: .75rem; color: #C9D6E2;
  background: rgba(11,20,32,.72); padding: 3px 8px; border-radius: 4px; }
.vmeta { font-size: .82rem; color: #55657A; margin-top: 8px; }

.panel { background: #fff; border: 1px solid #DDE3EA; border-radius: 10px; padding: 20px 22px; }
.panel h3 { font-family: 'Source Serif 4', serif; font-size: 1.25rem; margin: 0 0 4px 0; font-weight: 600; }
.panel .sub { color: #55657A; font-size: .9rem; margin-bottom: 14px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0 16px 0; }
.chip { background: #E3F2F3; color: #0A5961; border-radius: 999px; padding: 4px 12px; font-size: .85rem; font-weight: 500; }
.chip.none { background: #EEF1F4; color: #55657A; }

.row { display: grid; grid-template-columns: 150px 1fr 48px; align-items: center; gap: 12px; padding: 6px 0; }
.rname { font-size: .92rem; }
.rname.top { font-weight: 600; }
.track { background: #EDF1F5; height: 10px; border-radius: 5px; position: relative; }
.fill { height: 10px; border-radius: 5px; }
.mark { position: absolute; top: -4px; width: 2px; height: 18px; background: #0F1B2D; opacity: .55; }
.rval { font-variant-numeric: tabular-nums; font-size: .9rem; text-align: right; color: #0F1B2D; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: .8rem; color: #55657A; margin-top: 14px; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }

.empty { background: #fff; border: 1px solid #DDE3EA; border-radius: 10px; padding: 28px; margin-top: 14px; }
.empty h3 { font-family: 'Source Serif 4', serif; margin: 0 0 6px 0; font-weight: 600; }
.empty p { color: #55657A; margin: 0; line-height: 1.6; }

.step { display: flex; gap: 14px; padding: 14px 0; border-bottom: 1px solid #E6EBF0; }
.step:last-child { border-bottom: none; }
.step .n { flex: 0 0 30px; height: 30px; border-radius: 50%; background: #E3F2F3; color: #0A5961; font-weight: 600;
  display: flex; align-items: center; justify-content: center; font-size: .9rem; }
.step .t { font-weight: 600; }
.step .d { color: #55657A; font-size: .92rem; margin-top: 2px; line-height: 1.5; }

.notice { background: #FFF6E5; border: 1px solid #F0D9A8; color: #6B4A08; border-radius: 8px;
  padding: 10px 14px; font-size: .86rem; margin-bottom: 18px; line-height: 1.5; }
.foot { color: #7A8896; font-size: .8rem; margin-top: 2.2rem; text-align: center; }

@media (max-width: 720px) {
  .hero h1 { font-size: 1.9rem; }
  .row { grid-template-columns: 110px 1fr 40px; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(s: str):
    st.markdown("\n".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


@st.cache_resource
def load_session():
    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


@st.cache_data(show_spinner=False)
def run_inference(file_bytes: bytes):
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes)))
    im = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    x = ((x - MEAN) / STD)[None].astype(np.float32)
    sess = load_session()
    t = time.perf_counter()
    logits = sess.run(None, {"input": x})[0][0]
    ms = (time.perf_counter() - t) * 1000
    scores = 1 / (1 + np.exp(-logits))
    return scores.tolist(), ms, img.size


def to_b64(file_bytes: bytes) -> str:
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(file_bytes))).convert("RGB")
    im.thumbnail((900, 900))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def colour(score: float) -> str:
    return HIGH if score >= 0.5 else MOD if score >= 0.25 else LOW


model_mb = os.path.getsize(MODEL_PATH) / 1e6 if os.path.exists(MODEL_PATH) else None

html(f"""
<div class="hero">
<h1>Chest X-ray findings, on a CPU.</h1>
<p>Upload a frontal chest radiograph and get ranked scores for 14 thoracic findings in milliseconds, with no GPU required.</p>
</div>
<div class="stats">
<div class="stat"><div class="v">EfficientNet-B0</div><div class="k">Model architecture</div></div>
<div class="stat"><div class="v">ONNX FP16{f' · {model_mb:.1f} MB' if model_mb else ''}</div><div class="k">Runtime format, CPU only</div></div>
<div class="stat"><div class="v">NIH ChestX-ray14</div><div class="k">Training dataset, 14 labels</div></div>
</div>
""")

tab_run, tab_how, tab_about = st.tabs(["Analyze an X-ray", "How it works", "About this project"])

with tab_run:
    html("""
    <div class="notice">Research prototype. This is not a medical device and must not be used for diagnosis.
    Please do not upload real patient images.</div>
    """)
    file = st.file_uploader("Upload a frontal chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"],
                            label_visibility="collapsed")

    if file is None:
        html("""
        <div class="empty">
        <h3>No image yet</h3>
        <p>Drop a PNG or JPG above. The model scores every finding at once, and you can adjust the flag
        threshold to see how the ranking changes.</p>
        </div>
        """)
    else:
        data = file.getvalue()
        try:
            scores, ms, (w, h) = run_inference(data)
        except Exception:
            st.error("This file could not be read as an image. Upload a PNG or JPG chest X-ray.")
            st.stop()

        scores = np.array(scores)
        order = np.argsort(scores)[::-1]

        left, right = st.columns([5, 6], gap="large")

        with left:
            html(f"""
            <div class="viewer"><span class="tag">{file.name}</span>
            <img src="data:image/jpeg;base64,{to_b64(data)}" alt="Uploaded chest X-ray"/></div>
            <div class="vmeta">{w} × {h} px · resized to 224 × 224 for the model · inference {ms:.1f} ms on CPU</div>
            """)

        with right:
            thr = st.slider("Flag findings scoring at or above", 0.05, 0.95, 0.30, 0.05)
            flagged = [LABELS[i] for i in order if scores[i] >= thr]
            chips = ("".join(f'<span class="chip">{n}</span>' for n in flagged)
                     if flagged else '<span class="chip none">No finding at or above this threshold</span>')

            rows = ""
            for rank, i in enumerate(order):
                s = float(scores[i])
                rows += (f'<div class="row"><div class="rname{" top" if rank < 3 else ""}">{LABELS[i]}</div>'
                         f'<div class="track"><div class="fill" style="width:{s*100:.1f}%;background:{colour(s)}"></div>'
                         f'<div class="mark" style="left:{thr*100:.0f}%"></div></div>'
                         f'<div class="rval">{s:.2f}</div></div>')

            html(f"""
            <div class="panel">
            <h3>Model findings</h3>
            <div class="sub">{len(flagged)} of 14 findings at or above {thr:.2f}</div>
            <div class="chips">{chips}</div>
            {rows}
            <div class="legend">
            <span><span class="dot" style="background:{HIGH}"></span>0.50 and above</span>
            <span><span class="dot" style="background:{MOD}"></span>0.25 to 0.49</span>
            <span><span class="dot" style="background:{LOW}"></span>below 0.25</span>
            <span>The dark tick marks your threshold</span>
            </div>
            </div>
            """)
            st.caption("Scores are relative model outputs, not calibrated probabilities.")

            df = pd.DataFrame({"Finding": [LABELS[i] for i in order],
                               "Score": [round(float(scores[i]), 4) for i in order]})
            st.download_button("Download scores as CSV", df.to_csv(index=False),
                               file_name="xray_scores.csv", mime="text/csv")

with tab_how:
    html("""
    <div class="panel">
    <h3>From upload to ranked scores</h3>
    <div class="sub">The whole pipeline runs on the server's CPU, with no GPU and no external API.</div>
    <div class="step"><div class="n">1</div><div><div class="t">Read and prepare the image</div>
    <div class="d">The upload is converted to RGB and resized to 224 × 224 pixels.</div></div></div>
    <div class="step"><div class="n">2</div><div><div class="t">Normalize</div>
    <div class="d">Pixel values are scaled to 0–1, then standardized with ImageNet mean and standard deviation.</div></div></div>
    <div class="step"><div class="n">3</div><div><div class="t">Run the ONNX model</div>
    <div class="d">An FP16 EfficientNet-B0 runs through ONNX Runtime's CPU provider. The session is loaded once and reused.</div></div></div>
    <div class="step"><div class="n">4</div><div><div class="t">Convert logits to scores</div>
    <div class="d">A sigmoid is applied to each of the 14 outputs, because a single X-ray can show several findings at once.</div></div></div>
    <div class="step"><div class="n">5</div><div><div class="t">Rank and display</div>
    <div class="d">Findings are sorted by score and shown against the threshold you choose.</div></div></div>
    </div>
    """)

with tab_about:
    html("""
    <div class="panel">
    <h3>About this project</h3>
    <div class="sub">A study in making chest X-ray classification practical on ordinary hardware.</div>
    <p style="line-height:1.7">The model was trained on the NIH ChestX-ray14 dataset to score 14 thoracic findings:
    atelectasis, cardiomegaly, effusion, infiltration, mass, nodule, pneumonia, pneumothorax, consolidation, edema,
    emphysema, fibrosis, pleural thickening and hernia. It was exported to ONNX in FP16 to keep file size, memory use
    and latency low enough for CPU-only deployment.</p>
    <p style="line-height:1.7">This is a multi-label task, so the scores are independent and do not sum to one.
    They rank how strongly the model responds to each finding and are not calibrated probabilities.</p>
    </div>
    <div class="foot">Research prototype · not a medical device · not for diagnosis</div>
    """)
