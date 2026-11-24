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

NAVY, DEEP, MID, SKY = "#0B2447", "#19376D", "#2F63A8", "#8DB4E0"

SENSITIVITY = {
    "Strict": (0.50, "Flags only findings the model is most confident about. Fewer flags, but subtler findings can be missed."),
    "Balanced": (0.30, "A middle ground between missing findings and flagging too many."),
    "Sensitive": (0.15, "Flags anything the model responds to, even weakly. More flags, with more false alarms."),
}

st.set_page_config(page_title="Chest X-ray CPU Classifier", page_icon="🫁", layout="wide")

CSS = """
<style>
.stApp, .stApp *:not([data-testid="stIconMaterial"]):not([class*="material"]) {
  font-family: 'Times New Roman', Times, serif !important;
}
.stApp { background: #E6EEF8; color: #0B2447; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 1.6rem; padding-bottom: 3rem; }

.mast { background: #0B2447; color: #fff; padding: 44px 48px 0 48px; border-radius: 2px; margin-bottom: 26px; }
.mast h1 { font-size: 3rem; font-weight: 700; line-height: 1.1; margin: 0 0 18px 0; color: #fff; letter-spacing: -0.5px; }
.mast p { font-size: 1.22rem; line-height: 1.65; color: #CFE0F5; max-width: 780px; margin: 0 0 34px 0; }
.facts { display: flex; border-top: 1px solid #2F4F82; margin: 0 -48px; }
.fact { flex: 1; padding: 18px 48px; border-right: 1px solid #2F4F82; }
.fact:last-child { border-right: none; }
.fact .v { font-size: 1.2rem; font-weight: 700; color: #fff; }
.fact .k { font-size: .95rem; color: #9DB9DD; margin-top: 2px; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #BBD3EE; }
.stTabs [data-baseweb="tab"] { height: 46px; padding: 0 18px; color: #3B5B8C; font-size: 1.1rem; }
.stTabs [aria-selected="true"] { color: #0B2447 !important; font-weight: 700; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #19376D; height: 3px; }

.lead { font-size: 1.12rem; color: #2A4670; margin: 6px 0 12px 0; }
[data-testid="stFileUploader"] section { background: #F5F9FE; border: 2px dashed #6FA0D8; border-radius: 2px; padding: 24px; }
[data-testid="stFileUploader"] section:hover { border-color: #19376D; }
.stButton button, .stDownloadButton button { border-radius: 2px; background: #19376D; color: #fff; border: 1px solid #19376D; font-size: 1.05rem; }
.stButton button:hover, .stDownloadButton button:hover { background: #0B2447; color: #fff; border-color: #0B2447; }
.stButton button p, .stDownloadButton button p { color: #fff; }

.viewer { background: #071A33; padding: 14px; position: relative; border-radius: 2px; }
.viewer img { width: 100%; display: block; }
.viewer .tag { position: absolute; top: 22px; left: 24px; font-size: .9rem; color: #D6E4F7; background: rgba(7,26,51,.75); padding: 3px 9px; }
.vmeta { font-size: 1rem; color: #3B5B8C; margin-top: 8px; line-height: 1.5; }

.report { background: #F5F9FE; border: 1px solid #BBD3EE; border-top: 4px solid #19376D; padding: 22px 26px; border-radius: 2px; }
.report h3 { font-size: 1.7rem; margin: 0 0 4px 0; color: #0B2447; font-weight: 700; }
.report .sub { color: #3B5B8C; font-size: 1.05rem; margin-bottom: 14px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 18px 0; }
.chip { background: #19376D; color: #fff; padding: 4px 14px; font-size: 1.02rem; border-radius: 2px; }
.chip.none { background: #DCE8F6; color: #3B5B8C; }

.row { display: grid; grid-template-columns: 170px 1fr 52px; align-items: center; gap: 14px; padding: 6px 0; border-bottom: 1px solid #E1EBF7; }
.rname { font-size: 1.08rem; }
.rname.top { font-weight: 700; }
.track { background: #D5E3F4; height: 11px; position: relative; }
.fill { height: 11px; }
.mark { position: absolute; top: -5px; width: 2px; height: 21px; background: #0B2447; }
.rval { font-size: 1.08rem; text-align: right; font-variant-numeric: tabular-nums; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: .98rem; color: #3B5B8C; margin-top: 16px; }
.sw { display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: -1px; }
.note { font-size: .98rem; color: #3B5B8C; margin-top: 10px; line-height: 1.5; }

.sens-label { font-size: 1.2rem; font-weight: 700; margin: 2px 0 0 0; }
.sens-help { font-size: 1.02rem; color: #2A4670; margin: 2px 0 14px 0; line-height: 1.5; }

.empty { background: #F5F9FE; border: 1px solid #BBD3EE; border-left: 4px solid #19376D; padding: 22px 26px; margin-top: 14px; }
.empty h3 { margin: 0 0 6px 0; font-size: 1.4rem; }
.empty p { margin: 0; font-size: 1.1rem; line-height: 1.6; color: #2A4670; }

.step { display: flex; gap: 16px; padding: 15px 0; border-bottom: 1px solid #D5E3F4; }
.step:last-child { border-bottom: none; }
.step .n { flex: 0 0 34px; height: 34px; background: #19376D; color: #fff; font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: 1.05rem; }
.step .t { font-weight: 700; font-size: 1.15rem; }
.step .d { color: #2A4670; font-size: 1.05rem; margin-top: 2px; line-height: 1.55; }
.about p { font-size: 1.12rem; line-height: 1.75; color: #1D3A63; }

.foot { color: #4B6A98; font-size: .95rem; margin-top: 2.4rem; text-align: center; }

@media (max-width: 720px) {
  .mast { padding: 28px 22px 0 22px; }
  .mast h1 { font-size: 2.1rem; }
  .facts { flex-direction: column; margin: 0 -22px; }
  .fact { padding: 14px 22px; border-right: none; border-bottom: 1px solid #2F4F82; }
  .row { grid-template-columns: 120px 1fr 44px; }
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


def shade(score: float) -> str:
    return DEEP if score >= 0.5 else MID if score >= 0.25 else SKY


model_mb = os.path.getsize(MODEL_PATH) / 1e6 if os.path.exists(MODEL_PATH) else None
runtime_line = f"ONNX FP16, {model_mb:.1f} MB" if model_mb else "ONNX FP16"

html(f"""
<div class="mast">
<h1>Chest X-ray screening, without the GPU.</h1>
<p>Upload a frontal chest radiograph and the model reads it in a fraction of a second, scoring 14 thoracic
findings, from pneumonia and effusion to nodules and cardiomegaly, and ranking them by how strongly the image
matches each pattern. It is built on a compact EfficientNet-B0 and exported to ONNX FP16, so the whole analysis
runs on an ordinary CPU and does not depend on specialist hardware.</p>
<div class="facts">
<div class="fact"><div class="v">EfficientNet-B0</div><div class="k">Model architecture</div></div>
<div class="fact"><div class="v">{runtime_line}</div><div class="k">CPU-only runtime</div></div>
<div class="fact"><div class="v">NIH ChestX-ray14</div><div class="k">Training data, 14 finding labels</div></div>
</div>
</div>
""")

tab_run, tab_how, tab_about = st.tabs(["Analyze an X-ray", "How it works", "About this project"])

with tab_run:
    html('<div class="lead">Choose a frontal chest X-ray in PNG or JPG format.</div>')
    file = st.file_uploader("Upload a frontal chest X-ray", type=["png", "jpg", "jpeg"],
                            label_visibility="collapsed")

    if file is None:
        html("""
        <div class="empty">
        <h3>Waiting for an image</h3>
        <p>Once you upload an X-ray, every finding is scored at once and ranked from most to least likely.
        You can then choose how sensitive the flagging should be.</p>
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
            <div class="vmeta">Original size {w} × {h} pixels, resized to 224 × 224 for the model.<br>
            Analysed on CPU in {ms:.1f} ms.</div>
            """)

        with right:
            html('<div class="sens-label">Sensitivity</div>')
            level = st.radio("Sensitivity", list(SENSITIVITY), index=1, horizontal=True,
                             label_visibility="collapsed")
            thr, help_text = SENSITIVITY[level]
            html(f'<div class="sens-help">{help_text} Findings scoring {thr:.2f} or higher are flagged.</div>')

            flagged = [LABELS[i] for i in order if scores[i] >= thr]
            chips = ("".join(f'<span class="chip">{n}</span>' for n in flagged)
                     if flagged else '<span class="chip none">No finding reaches this level</span>')

            rows = ""
            for rank, i in enumerate(order):
                s = float(scores[i])
                rows += (f'<div class="row"><div class="rname{" top" if rank < 3 else ""}">{LABELS[i]}</div>'
                         f'<div class="track"><div class="fill" style="width:{s*100:.1f}%;background:{shade(s)}"></div>'
                         f'<div class="mark" style="left:{thr*100:.0f}%"></div></div>'
                         f'<div class="rval">{s:.2f}</div></div>')

            html(f"""
            <div class="report">
            <h3>Findings report</h3>
            <div class="sub">{len(flagged)} of 14 findings flagged at {level.lower()} sensitivity</div>
            <div class="chips">{chips}</div>
            {rows}
            <div class="legend">
            <span><span class="sw" style="background:{DEEP}"></span>0.50 or higher</span>
            <span><span class="sw" style="background:{MID}"></span>0.25 to 0.49</span>
            <span><span class="sw" style="background:{SKY}"></span>below 0.25</span>
            <span><span class="sw" style="background:{NAVY};width:3px"></span>flagging line</span>
            </div>
            <div class="note">Scores are relative model outputs, not calibrated probabilities.</div>
            </div>
            """)

            df = pd.DataFrame({"Finding": [LABELS[i] for i in order],
                               "Score": [round(float(scores[i]), 4) for i in order]})
            st.download_button("Download scores as CSV", df.to_csv(index=False),
                               file_name="xray_scores.csv", mime="text/csv")

with tab_how:
    html("""
    <div class="report">
    <h3>From upload to ranked scores</h3>
    <div class="sub">The entire pipeline runs on the server's CPU, with no GPU and no external API.</div>
    <div class="step"><div class="n">1</div><div><div class="t">Read and prepare the image</div>
    <div class="d">The upload is converted to RGB and resized to 224 × 224 pixels.</div></div></div>
    <div class="step"><div class="n">2</div><div><div class="t">Normalize</div>
    <div class="d">Pixel values are scaled to the 0 to 1 range, then standardized with the ImageNet mean and standard deviation.</div></div></div>
    <div class="step"><div class="n">3</div><div><div class="t">Run the ONNX model</div>
    <div class="d">An FP16 EfficientNet-B0 runs through ONNX Runtime's CPU provider. The session loads once and is reused.</div></div></div>
    <div class="step"><div class="n">4</div><div><div class="t">Convert logits to scores</div>
    <div class="d">A sigmoid is applied to each of the 14 outputs, because one X-ray can show several findings at once.</div></div></div>
    <div class="step"><div class="n">5</div><div><div class="t">Rank and display</div>
    <div class="d">Findings are sorted by score and compared with the sensitivity level you choose.</div></div></div>
    </div>
    """)

with tab_about:
    html("""
    <div class="report about">
    <h3>About this project</h3>
    <div class="sub">A study in making chest X-ray classification practical on ordinary hardware.</div>
    <p>The model was trained on the NIH ChestX-ray14 dataset to score 14 thoracic findings: atelectasis,
    cardiomegaly, effusion, infiltration, mass, nodule, pneumonia, pneumothorax, consolidation, edema,
    emphysema, fibrosis, pleural thickening and hernia. It was exported to ONNX in FP16 to keep file size,
    memory use and latency low enough for CPU-only deployment.</p>
    <p>This is a multi-label task, so the scores are independent and do not add up to one. They rank how
    strongly the model responds to each finding and are not calibrated probabilities.</p>
    </div>
    """)

html('<div class="foot">For research and education only. Not a medical device and not for diagnosis.</div>')
