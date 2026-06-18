import os
import time
import re
import pickle
import urllib.request

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="EmotiSense · Emotion Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# EMOTION CONFIG
# ─────────────────────────────────────────
EMOTION_CONFIG = {
    "anger":    {"emoji": "😡", "color": "#ef4444", "bg": "#fef2f2", "label": "Anger"},
    "fear":     {"emoji": "😨", "color": "#8b5cf6", "bg": "#f5f3ff", "label": "Fear"},
    "joy":      {"emoji": "😊", "color": "#f59e0b", "bg": "#fffbeb", "label": "Joy"},
    "love":     {"emoji": "❤️", "color": "#ec4899", "bg": "#fdf2f8", "label": "Love"},
    "sadness":  {"emoji": "😢", "color": "#3b82f6", "bg": "#eff6ff", "label": "Sadness"},
    "surprise": {"emoji": "😲", "color": "#10b981", "bg": "#ecfdf5", "label": "Surprise"},
}

EXAMPLES = [
    "I feel so happy and grateful today!",
    "This makes me so angry, I can't believe it.",
    "I'm terrified of what might happen next.",
    "I love spending time with my family.",
    "I feel so sad and lonely right now.",
    "Wow, I never expected that to happen!",
]

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* ── Root & Reset ── */
:root {
    --bg:       #f8f7f4;
    --surface:  #ffffff;
    --border:   #e8e4df;
    --text:     #1a1814;
    --muted:    #78716c;
    --accent:   #d97706;
    --radius:   16px;
    --shadow:   0 2px 16px rgba(0,0,0,.07);
}

html, body, .stApp {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 4rem !important; max-width: 1200px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}

/* ── Page title ── */
.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    font-weight: 400;
    color: var(--text);
    line-height: 1.1;
    margin-bottom: .25rem;
}
.page-sub {
    color: var(--muted);
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* ── Input box override ── */
.stTextArea textarea {
    border-radius: 12px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem !important;
    color: var(--text) !important;
    transition: border-color .2s;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(217,119,6,.12) !important;
}

/* ── Predict button ── */
.stButton > button {
    background: var(--text) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .7rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: .01em !important;
    cursor: pointer !important;
    transition: opacity .2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── Result badge ── */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: .6rem;
    padding: .6rem 1.2rem;
    border-radius: 100px;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

/* ── Confidence bar ── */
.conf-bar-wrap { margin: .4rem 0 1rem; }
.conf-bar-bg {
    background: var(--border);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width .6s cubic-bezier(.4,0,.2,1);
}

/* ── History item ── */
.hist-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: .65rem .9rem;
    border-radius: 10px;
    border: 1px solid var(--border);
    margin-bottom: .5rem;
    background: var(--surface);
    font-size: .9rem;
}
.hist-text { color: var(--text); flex: 1; margin-right: 1rem; white-space: nowrap;
             overflow: hidden; text-overflow: ellipsis; max-width: 200px; }
.hist-badge {
    display: inline-flex; align-items: center; gap: .3rem;
    padding: .25rem .7rem; border-radius: 99px; font-size: .8rem; font-weight: 600;
}

/* ── Example pills ── */
.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important;
}

/* ── Section label ── */
.sec-label {
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .75rem;
}

/* ── Model info ── */
.info-row {
    display: flex;
    justify-content: space-between;
    padding: .5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: .88rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: var(--muted); }
.info-val { font-weight: 600; color: var(--text); }

/* ── Stat box ── */
.stat-box {
    text-align: center;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: var(--surface);
}
.stat-num { font-family: 'DM Serif Display', serif; font-size: 1.8rem; }
.stat-lbl { font-size: .78rem; color: var(--muted); margin-top: .15rem; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ─────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────
# Small artifacts (tokenizer, label map, max length) ship with the repo.
# The trained model file is NOT committed to git (it's a binary build
# artifact, not source code) — it's attached to a GitHub Release on this
# same repo and downloaded once on first run, then cached by Streamlit.
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
MODEL_FILENAME = "best_model.keras"
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/nourannasser2210/emotion-classification-nlp/releases/download/v1.0.0/best_model.keras",
)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    nltk.download("stopwords", quiet=True)

    # Local override for development: drop best_model.keras into artifacts/
    # and it will be used instead of downloading from the Release.
    local_model_path = os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)
    if os.path.exists(local_model_path):
        model_path = local_model_path
    else:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "emotisense")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, MODEL_FILENAME)
        if not os.path.exists(model_path):
            urllib.request.urlretrieve(MODEL_URL, model_path)

    model = load_model(model_path)

    with open(os.path.join(ARTIFACTS_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "label_map.pkl"), "rb") as f:
        label_map = pickle.load(f)
    max_len = int(np.load(os.path.join(ARTIFACTS_DIR, "max_length.npy"))[0])

    return model, tokenizer, label_map, max_len


try:
    model, tokenizer, label_map, MAX_LEN = load_artifacts()
    loaded = True
except Exception as e:
    loaded = False
    load_error = str(e)


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
def clean_text(text):
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def predict_emotion(text, model, tokenizer, label_map, max_len):
    cleaned   = clean_text(text)
    seq       = tokenizer.texts_to_sequences([cleaned])
    padded    = pad_sequences(seq, maxlen=max_len, padding="post")
    probs     = model.predict(padded, verbose=0)[0]
    pred_idx  = int(np.argmax(probs))
    emotion   = label_map[pred_idx]
    scores    = {label_map[i]: float(probs[i]) for i in range(len(probs))}
    return emotion, scores


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="page-title" style="font-size:1.6rem">EmotiSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub" style="font-size:.85rem;margin-bottom:1.5rem">Emotion Intelligence</div>', unsafe_allow_html=True)

    # ── Model Info ──
    st.markdown('<div class="sec-label">Model Info</div>', unsafe_allow_html=True)
    info_rows = [
        ("Architecture",  "BiLSTM × 2"),
        ("Accuracy",      "91%"),
        ("Classes",       "6 Emotions"),
        ("Dataset",       "16,000 samples"),
        ("Vocab Size",    "10,000 words"),
        ("Framework",     "TensorFlow / Keras"),
    ]
    info_html = '<div class="card" style="padding:1rem">'
    for k, v in info_rows:
        info_html += f'<div class="info-row"><span class="info-key">{k}</span><span class="info-val">{v}</span></div>'
    info_html += "</div>"
    st.markdown(info_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Emotion Legend ──
    st.markdown('<div class="sec-label">Emotion Legend</div>', unsafe_allow_html=True)
    for em, cfg in EMOTION_CONFIG.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.6rem;padding:.3rem 0;">'
            f'<span style="font-size:1.2rem">{cfg["emoji"]}</span>'
            f'<span style="font-weight:500">{cfg["label"]}</span>'
            f'<span style="margin-left:auto;width:10px;height:10px;border-radius:50%;'
            f'background:{cfg["color"]}"></span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Stats ──
    st.markdown('<div class="sec-label">Session Stats</div>', unsafe_allow_html=True)
    total = len(st.session_state.history)
    if total:
        top_em = max(set(h["emotion"] for h in st.session_state.history),
                     key=lambda e: sum(1 for h in st.session_state.history if h["emotion"] == e))
        top_cfg = EMOTION_CONFIG[top_em]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{total}</div>'
                        f'<div class="stat-lbl">Predictions</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{top_cfg["emoji"]}</div>'
                        f'<div class="stat-lbl">Top Emotion</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#78716c;font-size:.85rem">No predictions yet.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
st.markdown('<div class="page-title">Emotion Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Understand the feeling behind any sentence — instantly.</div>', unsafe_allow_html=True)

if not loaded:
    st.error(f"⚠️ Could not load model artifacts.\n\n`{load_error}`")
    st.stop()

col_main, col_hist = st.columns([3, 2], gap="large")

# ─── LEFT: Input + Result ───────────────
with col_main:

    # Example selector
    st.markdown('<div class="sec-label">Try an example</div>', unsafe_allow_html=True)
    chosen_example = st.selectbox(
        label="", options=["— choose an example —"] + EXAMPLES,
        label_visibility="collapsed"
    )
    if chosen_example != "— choose an example —":
        st.session_state.input_text = chosen_example

    st.markdown('<div class="sec-label" style="margin-top:1rem">Your sentence</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        label="",
        value=st.session_state.input_text,
        height=130,
        placeholder="Type your sentence here...",
        label_visibility="collapsed",
        key="textarea"
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        predict_clicked = st.button("🔍  Predict Emotion", use_container_width=True)
    with c2:
        clear_clicked = st.button("✕  Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.input_text  = ""
        st.session_state.prediction  = None
        st.rerun()

    # ── Prediction ──
    if predict_clicked:
        if not user_input.strip():
            st.warning("⚠️  Please enter a sentence first.")
        else:
            with st.spinner("Analyzing emotion…"):
                time.sleep(0.4)
                emotion, scores = predict_emotion(user_input, model, tokenizer, label_map, MAX_LEN)
            st.session_state.prediction = {"text": user_input, "emotion": emotion, "scores": scores}
            # Add to history (avoid duplicates)
            if not st.session_state.history or st.session_state.history[-1]["text"] != user_input:
                st.session_state.history.insert(0, {"text": user_input, "emotion": emotion, "scores": scores})
                st.session_state.history = st.session_state.history[:10]

    # ── Result Display ──
    if st.session_state.prediction:
        pred   = st.session_state.prediction
        em     = pred["emotion"]
        cfg    = EMOTION_CONFIG[em]
        conf   = pred["scores"][em]

        st.markdown("---")
        st.markdown('<div class="sec-label">Result</div>', unsafe_allow_html=True)

        # Badge
        st.markdown(
            f'<div class="result-badge" style="background:{cfg["bg"]};color:{cfg["color"]}">'
            f'{cfg["emoji"]} &nbsp; {cfg["label"]}</div>',
            unsafe_allow_html=True
        )

        # Confidence bar
        pct = int(conf * 100)
        st.markdown(
            f'<div style="font-size:.85rem;color:#78716c;margin-bottom:.3rem">Confidence: <b style="color:{cfg["color"]}">{pct}%</b></div>'
            f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%;background:{cfg["color"]}"></div></div>',
            unsafe_allow_html=True
        )

        st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)

        # ── Bar Chart ──
        st.markdown('<div class="sec-label">All Emotion Scores</div>', unsafe_allow_html=True)
        emotions_sorted = sorted(pred["scores"].items(), key=lambda x: x[1], reverse=True)
        labels  = [f'{EMOTION_CONFIG[e]["emoji"]} {EMOTION_CONFIG[e]["label"]}' for e, _ in emotions_sorted]
        values  = [round(v * 100, 1) for _, v in emotions_sorted]
        colors  = [EMOTION_CONFIG[e]["color"] for e, _ in emotions_sorted]

        fig = go.Figure(go.Bar(
            x=values, y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v}%" for v in values],
            textposition="outside",
            textfont=dict(family="DM Sans", size=12),
        ))
        fig.update_layout(
            margin=dict(l=0, r=40, t=0, b=0),
            height=240,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False, range=[0, max(values) * 1.25]),
            yaxis=dict(showgrid=False, tickfont=dict(family="DM Sans", size=13)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─── RIGHT: History ────────────────────
with col_hist:
    st.markdown('<div class="sec-label">Prediction History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown(
            '<div class="card" style="text-align:center;color:#78716c;padding:2rem">'
            '📭<br><br>No history yet.<br>Make your first prediction!</div>',
            unsafe_allow_html=True
        )
    else:
        if st.button("🗑  Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        st.markdown('<div style="margin-top:.75rem"></div>', unsafe_allow_html=True)

        for item in st.session_state.history:
            em  = item["emotion"]
            cfg = EMOTION_CONFIG[em]
            conf_pct = int(item["scores"][em] * 100)
            short_text = item["text"][:45] + ("…" if len(item["text"]) > 45 else "")

            st.markdown(
                f'<div class="hist-item">'
                f'<span class="hist-text" title="{item["text"]}">{short_text}</span>'
                f'<span class="hist-badge" style="background:{cfg["bg"]};color:{cfg["color"]}">'
                f'{cfg["emoji"]} {cfg["label"]} · {conf_pct}%</span>'
                f'</div>',
                unsafe_allow_html=True
            )
