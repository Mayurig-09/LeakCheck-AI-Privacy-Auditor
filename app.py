import time
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="LeakCheck | AI Privacy Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------- DESIGN ---------------------------------------
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #080d1a; color: #edf2ff; }
[data-testid="stHeader"] { background: rgba(8,13,26,.92); }
[data-testid="stSidebar"] { background: #101a2d; border-right: 1px solid #233653; }
[data-testid="stSidebar"] * { color: #e7efff !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #9eb0ce !important; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
h1,h2,h3 { color: #f4f7ff; letter-spacing: -.025em; }
p,li { color: #bdc9df; }
.hero {
  background: linear-gradient(115deg, #14294a 0%, #20234b 58%, #30204e 100%);
  border: 1px solid #344c78; border-radius: 20px;
  padding: 28px 32px; margin: 4px 0 22px 0;
  box-shadow: 0 14px 40px rgba(0,0,0,.18);
}
.hero-kicker { color:#9fc4ff; text-transform:uppercase; letter-spacing:.14em;
  font-size:.74rem; font-weight:700; }
.hero h1 { font-size:2.35rem; margin:.35rem 0 .25rem 0; }
.hero p { color:#c8d6f1; font-size:1.02rem; margin:0; }
.section-label { color:#8fb8ff; text-transform:uppercase; letter-spacing:.12em;
  font-size:.72rem; font-weight:700; }
div[data-testid="stMetric"] {
  background: linear-gradient(145deg,#121f35,#101a2c);
  border:1px solid #293d5e; padding:16px; border-radius:15px;
}
div[data-testid="stMetric"] label { color:#9fb2d0 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#f3f6ff; }
div[data-testid="stMetricDelta"] { font-size:.82rem; }
div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {
  border-radius:10px; border:1px solid #477bd0; font-weight:650;
  min-height:2.7rem;
}
div[data-testid="stButton"] > button[kind="primary"] {
  background:linear-gradient(100deg,#2868d7,#6545c7); color:white; border:0;
}
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stDownloadButton"] > button {
  background:#14233b; color:#eaf1ff;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  background:#101a2b; border-color:#263a58 !important; border-radius:15px;
}
[data-testid="stDataFrame"] { border:1px solid #293d5e; border-radius:12px; overflow:hidden; }
hr { border-color:#263650; }
div[data-testid="stRadio"] label { padding:5px 0; }
.small-muted { color:#91a4c3; font-size:.84rem; }
.status-pill { display:inline-block; padding:5px 10px; border-radius:30px;
  background:#17365a; border:1px solid #315b8d; color:#b9d7ff;
  font-size:.75rem; font-weight:650; margin-right:6px; }
.info-panel { background:#101d31; border:1px solid #293e60; border-radius:13px;
  padding:15px 17px; color:#c3d0e6; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- AUDIT ENGINE -----------------------------------
def make_synthetic_data(n_samples, n_features, random_state):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, min(n_features - 1, int(n_features * .65))),
        n_redundant=max(0, min(2, n_features - 2)),
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[.5, .5],
        flip_y=.02,
        class_sep=1.0,
        random_state=random_state,
    )
    return X, y


def prediction_features(model, X, y):
