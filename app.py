import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

# -----------------------------
# Model definition (must match training)
# -----------------------------
class RiceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # logits
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Styling
# -----------------------------
st.set_page_config(page_title="Rice Classifier", page_icon="🍚", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      .title {font-size: 2.0rem; font-weight: 800; margin-bottom: .25rem;}
      .subtitle {color: #6b7280; margin-top: 0; margin-bottom: 1.2rem;}
      .card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 16px;
        padding: 18px 18px 8px 18px;
        background: rgba(255, 255, 255, 0.6);
      }
      .metric-row {display: flex; gap: 14px; flex-wrap: wrap;}
      .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 0.9rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        background: rgba(249, 250, 251, 0.9);
      }
      .small {color: #6b7280; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Load artifacts (cached)
# -----------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("maxabs_scaler.joblib")
    meta = joblib.load("metadata.joblib")

    feature_cols = meta["feature_cols"]
    class_map = meta.get("class_map", {0: "Class 0", 1: "Class 1"})
    hidden_dim = int(meta.get("hidden_dim", 32))

    device = torch.device("cpu")  # portable
    model = RiceNet(input_dim=len(feature_cols), hidden_dim=hidden_dim).to(device)

    state_dict = torch.load("rice_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler, feature_cols, class_map, device

def predict_one(raw_feature_dict, model, scaler, feature_cols, device, threshold=0.5):
    raw_x = np.array([raw_feature_dict[c] for c in feature_cols], dtype=np.float32).reshape(1, -1)
    x_scaled = scaler.transform(raw_x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(x_tensor)
        prob1 = torch.sigmoid(logits).item()

    pred = int(prob1 >= threshold)
    return prob1, pred

# -----------------------------
# App header
# -----------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }
        .divider {
            border-top: 1px solid rgba(49, 51, 63, 0.2);
            margin-top: 0.5rem;
            margin-bottom: 1.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🍚 Rice Grain Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Binary classification using a trained neural network (PyTorch) + MaxAbsScaler</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# -----------------------------
# Load
# -----------------------------
try:
    model, scaler, feature_cols, class_map, device = load_artifacts()
except Exception as e:
    st.error("Failed to load model/scaler artifacts.")
    st.exception(e)
    st.stop()

# -----------------------------
# Sensible defaults (from dataset stats if available)
# If you saved dataset stats in metadata, use them. Otherwise, use 0.
# -----------------------------
defaults = {c: 0.0 for c in feature_cols}
mins = {c: None for c in feature_cols}
maxs = {c: None for c in feature_cols}

# Optional: if you saved these in metadata.joblib, app will auto-fill
# meta could include: "feature_defaults", "feature_mins", "feature_maxs"
try:
    meta = joblib.load("metadata.joblib")
    defaults.update(meta.get("feature_defaults", {}))
    mins.update(meta.get("feature_mins", {}))
    maxs.update(meta.get("feature_maxs", {}))
except Exception:
    pass

# -----------------------------
# Layout: Sidebar inputs + Main results
# -----------------------------
st.sidebar.header("Inputs")
st.sidebar.caption("Enter raw feature values (original units).")

threshold = st.sidebar.slider("Decision threshold for Class 1", 0.05, 0.95, 0.50, 0.01)

with st.sidebar.expander("Feature order used by model", expanded=False):
    st.write(feature_cols)

# Group inputs into two columns inside sidebar for compactness
raw = {}
for feat in feature_cols:
    raw[feat] = st.sidebar.number_input(
        feat,
        value=float(defaults.get(feat, 0.0)),
        min_value=mins.get(feat, None),
        max_value=maxs.get(feat, None),
        format="%.6f",
    )

predict_clicked = st.sidebar.button("Predict", use_container_width=True)

# Main area: two columns (result + explanation)
left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.markdown("### Prediction")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if not predict_clicked:
        st.info("Set input values in the sidebar and click **Predict**.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        prob1, pred = predict_one(raw, model, scaler, feature_cols, device, threshold=threshold)
        prob0 = 1.0 - prob1
        label = class_map.get(pred, str(pred))

        # Progress bar looks nice for probability
        st.write("**Probability (Class 1)**")
        st.progress(min(max(prob1, 0.0), 1.0))

        st.markdown(
            f"""
            <div class="metric-row">
              <span class="pill"><b>P(Class=1)</b>: {prob1:.4f}</span>
              <span class="pill"><b>P(Class=0)</b>: {prob0:.4f}</span>
              <span class="pill"><b>Threshold</b>: {threshold:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"#### Predicted class: **{pred} → {label}**")

        # Optional: show "confidence" as max(prob0, prob1)
        conf = max(prob0, prob1)
        st.caption(f"Confidence: {conf:.2%}")

        st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("### About")
    st.markdown(
        """
        - **Model:** 1-hidden-layer MLP (ReLU)  
        - **Output:** logits → sigmoid probability  
        - **Scaling:** MaxAbsScaler (fit on training set)  
        - **Decision rule:** predict Class 1 if `P(Class=1) ≥ threshold`
        """)
    st.markdown("### Tips")
    st.markdown(
        """
        - Use realistic feature values (similar to the dataset range).
        - If predictions look extreme (0.0000 or 1.0000), inputs may be out of range.
        - You can tune the threshold if you care more about precision vs recall.
        """)
    st.markdown('<p class="small">If you want, we can auto-fill defaults using dataset mean/median by saving stats into metadata.</p>', unsafe_allow_html=True)