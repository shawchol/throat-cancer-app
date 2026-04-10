import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from scipy.ndimage import zoom
import io
import os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Throat Cancer Detection AI",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SAMPLE_RATE  = 16000
DURATION     = 3
SAMPLES      = SAMPLE_RATE * DURATION
N_MFCC       = 40
MAX_PAD_LEN  = 174
CLASS_NAMES  = ['Healthy', 'Laryngozele', 'Vox senilis']
BINARY_NAMES = ['Healthy', 'Diseased']

# ─────────────────────────────────────────────
# Custom Layers (needed to load model)
# ─────────────────────────────────────────────
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_W',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform', trainable=True)
        self.V = self.add_weight(
            name='att_V',
            shape=(input_shape[-1], 1),
            initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(
            name='att_b',
            shape=(input_shape[1], 1),
            initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        score   = K.tanh(K.dot(x, self.W))
        score   = K.dot(score, self.V) + self.b
        alpha   = K.softmax(score, axis=1)
        return K.sum(alpha * x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(AttentionLayer, self).get_config()


def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0)
        ce     = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
    return focal_loss_fn


def focal_loss_binary(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(y_pred, 1e-8, 1.0)
        bce     = -(y_true * tf.math.log(y_pred) +
                    (1 - y_true) * tf.math.log(1 - y_pred))
        p_t     = tf.where(tf.cast(y_true, tf.bool), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.cast(y_true, tf.bool),
                           tf.ones_like(y_pred) * alpha,
                           tf.ones_like(y_pred) * (1 - alpha))
        return tf.reduce_mean(alpha_t * tf.pow(1 - p_t, gamma) * bce)
    return focal_loss_fn


# ─────────────────────────────────────────────
# Load Models (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    custom_objs = {
        'AttentionLayer': AttentionLayer,
        'focal_loss_fn' : focal_loss(),
    }
    model_multi  = load_model(
        'cnn_bilstm_attention_multiclass_v2_final.h5',
        custom_objects=custom_objs
    )
    model_binary = load_model(
        'cnn_bilstm_attention_binary_v2_final.h5',
        custom_objects=custom_objs
    )
    return model_multi, model_binary


# ─────────────────────────────────────────────
# Audio Processing
# ─────────────────────────────────────────────
def preprocess_audio(audio_bytes):
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = np.pad(audio, (0, SAMPLES - len(audio)), mode='constant')
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio


def extract_mfcc(signal):
    mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_PAD_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_PAD_LEN - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_PAD_LEN]
    return mfcc[..., np.newaxis]   # (40, 174, 1)


# ─────────────────────────────────────────────
# XAI: Attention
# ─────────────────────────────────────────────
def get_attention_weights(model, sample):
    bilstm_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer('BiLSTM').output
    )
    bilstm_out = bilstm_model.predict(sample[np.newaxis, ...], verbose=0)[0]
    att_layer  = model.get_layer('Attention')
    W = att_layer.get_weights()[0]
    V = att_layer.get_weights()[1]
    b = att_layer.get_weights()[2]
    score = np.tanh(bilstm_out @ W) @ V + b
    score = score - score.max()
    alpha = np.exp(score) / np.exp(score).sum()
    return alpha.squeeze()


# ─────────────────────────────────────────────
# XAI: Grad-CAM
# ─────────────────────────────────────────────
def compute_gradcam(model, sample, class_idx):
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer('Conv4').output, model.output]
    )
    with tf.GradientTape() as tape:
        inp       = tf.cast(sample[np.newaxis, ...], tf.float32)
        conv_out, preds = grad_model(inp)
        loss      = preds[:, class_idx]
    grads        = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_np      = conv_out[0].numpy()
    cam          = np.zeros(conv_np.shape[:2], dtype=np.float32)
    for k, w in enumerate(pooled_grads.numpy()):
        cam += w * conv_np[:, :, k]
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    zf  = (sample.shape[0] / cam.shape[0], sample.shape[1] / cam.shape[1])
    return zoom(cam, zf)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🧠 Throat Cancer Detection from Voice")
st.markdown("""
**Capstone Project** — CNN-BiLSTM-Attention with Explainable AI

Upload a `.wav` voice recording to get:
- 🎯 Diagnosis (Healthy / Laryngozele / Vox Senilis)
- 📊 Confidence scores
- 🔍 XAI explanation (Attention + Grad-CAM)
""")

st.divider()

# ── Sidebar ──
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** CNN-BiLSTM + Attention

    **Classes:**
    - ✅ Healthy (Normal)
    - 🔴 Laryngozele
    - 🟠 Vox Senilis

    **XAI Methods:**
    - Attention Visualization
    - Grad-CAM Heatmap

    **Input:** `.wav` audio file
    """)
    st.divider()
    st.caption("Capstone Project — Throat Cancer Detection AI")

# ── File Upload ──
uploaded_file = st.file_uploader(
    "📁 Upload a voice recording (.wav)",
    type=["wav"],
    help="Upload a .wav file of a voice recording"
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    audio_bytes = uploaded_file.read()

    with st.spinner("🔄 Processing audio..."):
        # Preprocess
        signal = preprocess_audio(audio_bytes)
        mfcc   = extract_mfcc(signal)   # (40, 174, 1)

        # Load models
        try:
            model_multi, model_binary = load_models()
        except Exception as e:
            st.error(f"❌ Could not load models: {e}")
            st.stop()

        # Predict
        mfcc_input   = mfcc[np.newaxis, ...]              # (1, 40, 174, 1)
        pred_multi   = model_multi.predict(mfcc_input, verbose=0)[0]   # (3,)
        pred_binary  = model_binary.predict(mfcc_input, verbose=0)[0]  # (1,)

        pred_class   = np.argmax(pred_multi)
        pred_label   = CLASS_NAMES[pred_class]
        confidence   = pred_multi[pred_class] * 100

        binary_prob  = float(pred_binary[0])
        binary_label = "Diseased" if binary_prob >= 0.5 else "Healthy"
        binary_conf  = binary_prob * 100 if binary_prob >= 0.5 else (1 - binary_prob) * 100

    st.divider()

    # ── Results ──
    st.header("📊 Diagnosis Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        color = "🟢" if pred_label == "Healthy" else "🔴"
        st.metric(
            label="Multi-class Diagnosis",
            value=f"{color} {pred_label}",
            delta=f"Confidence: {confidence:.1f}%"
        )

    with col2:
        color2 = "🟢" if binary_label == "Healthy" else "🔴"
        st.metric(
            label="Binary Diagnosis",
            value=f"{color2} {binary_label}",
            delta=f"Confidence: {binary_conf:.1f}%"
        )

    with col3:
        st.metric(
            label="Voice Duration",
            value="3 seconds",
            delta=f"Sample Rate: {SAMPLE_RATE} Hz"
        )

    # ── Confidence bar chart ──
    st.subheader("📈 Confidence Scores (Multi-class)")
    fig, ax = plt.subplots(figsize=(8, 3))
    colors  = ['#4CAF50' if i == pred_class else '#B0BEC5' for i in range(3)]
    bars    = ax.barh(CLASS_NAMES, pred_multi * 100, color=colors, edgecolor='white')
    ax.set_xlabel('Confidence (%)')
    ax.set_xlim(0, 100)
    for bar, val in zip(bars, pred_multi):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', fontsize=11)
    ax.set_title('Prediction Confidence per Class')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── MFCC Visualization ──
    st.header("🎵 MFCC Visualization")
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(mfcc[:, :, 0], aspect='auto', origin='lower', cmap='magma')
    ax.set_title('MFCC of Uploaded Voice')
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('MFCC Coefficient')
    plt.colorbar(img, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── XAI Section ──
    st.header("🔍 Explainable AI (XAI)")
    st.markdown("Understanding **why** the model made this decision:")

    xai_col1, xai_col2 = st.columns(2)

    # ── Attention Weights ──
    with xai_col1:
        st.subheader("🎯 Attention Weights")
        st.caption("Higher bar = model focused more on that time step")

        with st.spinner("Computing attention..."):
            alpha = get_attention_weights(model_multi, mfcc)

        fig, ax = plt.subplots(figsize=(7, 3))
        bar_colors = mpl_cm.RdYlGn(alpha / alpha.max())
        ax.bar(range(len(alpha)), alpha, color=bar_colors, alpha=0.9, width=1.0)

        top5 = np.argsort(alpha)[-5:]
        for t in top5:
            ax.bar(t, alpha[t], color='gold', alpha=1.0, width=1.0,
                   edgecolor='black', linewidth=0.8)

        ax.axhline(y=alpha.mean(), color='navy', linestyle='--',
                   linewidth=1.2, label=f'Mean = {alpha.mean():.4f}')
        ax.set_xlabel('BiLSTM Time Step')
        ax.set_ylabel('Attention Weight α')
        ax.set_title(f'Attention — Predicted: {pred_label}')
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info("🟡 Gold bars = Top-5 most important voice segments")

    # ── Grad-CAM ──
    with xai_col2:
        st.subheader("🌡️ Grad-CAM Heatmap")
        st.caption("Red/warm regions = CNN focused here for diagnosis")

        with st.spinner("Computing Grad-CAM..."):
            cam = compute_gradcam(model_multi, mfcc, pred_class)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.imshow(mfcc[:, :, 0], aspect='auto', origin='lower',
                  cmap='magma', alpha=0.5, interpolation='nearest')
        ax.imshow(cam, aspect='auto', origin='lower',
                  cmap='jet', alpha=0.6, interpolation='nearest')
        ax.set_title(f'Grad-CAM — Predicted: {pred_label}')
        ax.set_xlabel('Time Frame')
        ax.set_ylabel('MFCC Coefficient')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info("🔴 Red = CNN activation | 🔵 Blue = Less important")

    st.divider()

    # ── Summary ──
    st.header("📋 Summary")
    if pred_label == "Healthy":
        st.success(f"✅ Voice appears **Healthy** ({confidence:.1f}% confidence)")
    else:
        st.error(f"⚠️ Voice shows signs of **{pred_label}** ({confidence:.1f}% confidence). Please consult a doctor.")

    st.warning("⚠️ This is an AI research tool. Always consult a medical professional for diagnosis.")

else:
    # ── Placeholder when no file uploaded ──
    st.info("👆 Please upload a `.wav` voice recording to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Diagnosis")
        st.markdown("Get multi-class and binary diagnosis from voice")
    with col2:
        st.markdown("### 📊 Confidence")
        st.markdown("See probability scores for each class")
    with col3:
        st.markdown("### 🔍 XAI")
        st.markdown("Understand why the model made its decision")
