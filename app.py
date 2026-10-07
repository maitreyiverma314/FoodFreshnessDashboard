import os
import io
import time
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
import matplotlib.cm as cm

from supabase import create_client


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Food Freshness Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "ResNet50"

MODEL_FILENAME = "best_ResNet50.keras"

MODEL_PATH = os.path.join(
    BASE_DIR,
    MODEL_FILENAME
)

OPTIMIZER_NAME = "RMSprop"

PRETRAINED_WEIGHTS = "ImageNet"

LEARNING_STRATEGY = "Transfer Learning"

IMG_SIZE = (224, 224)

THRESHOLD = 0.5

TRAINING_IMAGES = 8389

VALIDATION_IMAGES = 1798

TEST_IMAGES = 1799

DENSE_UNITS = 128

DROPOUT = 0.3

LEARNING_RATE = 0.0003


# ============================================================
# RESNET50 TEST CONFUSION MATRIX
# ============================================================

#                 Predicted
#                 Fresh   Rotten
#
# Actual Fresh      902      15
#        Rotten      23     859

RESNET_CONFUSION_MATRIX = np.array(
    [
        [902, 15],
        [23, 859]
    ],
    dtype=int
)


# ============================================================
# RESULT FILES
# ============================================================

COMPARISON_PATH = os.path.join(
    BASE_DIR,
    "final_comparison_percentage.csv"
)

TRAINING_RESULTS_PATH = os.path.join(
    BASE_DIR,
    "final_training_results.csv"
)


# ============================================================
# SUPABASE
# ============================================================

HISTORY_TABLE = "detections"


@st.cache_resource
def get_supabase_client():

    try:

        url = st.secrets["SUPABASE_URL"]

        key = st.secrets["SUPABASE_KEY"]

        return create_client(
            url,
            key
        )

    except Exception:

        return None


supabase = get_supabase_client()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #ffffff;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.7rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        color: #26354a;
    }

    p, div, span, label {
        color: #26354a;
    }

    [data-testid="stSidebar"] {
        background-color: #f0f3f7;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem;
    }

    .sidebar-brand {
        font-size: 1.35rem;
        font-weight: 750;
        color: #26354a;
        margin-bottom: 1.5rem;
    }

    .sidebar-heading {
        font-size: 1.05rem;
        font-weight: 750;
        color: #26354a;
        margin-top: 1.1rem;
        margin-bottom: 0.8rem;
    }

    .sidebar-text {
        font-size: 0.92rem;
        line-height: 1.65;
        color: #344054;
    }

    .sidebar-divider {
        height: 1px;
        background: #d3d8df;
        margin: 23px 0;
    }

    .status-card {
        border-radius: 12px;
        padding: 14px 18px;
        margin: 10px 0;
        font-weight: 650;
        font-size: 0.98rem;
    }

    .fresh-card {
        background-color: #dff3e9;
        color: #078749 !important;
    }

    .rotten-card {
        background-color: #f5dde1;
        color: #d52d40 !important;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #2f3442;
        margin-bottom: 0.25rem;
    }

    .main-subtitle {
        font-size: 1.28rem;
        font-weight: 650;
        color: #26354a;
        margin-bottom: 0.7rem;
    }

    .technology-line {
        font-size: 0.95rem;
        color: #334155;
        margin-bottom: 1.5rem;
    }

    .section-divider {
        height: 1px;
        background: #d9dde2;
        margin: 1.45rem 0 1.75rem 0;
    }

    .section-title {
        font-size: 1.9rem;
        font-weight: 750;
        color: #29384d;
        margin-bottom: 0.8rem;
    }

    .subsection-title {
        font-size: 1.38rem;
        font-weight: 700;
        color: #2c3d55;
        margin-top: 0.8rem;
        margin-bottom: 0.75rem;
    }

    .metric-card {
        border: 1px solid #dce1e7;
        border-radius: 14px;
        padding: 18px;
        background-color: #ffffff;
        min-height: 125px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.035);
    }

    .metric-label {
        font-size: 0.91rem;
        color: #667085 !important;
        margin-bottom: 7px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #26354a !important;
    }

    .result-card {
        border: 1px solid #dde3e9;
        border-radius: 14px;
        padding: 18px;
        background-color: #ffffff;
        margin-bottom: 16px;
    }

    .result-fresh {
        border-left: 6px solid #21b86b;
    }

    .result-rotten {
        border-left: 6px solid #e33d4d;
    }

    .result-title {
        font-size: 1.3rem;
        font-weight: 750;
        margin-bottom: 7px;
    }

    .result-confidence {
        font-size: 1.02rem;
        font-weight: 650;
    }

    .result-line {
        font-size: 0.95rem;
        line-height: 1.7;
    }

    .info-box {
        background-color: #eaf3ff;
        border: 1px solid #d4e6fa;
        border-radius: 12px;
        padding: 17px;
        color: #1258a4 !important;
        line-height: 1.6;
    }

    .pipeline-card {
        border: 1px solid #dfe4e9;
        border-radius: 14px;
        padding: 18px;
        min-height: 175px;
        background-color: #ffffff;
    }

    .pipeline-icon {
        font-size: 2rem;
        margin-bottom: 11px;
    }

    .pipeline-title {
        font-size: 1rem;
        font-weight: 700;
        color: #26354a;
        margin-bottom: 9px;
    }

    .pipeline-description {
        font-size: 0.89rem;
        line-height: 1.5;
        color: #798190 !important;
    }

    .project-text {
        font-size: 1rem;
        line-height: 1.85;
        color: #344054;
    }

    .footer {
        color: #89919d !important;
        font-size: 0.88rem;
        margin-top: 30px;
        padding-top: 17px;
        border-top: 1px solid #d9dde2;
        line-height: 1.6;
    }

    div.stButton > button {
        border-radius: 9px;
        font-weight: 650;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SUPABASE HISTORY
# ============================================================

def load_cloud_history():

    if supabase is None:
        return []

    try:

        response = (
            supabase
            .table(HISTORY_TABLE)
            .select("*")
            .order(
                "created_at",
                desc=True
            )
            .limit(500)
            .execute()
        )

        return response.data or []

    except Exception as error:

        st.warning(
            f"Could not load cloud history: {error}"
        )

        return []


def save_batch_to_cloud(
    results
):

    if supabase is None:

        return False

    if not results:

        return True

    records = []

    for result in results:

        records.append(
            {
                "image_name":
                    result["name"],

                "prediction":
                    result["label"],

                "confidence":
                    float(
                        result["confidence"]
                    ),

                "fresh_probability":
                    float(
                        result["fresh_probability"]
                    ),

                "rotten_probability":
                    float(
                        result["rotten_probability"]
                    ),

                "image_quality":
                    result["quality"],

                "inference_ms":
                    float(
                        result["inference_ms"]
                    ),

                "model_name":
                    "ResNet50"
            }
        )

    try:

        (
            supabase
            .table(HISTORY_TABLE)
            .insert(records)
            .execute()
        )

        return True

    except Exception as error:

        st.warning(
            f"Could not save detections: {error}"
        )

        return False


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = (
        load_cloud_history()
    )


if "analysis_results" not in st.session_state:

    st.session_state.analysis_results = []


if "camera_images" not in st.session_state:

    st.session_state.camera_images = []


if "camera_key" not in st.session_state:

    st.session_state.camera_key = 0


if "last_camera_hash" not in st.session_state:

    st.session_state.last_camera_hash = None


# ============================================================
# LOAD RESNET50
# ============================================================

@st.cache_resource
def load_resnet50():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"""
ResNet50 model file not found.

Expected:
{MODEL_PATH}

Make sure the file:

best_ResNet50.keras

is in the same folder as app.py.
"""
        )

    loaded_model = (
        tf.keras.models.load_model(
            MODEL_PATH
        )
    )

    return loaded_model


try:

    model = load_resnet50()

except Exception as error:

    st.error(
        f"Unable to load ResNet50 model:\n\n{error}"
    )

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

print("=" * 70)
print("AI FOOD FRESHNESS DETECTION")
print("=" * 70)
print("Model       :", MODEL_NAME)
print("Optimizer   :", OPTIMIZER_NAME)
print("Model file  :", MODEL_PATH)
print("Input size  :", IMG_SIZE)
print(
    "Parameters  :",
    f"{model.count_params():,}"
)
print(
    "Supabase    :",
    "Connected"
    if supabase is not None
    else "Not connected"
)
print("=" * 70)


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def preprocess_resnet50(
    image
):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )

    array = np.asarray(
        image,
        dtype=np.float32
    )

    array = (
        tf.keras
        .applications
        .resnet50
        .preprocess_input(
            array
        )
    )

    return np.expand_dims(
        array,
        axis=0
    )


# ============================================================
# LOAD CSV FILES
# ============================================================

@st.cache_data
def load_csv(
    path
):

    if not os.path.exists(path):

        return None

    try:

        return pd.read_csv(
            path
        )

    except Exception:

        return None


comparison_df = load_csv(
    COMPARISON_PATH
)

training_df = load_csv(
    TRAINING_RESULTS_PATH
)


# ============================================================
# FIND MODEL ROW
# ============================================================

def get_model_row(
    dataframe,
    model_name
):

    if dataframe is None:
        return None

    if "Model" not in dataframe.columns:
        return None

    rows = dataframe[
        dataframe["Model"]
        .astype(str)
        .str.lower()
        == model_name.lower()
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


comparison_row = get_model_row(
    comparison_df,
    "ResNet50"
)

training_row = get_model_row(
    training_df,
    "ResNet50"
)


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def get_metric(
    row,
    names
):

    if row is None:
        return None

    for name in names:

        if name in row.index:

            value = row[name]

            if pd.isna(value):
                continue

            try:

                return float(value)

            except Exception:

                continue

    return None


def normalize_percentage(
    value
):

    if value is None:
        return None

    try:

        value = float(value)

        if value <= 1:

            return value * 100

        return value

    except Exception:

        return None


def format_percentage(
    value
):

    value = normalize_percentage(
        value
    )

    if value is None:

        return "—"

    return f"{value:.2f}%"


# ============================================================
# MODEL METRICS
# ============================================================

TEST_ACCURACY = get_metric(
    comparison_row,
    [
        "Accuracy",
        "Test Accuracy"
    ]
)


PRECISION = get_metric(
    comparison_row,
    [
        "Precision"
    ]
)


RECALL = get_metric(
    comparison_row,
    [
        "Recall"
    ]
)


F1_SCORE = get_metric(
    comparison_row,
    [
        "F1-Score",
        "F1 Score"
    ]
)


BEST_VALIDATION_ACCURACY = get_metric(
    training_row,
    [
        "Best Val Accuracy",
        "Best Validation Accuracy",
        "Val Accuracy"
    ]
)


# ============================================================
# FALLBACK METRICS
# ============================================================

if TEST_ACCURACY is None:

    correct = (
        RESNET_CONFUSION_MATRIX[0, 0]
        + RESNET_CONFUSION_MATRIX[1, 1]
    )

    total = int(
        RESNET_CONFUSION_MATRIX.sum()
    )

    TEST_ACCURACY = (
        correct / total
    )


if PRECISION is None:

    tp = RESNET_CONFUSION_MATRIX[1, 1]

    fp = RESNET_CONFUSION_MATRIX[0, 1]

    PRECISION = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )


if RECALL is None:

    tp = RESNET_CONFUSION_MATRIX[1, 1]

    fn = RESNET_CONFUSION_MATRIX[1, 0]

    RECALL = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )


if F1_SCORE is None:

    p = float(
        PRECISION
    )

    r = float(
        RECALL
    )

    if p + r > 0:

        F1_SCORE = (
            2 * p * r
            / (p + r)
        )

    else:

        F1_SCORE = 0


# ============================================================
# MODEL SIZE
# ============================================================

try:

    MODEL_SIZE_MB = (
        os.path.getsize(
            MODEL_PATH
        )
        / (
            1024 * 1024
        )
    )

except Exception:

    MODEL_SIZE_MB = None


# ============================================================
# IMAGE QUALITY
# ============================================================

def assess_image_quality(
    image
):

    gray = np.asarray(
        image.convert("L"),
        dtype=np.float32
    )

    brightness = float(
        gray.mean()
    )

    contrast = float(
        gray.std()
    )

    if gray.shape[1] > 1:

        horizontal_detail = float(
            np.abs(
                np.diff(
                    gray,
                    axis=1
                )
            ).mean()
        )

    else:

        horizontal_detail = 0.0


    if gray.shape[0] > 1:

        vertical_detail = float(
            np.abs(
                np.diff(
                    gray,
                    axis=0
                )
            ).mean()
        )

    else:

        vertical_detail = 0.0


    sharpness = (
        horizontal_detail
        + vertical_detail
    ) / 2


    if (
        40 <= brightness <= 220
        and contrast >= 24
        and sharpness >= 7
    ):

        quality = "Good"

    elif (
        25 <= brightness <= 235
        and contrast >= 15
        and sharpness >= 4
    ):

        quality = "Fair"

    else:

        quality = "Low"


    return {
        "quality": quality,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_images(
    image_records
):

    processed = []


    for record in image_records:

        image = record[
            "image"
        ].convert("RGB")


        resized = image.resize(
            IMG_SIZE,
            Image.Resampling.BILINEAR
        )


        array = np.asarray(
            resized,
            dtype=np.float32
        )


        array = (
            tf.keras
            .applications
            .resnet50
            .preprocess_input(
                array
            )
        )


        processed.append(
            array
        )


    batch = np.stack(
        processed,
        axis=0
    )


    start_time = (
        time.perf_counter()
    )


    predictions = model.predict(
        batch,
        verbose=0
    )


    elapsed = (
        time.perf_counter()
        - start_time
    )


    predictions = (
        predictions
        .reshape(-1)
    )


    inference_ms = (
        elapsed
        / len(image_records)
        * 1000
    )


    results = []


    for index, rotten_probability in enumerate(
        predictions
    ):

        rotten_probability = float(
            rotten_probability
        )


        fresh_probability = (
            1
            - rotten_probability
        )


        if (
            rotten_probability
            >= THRESHOLD
        ):

            label = "Rotten"

            confidence = (
                rotten_probability
            )

        else:

            label = "Fresh"

            confidence = (
                fresh_probability
            )


        quality = assess_image_quality(
            image_records[index]["image"]
        )


        results.append(
            {
                "name":
                    image_records[index]["name"],

                "image":
                    image_records[index]["image"],

                "label":
                    label,

                "confidence":
                    confidence,

                "fresh_probability":
                    fresh_probability,

                "rotten_probability":
                    rotten_probability,

                "quality":
                    quality["quality"],

                "brightness":
                    quality["brightness"],

                "contrast":
                    quality["contrast"],

                "sharpness":
                    quality["sharpness"],

                "inference_ms":
                    inference_ms,

                "timestamp":
                    datetime.now(
                        timezone.utc
                    ).isoformat()
            }
        )


    return results


# ============================================================
# ROBUST GRAD-CAM
# ============================================================

def find_last_conv_layer(
    current_model
):

    """
    Find the last convolutional layer anywhere in the model.

    This does NOT require ResNet50 to appear as a nested
    tf.keras.Model.
    """

    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D
    )


    # --------------------------------------------------------
    # First search direct top-level layers
    # --------------------------------------------------------

    for layer in reversed(
        current_model.layers
    ):

        if isinstance(
            layer,
            conv_types
        ):

            try:

                if (
                    len(
                        layer.output.shape
                    )
                    == 4
                ):

                    return layer

            except Exception:

                pass


    # --------------------------------------------------------
    # Search nested layers recursively
    # --------------------------------------------------------

    def recursive_search(
        layers
    ):

        for layer in reversed(
            layers
        ):

            if isinstance(
                layer,
                conv_types
            ):

                try:

                    if (
                        len(
                            layer.output.shape
                        )
                        == 4
                    ):

                        return layer

                except Exception:

                    pass


            if hasattr(
                layer,
                "layers"
            ):

                try:

                    found = recursive_search(
                        layer.layers
                    )

                    if found is not None:

                        return found

                except Exception:

                    pass


        return None


    return recursive_search(
        current_model.layers
    )


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    conv_outputs,
    gradients
):

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )


    feature_maps = (
        conv_outputs[0]
    )

    weights = (
        pooled_gradients[0]
    )


    weighted_features = (
        feature_maps
        * weights
    )


    heatmap = tf.reduce_sum(
        weighted_features,
        axis=-1
    )


    heatmap = tf.maximum(
        heatmap,
        0
    )


    maximum = tf.reduce_max(
        heatmap
    )


    maximum_value = float(
        maximum.numpy()
    )


    if maximum_value <= 0:

        raise RuntimeError(
            "Grad-CAM produced an empty heatmap."
        )


    heatmap = (
        heatmap
        / (
            maximum
            + tf.keras.backend.epsilon()
        )
    )


    heatmap = heatmap.numpy()


    original = image.convert(
        "RGB"
    )


    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )


    heatmap_image = heatmap_image.resize(
        original.size,
        Image.Resampling.BILINEAR
    )


    heatmap_array = (
        np.asarray(
            heatmap_image,
            dtype=np.float32
        )
        / 255.0
    )


    colored_heatmap = (
        cm.jet(
            heatmap_array
        )[:, :, :3]
        * 255
    ).astype(
        np.uint8
    )


    colored_heatmap = Image.fromarray(
        colored_heatmap
    )


    overlay = Image.blend(
        original,
        colored_heatmap,
        alpha=0.40
    )


    return overlay


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    image,
    predicted_label
):

    # --------------------------------------------------------
    # Find final convolutional layer
    # --------------------------------------------------------

    last_conv_layer = find_last_conv_layer(
        model
    )


    if last_conv_layer is None:

        raise RuntimeError(
            "No convolutional layer could be found "
            "inside the loaded ResNet50 model."
        )


    print(
        "Grad-CAM layer:",
        last_conv_layer.name
    )


    print(
        "Grad-CAM shape:",
        last_conv_layer.output.shape
    )


    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    input_tensor = preprocess_resnet50(
        image
    )


    # --------------------------------------------------------
    # Try the normal Keras Grad-CAM graph
    # --------------------------------------------------------

    try:

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                last_conv_layer.output,
                model.output
            ]
        )


    except Exception as error:

        raise RuntimeError(
            "The last convolutional layer was found "
            f"({last_conv_layer.name}), but it could not "
            "be connected to the final model output.\n\n"
            f"Details: {error}"
        )


    # --------------------------------------------------------
    # Gradient calculation
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            grad_model(
                input_tensor,
                training=False
            )
        )


        rotten_probability = (
            predictions[:, 0]
        )


        if predicted_label == "Rotten":

            target_score = (
                rotten_probability
            )

        else:

            target_score = (
                1.0
                - rotten_probability
            )


    # --------------------------------------------------------
    # Calculate gradients
    # --------------------------------------------------------

    gradients = tape.gradient(
        target_score,
        conv_outputs
    )


    if gradients is None:

        raise RuntimeError(
            f"Gradients were None for layer "
            f"'{last_conv_layer.name}'."
        )


    # --------------------------------------------------------
    # Generate visualization
    # --------------------------------------------------------

    return create_gradcam_overlay(
        image,
        conv_outputs,
        gradients
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class='sidebar-brand'>
        🤖 AI Food Vision
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "<div class='sidebar-divider'></div>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class='sidebar-heading'>
        About the System
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class='sidebar-text'>

        An AI-powered computer vision system that
        classifies food images as <b>Fresh</b> or
        <b>Rotten</b> using ResNet50 transfer learning.

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class='status-card fresh-card'>
        🟢 Fresh
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class='status-card rotten-card'>
        🔴 Rotten
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "<div class='sidebar-divider'></div>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class='sidebar-heading'>
        🧠 Model
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class='sidebar-text'>

        <b>Architecture:</b>
        ResNet50

        <br><br>

        <b>Learning:</b>
        Transfer Learning

        <br><br>

        <b>Pretrained Weights:</b>
        ImageNet

        <br><br>

        <b>Optimizer:</b>
        RMSprop

        <br><br>

        <b>Input:</b>
        224 × 224 pixels

        <br><br>

        <b>Classes:</b>
        Fresh / Rotten

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "<div class='sidebar-divider'></div>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class='sidebar-heading'>
        📊 Test Performance
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div class='sidebar-text'>

        <b>Test Accuracy</b>

        <br>

        <span style='font-size:2rem;'>
        {format_percentage(TEST_ACCURACY)}
        </span>

        <br><br>

        <b>Best Validation Accuracy</b>

        <br>

        <span style='font-size:2rem;'>
        {format_percentage(BEST_VALIDATION_ACCURACY)}
        </span>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "<div class='sidebar-divider'></div>",
        unsafe_allow_html=True
    )


    if supabase is not None:

        st.success(
            "☁️ Cloud history connected"
        )

    else:

        st.warning(
            "☁️ Cloud history unavailable"
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class='main-title'>
    🤖 AI Food Freshness Detection
    </div>

    <div class='main-subtitle'>
    Computer Vision Based Food Quality Classification
    using ResNet50 Transfer Learning
    </div>

    <div class='technology-line'>
    🧠 Deep Learning • Computer Vision • Artificial Intelligence •
    Explainable AI
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True
)


# ============================================================
# FOOD IMAGE INPUT
# ============================================================

st.markdown(
    """
    <div class='section-title'>
    📷 Food Image Input
    </div>
    """,
    unsafe_allow_html=True
)


input_mode = st.radio(
    "Select input method",
    [
        "Upload Multiple Images",
        "Capture Multiple Photos"
    ],
    horizontal=True
)


# ============================================================
# UPLOAD MODE
# ============================================================

if input_mode == "Upload Multiple Images":

    uploaded_files = st.file_uploader(
        "Choose one or more food images",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True
    )


    image_records = []


    if uploaded_files:

        for uploaded_file in uploaded_files:

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")


                image_records.append(
                    {
                        "name":
                            uploaded_file.name,

                        "image":
                            image
                    }
                )


            except Exception:

                st.warning(
                    f"Could not read {uploaded_file.name}"
                )


# ============================================================
# CAMERA MODE
# ============================================================

else:

    st.markdown(
        """
        <div class='info-box'>

        Capture a food photo, then capture another photo
        to build a batch. All captured photos can be
        analyzed together.

        </div>
        """,
        unsafe_allow_html=True
    )


    camera_photo = st.camera_input(
        "Capture food photo",
        key=f"camera_{st.session_state.camera_key}"
    )


    if camera_photo is not None:

        camera_bytes = (
            camera_photo
            .getvalue()
        )


        camera_hash = hashlib.md5(
            camera_bytes
        ).hexdigest()


        if (
            camera_hash
            != st.session_state.last_camera_hash
        ):

            st.session_state.camera_images.append(
                {
                    "name":
                        f"Camera Photo "
                        f"{len(st.session_state.camera_images) + 1}.jpg",

                    "bytes":
                        camera_bytes
                }
            )


            st.session_state.last_camera_hash = (
                camera_hash
            )


            st.session_state.camera_key += 1


            st.rerun()


    camera_col1, camera_col2 = (
        st.columns(2)
    )


    with camera_col1:

        if st.session_state.camera_images:

            st.success(
                f"{len(st.session_state.camera_images)} "
                f"photo(s) captured."
            )


    with camera_col2:

        if st.button(
            "Clear Captured Photos",
            use_container_width=True
        ):

            st.session_state.camera_images = []

            st.session_state.camera_key = 0

            st.session_state.last_camera_hash = None

            st.rerun()


    image_records = []


    for camera_record in (
        st.session_state.camera_images
    ):

        try:

            image = Image.open(
                io.BytesIO(
                    camera_record["bytes"]
                )
            ).convert("RGB")


            image_records.append(
                {
                    "name":
                        camera_record["name"],

                    "image":
                        image
                }
            )


        except Exception:

            pass


# ============================================================
# IMAGE PREVIEW
# ============================================================

if image_records:

    st.markdown(
        """
        <div class='subsection-title'>
        Selected Images
        </div>
        """,
        unsafe_allow_html=True
    )


    preview_count = min(
        4,
        len(image_records)
    )


    preview_columns = st.columns(
        preview_count
    )


    for index, record in enumerate(
        image_records
    ):

        with preview_columns[
            index % preview_count
        ]:

            st.image(
                record["image"],
                caption=record["name"],
                use_container_width=True
            )


# ============================================================
# ANALYZE BUTTON
# ============================================================

if image_records:

    if st.button(
        "🔍 Analyze Food Images",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing food images using ResNet50..."
        ):

            results = predict_images(
                image_records
            )


        st.session_state.analysis_results = (
            results
        )


        # ----------------------------------------------------
        # SAVE PERMANENTLY TO SUPABASE
        # ----------------------------------------------------

        if supabase is not None:

            save_success = (
                save_batch_to_cloud(
                    results
                )
            )


            if save_success:

                st.session_state.prediction_history = (
                    load_cloud_history()
                )

            else:

                st.warning(
                    "Prediction completed, but the "
                    "results could not be saved to the cloud."
                )


        else:

            st.warning(
                "Prediction completed, but Supabase "
                "is not connected."
            )


# ============================================================
# PREDICTION RESULTS
# ============================================================

if st.session_state.analysis_results:

    st.markdown(
        "<div class='section-divider'></div>",
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class='section-title'>
        🎯 Prediction Results
        </div>
        """,
        unsafe_allow_html=True
    )


    for result in (
        st.session_state.analysis_results
    ):

        if result["label"] == "Fresh":

            card_class = (
                "result-card result-fresh"
            )

            icon = "🟢"

        else:

            card_class = (
                "result-card result-rotten"
            )

            icon = "🔴"


        result_image_col, result_info_col = (
            st.columns(
                [1, 2]
            )
        )


        with result_image_col:

            st.image(
                result["image"],
                caption=result["name"],
                use_container_width=True
            )


        with result_info_col:

            st.markdown(
                f"""
                <div class='{card_class}'>

                <div class='result-title'>
                {icon} {result["label"]}
                </div>

                <div class='result-confidence'>
                Confidence:
                {result["confidence"] * 100:.2f}%
                </div>

                <br>

                <div class='result-line'>
                Fresh probability:
                {result["fresh_probability"] * 100:.2f}%
                </div>

                <div class='result-line'>
                Rotten probability:
                {result["rotten_probability"] * 100:.2f}%
                </div>

                <div class='result-line'>
                Image quality:
                {result["quality"]}
                </div>

                <div class='result-line'>
                Approx. inference:
                {result["inference_ms"]:.3f} ms/image
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# DETECTION HISTORY & ANALYTICS
# ============================================================

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class='section-title'>
    📊 Detection History & Analytics
    </div>
    """,
    unsafe_allow_html=True
)


history_df = pd.DataFrame(
    st.session_state.prediction_history
)


history_col1, history_col2 = (
    st.columns(
        [4, 1]
    )
)


with history_col1:

    if history_df.empty:

        st.info(
            "No detections have been recorded yet."
        )

    else:

        st.success(
            f"✅ {len(history_df)} detection(s) "
            "stored in the cloud."
        )


with history_col2:

    if st.button(
        "🔄 Refresh",
        use_container_width=True
    ):

        if supabase is not None:

            st.session_state.prediction_history = (
                load_cloud_history()
            )

        st.rerun()


if not history_df.empty:

    # ========================================================
    # ANALYTICS
    # ========================================================

    total_predictions = len(
        history_df
    )


    if "prediction" in history_df.columns:

        fresh_count = int(
            (
                history_df["prediction"]
                == "Fresh"
            ).sum()
        )


        rotten_count = int(
            (
                history_df["prediction"]
                == "Rotten"
            ).sum()
        )

    else:

        fresh_count = 0

        rotten_count = 0


    if "confidence" in history_df.columns:

        average_confidence = (
            history_df["confidence"]
            .astype(float)
            .mean()
            * 100
        )

    else:

        average_confidence = 0


    analytics_col1, analytics_col2, analytics_col3, analytics_col4 = (
        st.columns(4)
    )


    with analytics_col1:

        st.metric(
            "Total Detections",
            total_predictions
        )


    with analytics_col2:

        st.metric(
            "Fresh",
            fresh_count
        )


    with analytics_col3:

        st.metric(
            "Rotten",
            rotten_count
        )


    with analytics_col4:

        st.metric(
            "Average Confidence",
            f"{average_confidence:.2f}%"
        )


    # ========================================================
    # DISPLAY TABLE
    # ========================================================

    display_history = history_df.copy()


    if "confidence" in display_history.columns:

        display_history["confidence"] = (
            display_history["confidence"]
            .astype(float)
            * 100
        )


    if "fresh_probability" in display_history.columns:

        display_history["fresh_probability"] = (
            display_history[
                "fresh_probability"
            ]
            .astype(float)
            * 100
        )


    if "rotten_probability" in display_history.columns:

        display_history["rotten_probability"] = (
            display_history[
                "rotten_probability"
            ]
            .astype(float)
            * 100
        )


    display_history = display_history.rename(
        columns={
            "id":
                "ID",

            "created_at":
                "Timestamp",

            "image_name":
                "Image",

            "prediction":
                "Prediction",

            "confidence":
                "Confidence (%)",

            "fresh_probability":
                "Fresh Probability (%)",

            "rotten_probability":
                "Rotten Probability (%)",

            "image_quality":
                "Image Quality",

            "inference_ms":
                "Inference (ms)",

            "model_name":
                "Model"
        }
    )


    for column in [
        "Confidence (%)",
        "Fresh Probability (%)",
        "Rotten Probability (%)",
        "Inference (ms)"
    ]:

        if column in display_history.columns:

            display_history[column] = (
                display_history[column]
                .astype(float)
                .round(2)
            )


    st.dataframe(
        display_history,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    download_data = (
        display_history
        .to_csv(
            index=False
        )
        .encode("utf-8")
    )


    st.download_button(
        "⬇️ Download Detection History",
        data=download_data,
        file_name=(
            "food_freshness_detection_history.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class='section-title'>
    📈 Model Performance
    </div>
    """,
    unsafe_allow_html=True
)


performance_col1, performance_col2, performance_col3, performance_col4 = (
    st.columns(4)
)


with performance_col1:

    st.markdown(
        f"""
        <div class='metric-card'>

        <div class='metric-label'>
        Test Accuracy
        </div>

        <div class='metric-value'>
        {format_percentage(TEST_ACCURACY)}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with performance_col2:

    st.markdown(
        f"""
        <div class='metric-card'>

        <div class='metric-label'>
        Best Validation Accuracy
        </div>

        <div class='metric-value'>
        {format_percentage(BEST_VALIDATION_ACCURACY)}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with performance_col3:

    st.markdown(
        f"""
        <div class='metric-card'>

        <div class='metric-label'>
        Test Images
        </div>

        <div class='metric-value'>
        {TEST_IMAGES:,}
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with performance_col4:

    st.markdown(
        """
        <div class='metric-card'>

        <div class='metric-label'>
        Classes
        </div>

        <div class='metric-value'>
        2
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

st.markdown(
    """
    <div class='subsection-title'>
    🎯 Classification Results
    </div>
    """,
    unsafe_allow_html=True
)


fresh_correct = int(
    RESNET_CONFUSION_MATRIX[0, 0]
)

fresh_to_rotten = int(
    RESNET_CONFUSION_MATRIX[0, 1]
)

rotten_correct = int(
    RESNET_CONFUSION_MATRIX[1, 1]
)

rotten_to_fresh = int(
    RESNET_CONFUSION_MATRIX[1, 0]
)


classification_col1, classification_col2, classification_col3, classification_col4 = (
    st.columns(4)
)


with classification_col1:

    st.metric(
        "Fresh Correct",
        fresh_correct
    )


with classification_col2:

    st.metric(
        "Fresh → Rotten",
        fresh_to_rotten
    )


with classification_col3:

    st.metric(
        "Rotten Correct",
        rotten_correct
    )


with classification_col4:

    st.metric(
        "Rotten → Fresh",
        rotten_to_fresh
    )


st.caption(
    "These values correspond to the ResNet50 test confusion matrix."
)


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

metric_col1, metric_col2, metric_col3 = (
    st.columns(3)
)


with metric_col1:

    st.metric(
        "Precision",
        format_percentage(PRECISION)
    )


with metric_col2:

    st.metric(
        "Recall",
        format_percentage(RECALL)
    )


with metric_col3:

    st.metric(
        "F1-Score",
        format_percentage(F1_SCORE)
    )


# ============================================================
# TECHNICAL SPECIFICATIONS
# ============================================================

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class='section-title'>
    ⚙️ Technical Specifications
    </div>
    """,
    unsafe_allow_html=True
)


technical_col1, technical_col2 = (
    st.columns(2)
)


with technical_col1:

    st.markdown(
        """
        <div class='metric-card'>

        <h3>🧠 Deep Learning</h3>

        <p>
        <b>Architecture:</b>
        ResNet50
        </p>

        <p>
        <b>Learning Strategy:</b>
        Transfer Learning
        </p>

        <p>
        <b>Pretrained Weights:</b>
        ImageNet
        </p>

        <p>
        <b>Optimizer:</b>
        RMSprop
        </p>

        <p>
        <b>Input Resolution:</b>
        224 × 224 pixels
        </p>

        <p>
        <b>Output:</b>
        Binary Classification
        </p>

        <p>
        <b>Classes:</b>
        Fresh / Rotten
        </p>

        <p>
        <b>Classification Head:</b>
        GlobalAveragePooling2D →
        Dense(128, ReLU) →
        Dropout(0.3) →
        Dense(1, Sigmoid)
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with technical_col2:

    model_size_text = (
        f"{MODEL_SIZE_MB:.2f} MB"
        if MODEL_SIZE_MB is not None
        else "—"
    )


    st.markdown(
        f"""
        <div class='metric-card'>

        <h3>📊 Evaluation</h3>

        <p>
        <b>Training Images:</b>
        {TRAINING_IMAGES:,}
        </p>

        <p>
        <b>Validation Images:</b>
        {VALIDATION_IMAGES:,}
        </p>

        <p>
        <b>Testing Images:</b>
        {TEST_IMAGES:,}
        </p>

        <p>
        <b>Test Accuracy:</b>
        {format_percentage(TEST_ACCURACY)}
        </p>

        <p>
        <b>Best Validation Accuracy:</b>
        {format_percentage(BEST_VALIDATION_ACCURACY)}
        </p>

        <p>
        <b>Precision:</b>
        {format_percentage(PRECISION)}
        </p>

        <p>
        <b>Recall:</b>
        {format_percentage(RECALL)}
        </p>

        <p>
        <b>F1-Score:</b>
        {format_percentage(F1_SCORE)}
        </p>

        <p>
        <b>Model Size:</b>
        {model_size_text}
        </p>

        <p>
        <b>Loss Function:</b>
        Binary Cross-Entropy
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# AI PROCESSING PIPELINE
# ============================================================

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class='section-title'>
    🔄 AI Processing Pipeline
    </div>
    """,
    unsafe_allow_html=True
)


pipeline_items = [

    (
        "📷",
        "Input",
        "Upload or capture multiple images"
    ),

    (
        "🔍",
        "Quality",
        "Image quality assessment"
    ),

    (
        "⚙️",
        "Preprocess",
        "224 × 224 + ResNet50 preprocessing"
    ),

    (
        "🧠",
        "ResNet50",
        "Transfer-learning inference"
    ),

    (
        "🎯",
        "Prediction",
        "Fresh / Rotten"
    ),

    (
        "🔥",
        "Grad-CAM",
        "Explain selected image"
    )
]


pipeline_columns = st.columns(
    6
)


for column, item in zip(
    pipeline_columns,
    pipeline_items
):

    icon, title, description = item

    with column:

        st.markdown(
            f"""
            <div class='pipeline-card'>

            <div class='pipeline-icon'>
            {icon}
            </div>

            <div class='pipeline-title'>
            {title}
            </div>

            <div class='pipeline-description'>
            {description}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# GRAD-CAM
# ============================================================

st.markdown(
    """
    <div class='subsection-title'>
    🔥 Explainable AI — Grad-CAM
    </div>
    """,
    unsafe_allow_html=True
)


if not st.session_state.analysis_results:

    st.info(
        "Run a prediction first to enable Grad-CAM."
    )


else:

    gradcam_options = [
        result["name"]
        for result
        in st.session_state.analysis_results
    ]


    selected_gradcam = st.selectbox(
        "Select an image to explain",
        gradcam_options
    )


    selected_result = next(
        (
            result
            for result
            in st.session_state.analysis_results
            if result["name"]
            == selected_gradcam
        ),
        None
    )


    if selected_result is not None:

        st.markdown(
            f"""
            <div class='info-box'>

            Predicted class:
            <b>{selected_result["label"]}</b>

            &nbsp;&nbsp;|&nbsp;&nbsp;

            Confidence:
            <b>
            {selected_result["confidence"] * 100:.2f}%
            </b>

            </div>
            """,
            unsafe_allow_html=True
        )


        if st.button(
            "🔥 Generate Grad-CAM Explanation",
            use_container_width=True
        ):

            with st.spinner(
                "Generating Grad-CAM explanation..."
            ):

                try:

                    gradcam_image = (
                        generate_gradcam(
                            selected_result["image"],
                            selected_result["label"]
                        )
                    )


                    gradcam_col1, gradcam_col2 = (
                        st.columns(2)
                    )


                    with gradcam_col1:

                        st.image(
                            selected_result["image"],
                            caption="Original Image",
                            use_container_width=True
                        )


                    with gradcam_col2:

                        st.image(
                            gradcam_image,
                            caption="Grad-CAM Explanation",
                            use_container_width=True
                        )


                    st.markdown(
                        """
                        <div class='info-box'>

                        <b>How to interpret Grad-CAM:</b><br>

                        The highlighted regions indicate
                        image areas that contributed strongly
                        to the ResNet50 prediction.

                        Warmer regions represent stronger
                        contribution to the selected prediction.

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                except Exception as error:

                    st.error(
                        f"Grad-CAM error:\n\n{error}"
                    )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class='section-title'>
    📚 Project Information
    </div>
    """,
    unsafe_allow_html=True
)


objective_tab, technology_tab, methodology_tab, future_tab = (
    st.tabs(
        [
            "🎯 Objective",
            "🧠 Technology",
            "🔬 Methodology",
            "🚀 Future Scope"
        ]
    )
)


# ============================================================
# OBJECTIVE
# ============================================================

with objective_tab:

    st.markdown(
        """
        <div class='project-text'>

        <h3>Project Objective</h3>

        The objective of this project is to develop an
        AI-powered computer vision system that classifies
        food images as <b>Fresh</b> or <b>Rotten</b>.

        <br><br>

        The system uses <b>ResNet50</b> transfer learning
        to extract visual features and perform binary
        classification.

        <br><br>

        The project compares multiple CNN architectures
        under a common experimental procedure and selects
        a suitable model based on classification performance
        and practical deployment considerations.

        <br><br>

        The dashboard supports multi-image upload,
        camera-based batch inference and explainable AI
        using Grad-CAM.

        <br><br>

        Detection results are stored in a cloud database
        so that the detection history persists across
        application sessions.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TECHNOLOGY
# ============================================================

with technology_tab:

    st.markdown(
        """
        <div class='project-text'>

        <h3>Technology Stack</h3>

        <b>Programming Language:</b>
        Python

        <br><br>

        <b>Deep Learning:</b>
        TensorFlow / Keras

        <br><br>

        <b>Computer Vision:</b>
        Image preprocessing and visual feature extraction

        <br><br>

        <b>Model:</b>
        ResNet50

        <br><br>

        <b>Learning Strategy:</b>
        Transfer Learning

        <br><br>

        <b>Pretrained Weights:</b>
        ImageNet

        <br><br>

        <b>Optimizer:</b>
        RMSprop

        <br><br>

        <b>Dashboard:</b>
        Streamlit

        <br><br>

        <b>Cloud Database:</b>
        Supabase PostgreSQL

        <br><br>

        <b>Data Processing:</b>
        NumPy / Pandas / PIL

        <br><br>

        <b>Explainability:</b>
        Grad-CAM

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# METHODOLOGY
# ============================================================

with methodology_tab:

    st.markdown(
        """
        <div class='project-text'>

        <h3>Methodology</h3>

        <b>1. Dataset Preparation</b><br>

        Food images are organized into Fresh and Rotten
        classes and divided into training, validation
        and testing subsets.

        <br><br>

        <b>2. Image Preprocessing</b><br>

        Images are resized to 224 × 224 pixels and
        ResNet50-specific ImageNet preprocessing is applied.

        <br><br>

        <b>3. Data Augmentation</b><br>

        Training images use rotation, shifts, zoom and
        horizontal flipping to improve generalization.

        <br><br>

        <b>4. Transfer Learning</b><br>

        ResNet50 pretrained on ImageNet is used as the
        visual feature extractor.

        <br><br>

        <b>5. Optimizer Screening</b><br>

        Adam, RMSprop and SGD were compared during the
        optimizer screening stage.

        <br><br>

        <b>6. Final Training</b><br>

        ResNet50 is trained using the selected RMSprop
        optimizer with early stopping and adaptive
        learning-rate reduction.

        <br><br>

        <b>7. Evaluation</b><br>

        Accuracy, precision, recall, F1-score and
        confusion matrix are used for evaluation.

        <br><br>

        <b>8. Explainable AI</b><br>

        Grad-CAM is used to visualize image regions
        that contribute strongly to the model's prediction.

        <br><br>

        <b>9. Cloud Persistence</b><br>

        Detection metadata is stored in Supabase so
        that detection history persists across sessions.

        <br><br>

        <b>10. Deployment</b><br>

        The trained ResNet50 model is integrated into
        a Streamlit dashboard for visual inference.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FUTURE SCOPE
# ============================================================

with future_tab:

    st.markdown(
        """
        <div class='project-text'>

        <h3>Future Scope</h3>

        Multi-level freshness estimation instead of
        binary classification.

        <br><br>

        Expansion to additional food categories.

        <br><br>

        Object detection for multiple food items
        within one image.

        <br><br>

        Mobile and edge-device deployment.

        <br><br>

        Continuous learning using newly collected
        food images.

        <br><br>

        Integration with smart storage and inventory
        management systems.

        <br><br>

        Integration with larger food-quality databases
        for long-term monitoring and analytics.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class='info-box'>

    <b>Important:</b>

    This system performs visual AI-based classification.
    A high-confidence prediction does not constitute a
    laboratory food-safety assessment.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class='footer'>

    AI Food Freshness Detection System |
    ResNet50 • Transfer Learning • Computer Vision •
    Batch Inference • Explainable AI • Cloud History

    <br>

    B.Tech Artificial Intelligence & Machine Learning |
    Woxsen University

    </div>
    """,
    unsafe_allow_html=True
)