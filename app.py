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
# MODEL CONFIGURATION — FOOD FRESHNESS DETECTION SYSTEM V2.0
# ------------------------------------------------------------------
# Place both downloaded .keras files either beside app.py or inside
# a folder named "models". Environment-variable overrides are also
# supported (see README.txt).

MODEL_NAME = "MobileNetV2 + ResNet50"
VALIDATOR_MODEL_FILENAME = "mobilenetv2_validator.keras"
FRESHNESS_MODEL_FILENAME = "resnet50_freshness.keras"

IMG_SIZE = (224, 224)
VALIDATOR_THRESHOLD = 0.85
FRESHNESS_THRESHOLD = 0.50

PRETRAINED_WEIGHTS = "ImageNet"
LEARNING_STRATEGY = "Transfer Learning"

# Validator dataset: expanded training and validation data.
VALIDATOR_TRAINING_IMAGES = 20866
VALIDATOR_VALIDATION_IMAGES = 5956
VALIDATOR_TEST_IMAGES = 5373
# Held-out supported-produce freshness test split.
FRESHNESS_TEST_IMAGES = 1757
FOOD101_EXTERNAL_TEST_IMAGES = 2500

# Final V2.0 improved-validator evaluation.
# Acceptance metrics use the probability threshold below; the 3-class
# confusion matrix records argmax predictions before the acceptance gate.
VALIDATOR_PRODUCE_RECALL = 0.9670
VALIDATOR_FALSE_ACCEPT_RATE = 0.0003
VALIDATOR_ACCEPTED_PRECISION = 0.9994
FOOD101_EXTERNAL_REJECTION_RATE = 0.9996
VALIDATOR_3CLASS_ACCURACY = 5310 / 5373

# Rows = actual; columns = predicted (argmax, before confidence gate).
# Labels: Supported Produce, Other Food, Non-Food.
VALIDATOR_CONFUSION_MATRIX = np.array(
    [
        [1728, 16, 13],
        [7, 3094, 15],
        [4, 8, 488]
    ],
    dtype=int
)


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

    .result-rejected {
        border-left: 6px solid #98a2b3;
        background-color: #f7f8fa;
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


def save_batch_to_cloud(results):
    """Save V2 predictions using the existing Supabase table schema."""
    if supabase is None:
        return False

    if not results:
        return True

    records = []
    for result in results:
        records.append(
            {
                "image_name": result["name"],
                "prediction": result["label"],
                "confidence": float(result["confidence"]),
                # These are NULL for rejected images because ResNet50
                # is not called unless the validator accepts the image.
                "fresh_probability": result.get("fresh_probability"),
                "rotten_probability": result.get("rotten_probability"),
                "image_quality": result["quality"],
                "inference_ms": float(result["inference_ms"]),
                "model_name": "MobileNetV2 + ResNet50 V2.0"
            }
        )

    try:
        supabase.table(HISTORY_TABLE).insert(records).execute()
        return True
    except Exception as error:
        st.warning(
            "Could not save detections to Supabase. Make sure the "
            "fresh_probability and rotten_probability columns allow NULL "
            f"for rejected images. Details: {error}"
        )
        return False


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
# LOAD BOTH V2 MODELS
# ============================================================

def resolve_model_path(filename, environment_variable):
    """Find a model beside app.py or in ./models, with env override."""
    override = os.getenv(environment_variable)
    candidates = []

    if override:
        candidates.append(os.path.abspath(os.path.expanduser(override)))

    candidates.extend(
        [
            os.path.join(BASE_DIR, filename),
            os.path.join(BASE_DIR, "models", filename),
        ]
    )

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    searched = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        f"Model file '{filename}' was not found. Checked:\n{searched}\n\n"
        "Download the model from your Kaggle output and place it beside "
        "app.py or inside the models folder."
    )


@st.cache_resource(show_spinner="Loading V2.0 AI models...")
def load_keras_model(path):
    # compile=False avoids loading training-only optimizer/loss state.
    return tf.keras.models.load_model(path, compile=False)


try:
    VALIDATOR_MODEL_PATH = resolve_model_path(
        VALIDATOR_MODEL_FILENAME,
        "FOOD_VALIDATOR_MODEL_PATH"
    )
    FRESHNESS_MODEL_PATH = resolve_model_path(
        FRESHNESS_MODEL_FILENAME,
        "FOOD_FRESHNESS_MODEL_PATH"
    )

    validator_model = load_keras_model(VALIDATOR_MODEL_PATH)
    freshness_model = load_keras_model(FRESHNESS_MODEL_PATH)

    # Existing Grad-CAM functions use the name `model`.
    model = freshness_model

    if int(validator_model.output_shape[-1]) != 3:
        raise ValueError(
            "The MobileNetV2 validator must output 3 class probabilities."
        )

    if int(np.prod(freshness_model.output_shape[1:])) != 1:
        raise ValueError(
            "The ResNet50 freshness model must output one sigmoid probability."
        )

except Exception as error:
    st.error(f"Unable to load the V2.0 models:\n\n{error}")
    st.stop()


# MODEL INFORMATION
# ============================================================

print("=" * 70)
print("FOOD FRESHNESS DETECTION SYSTEM V2.0 — GRAD-CAM FIX")
print("Validator   : MobileNetV2")
print("Freshness   : ResNet50")
print("Validator threshold:", VALIDATOR_THRESHOLD)
print("Freshness threshold:", FRESHNESS_THRESHOLD)
print("Validator file:", VALIDATOR_MODEL_PATH)
print("Freshness file:", FRESHNESS_MODEL_PATH)
print("Input size  :", IMG_SIZE)
print("Validator parameters:", f"{validator_model.count_params():,}")
print("Freshness parameters:", f"{freshness_model.count_params():,}")
print("Supabase    :", "Connected" if supabase is not None else "Not connected")
print("=" * 70)


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
# V2.0 METRIC FORMATTING
# ============================================================

def format_percentage(value):
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"



# MODEL SIZE
# ============================================================

def get_file_size_mb(path):
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return None


VALIDATOR_MODEL_SIZE_MB = get_file_size_mb(VALIDATOR_MODEL_PATH)
FRESHNESS_MODEL_SIZE_MB = get_file_size_mb(FRESHNESS_MODEL_PATH)
MODEL_SIZE_MB = (
    VALIDATOR_MODEL_SIZE_MB + FRESHNESS_MODEL_SIZE_MB
    if VALIDATOR_MODEL_SIZE_MB is not None
    and FRESHNESS_MODEL_SIZE_MB is not None
    else None
)


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
# TWO-STAGE V2.0 PREDICTION PIPELINE
# ============================================================

VALIDATOR_CLASS_NAMES = {
    0: "Supported Produce",
    1: "Other Food",
    2: "Non-Food"
}


def preprocess_validator_image(image):
    image = image.convert("RGB").resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32)
    return tf.keras.applications.mobilenet_v2.preprocess_input(array)


def preprocess_freshness_image(image):
    image = image.convert("RGB").resize(
        IMG_SIZE,
        Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32)
    return tf.keras.applications.resnet50.preprocess_input(array)


def predict_images(image_records):
    """Validate each image first, then classify accepted produce freshness."""
    if not image_records:
        return []

    start_time = time.perf_counter()

    # Stage 1: MobileNetV2 image validation.
    validator_batch = np.stack(
        [preprocess_validator_image(record["image"]) for record in image_records],
        axis=0
    )

    validator_probabilities = validator_model.predict(
        validator_batch,
        batch_size=32,
        verbose=0
    )

    validator_classes = np.argmax(validator_probabilities, axis=1)
    supported_probabilities = validator_probabilities[:, 0]

    accepted_indices = [
        index
        for index, predicted_class in enumerate(validator_classes)
        if (
            int(predicted_class) == 0
            and float(supported_probabilities[index]) >= VALIDATOR_THRESHOLD
        )
    ]

    # Stage 2: ResNet50 runs only on images accepted by MobileNetV2.
    freshness_probabilities = {}

    if accepted_indices:
        freshness_batch = np.stack(
            [
                preprocess_freshness_image(image_records[index]["image"])
                for index in accepted_indices
            ],
            axis=0
        )

        predictions = freshness_model.predict(
            freshness_batch,
            batch_size=32,
            verbose=0
        ).reshape(-1)

        freshness_probabilities = {
            original_index: float(probability)
            for original_index, probability in zip(accepted_indices, predictions)
        }

    elapsed_ms_per_image = (
        (time.perf_counter() - start_time)
        / len(image_records)
        * 1000.0
    )

    results = []

    for index, record in enumerate(image_records):
        predicted_validator_class = int(validator_classes[index])
        validator_class_confidence = float(
            validator_probabilities[index, predicted_validator_class]
        )
        supported_probability = float(supported_probabilities[index])

        quality = assess_image_quality(record["image"])

        if index in freshness_probabilities:
            # Accepted supported produce: freshness probability from ResNet50.
            rotten_probability = freshness_probabilities[index]
            fresh_probability = 1.0 - rotten_probability

            if rotten_probability >= FRESHNESS_THRESHOLD:
                label = "Rotten"
                confidence = rotten_probability
            else:
                label = "Fresh"
                confidence = fresh_probability

            reason = None
            status = "Accepted"

        else:
            # Rejected by validator: do not invoke the freshness model.
            label = "Rejected"
            status = "Rejected"

            if predicted_validator_class == 0:
                reason = "Uncertain Image"
            else:
                reason = VALIDATOR_CLASS_NAMES[predicted_validator_class]

            confidence = validator_class_confidence
            fresh_probability = None
            rotten_probability = None

        results.append(
            {
                "name": record["name"],
                "image": record["image"],
                "label": label,
                "status": status,
                "reason": reason,
                "confidence": float(confidence),
                "validator_label": VALIDATOR_CLASS_NAMES[predicted_validator_class],
                "validator_confidence": validator_class_confidence,
                "supported_probability": supported_probability,
                "fresh_probability": fresh_probability,
                "rotten_probability": rotten_probability,
                "quality": quality["quality"],
                "brightness": quality["brightness"],
                "contrast": quality["contrast"],
                "sharpness": quality["sharpness"],
                "inference_ms": elapsed_ms_per_image,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    return results


# ROBUST GRAD-CAM — supports direct and nested ResNet50 models
# ============================================================

GRADCAM_LAST_CONV_LAYER = "conv5_block3_3_conv"


def _is_convolution_layer(layer):
    """Return True for supported spatial convolution layers."""
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )
    return isinstance(layer, conv_types)


def find_last_conv_layer(current_model):
    """Find the final 4-D convolutional layer, including nested models."""
    # Prefer the explicitly intended final ResNet50 convolution when present.
    try:
        candidate = current_model.get_layer(GRADCAM_LAST_CONV_LAYER)
        if _is_convolution_layer(candidate) and len(candidate.output.shape) == 4:
            return candidate
    except (ValueError, AttributeError):
        pass

    # Search top-level convolution layers first.
    for layer in reversed(getattr(current_model, "layers", [])):
        if _is_convolution_layer(layer):
            try:
                if len(layer.output.shape) == 4:
                    return layer
            except (AttributeError, ValueError):
                continue

    # Search nested Functional / Sequential models recursively.
    for layer in reversed(getattr(current_model, "layers", [])):
        if isinstance(layer, tf.keras.Model):
            found = find_last_conv_layer(layer)
            if found is not None:
                return found

    return None


def _find_nested_backbone_and_conv(current_model):
    """Find a nested model containing the final ResNet50 convolution."""
    for layer in reversed(getattr(current_model, "layers", [])):
        if not isinstance(layer, tf.keras.Model):
            continue

        try:
            conv_layer = layer.get_layer(GRADCAM_LAST_CONV_LAYER)
            if _is_convolution_layer(conv_layer):
                return layer, conv_layer
        except (ValueError, AttributeError):
            pass

        # Fallback for a nested backbone whose final Conv2D has a different name.
        conv_layer = find_last_conv_layer(layer)
        if conv_layer is not None:
            return layer, conv_layer

    return None, None


def build_gradcam_model(current_model):
    """Build a model returning [last-conv activations, classifier output].

    Two supported layouts:
      1. A Functional model with ResNet50 layers at the top level.
      2. A Sequential/wrapper model containing ResNet50 as a nested model,
         followed by a pooling/classification head.

    For the nested layout, the inner backbone is called again on the *outer
    model input*. This creates a single connected graph for both outputs.
    """
    if not getattr(current_model, "inputs", None):
        raise RuntimeError(
            "The freshness model has no symbolic input. "
            "Make sure the loaded model is built before Grad-CAM is requested."
        )

    # Layout 1: convolution is directly part of the complete model graph.
    direct_conv = None
    for layer in reversed(current_model.layers):
        if _is_convolution_layer(layer):
            direct_conv = layer
            break

    if direct_conv is not None:
        grad_model = tf.keras.Model(
            inputs=current_model.inputs,
            outputs=[direct_conv.output, current_model.outputs[0]],
            name="freshness_gradcam_model",
        )
        return grad_model, direct_conv.name

    # Layout 2: ResNet50 is nested inside a Sequential/wrapper classifier.
    backbone, conv_layer = _find_nested_backbone_and_conv(current_model)
    if backbone is None or conv_layer is None:
        raise RuntimeError(
            "Could not locate a ResNet50 backbone containing "
            f"'{GRADCAM_LAST_CONV_LAYER}'."
        )

    if len(current_model.inputs) != 1:
        raise RuntimeError(
            "The current Grad-CAM implementation expects one image input; "
            f"the loaded model has {len(current_model.inputs)} inputs."
        )

    # This inner model exposes both the convolutional activations and the
    # ResNet50 output from the same internal graph.
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[conv_layer.output, backbone.outputs[0]],
        name="resnet50_backbone_gradcam",
    )

    # Rebuild the outer model path from its actual input tensor. This is the
    # important connection missing from the previous implementation.
    outer_x = current_model.inputs[0]
    conv_outputs = None
    backbone_was_called = False

    for layer in current_model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        if layer is backbone:
            conv_outputs, outer_x = backbone_grad_model(
                outer_x,
                training=False,
            )
            backbone_was_called = True
        else:
            outer_x = layer(outer_x)

    if not backbone_was_called or conv_outputs is None:
        raise RuntimeError(
            "The nested ResNet50 backbone was found but could not be "
            "reconnected to the outer freshness-classifier graph."
        )

    grad_model = tf.keras.Model(
        inputs=current_model.inputs,
        outputs=[conv_outputs, outer_x],
        name="freshness_gradcam_model",
    )

    return grad_model, conv_layer.name


# ============================================================
# GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(image, conv_outputs, gradients):
    """Convert convolutional activations and gradients into an image overlay."""
    if conv_outputs.shape.rank != 4:
        raise RuntimeError(
            "Grad-CAM expected a 4-D convolution tensor "
            "(batch, height, width, channels), but received "
            f"shape {conv_outputs.shape}."
        )

    if gradients is None:
        raise RuntimeError("Grad-CAM gradients are None.")

    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    feature_maps = conv_outputs[0]
    channel_weights = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        feature_maps * channel_weights,
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0)

    maximum = tf.reduce_max(heatmap)
    maximum_value = float(maximum.numpy())

    if not np.isfinite(maximum_value) or maximum_value <= 0:
        raise RuntimeError(
            "Grad-CAM produced an empty or non-finite heatmap."
        )

    heatmap = heatmap / (
        maximum + tf.keras.backend.epsilon()
    )
    heatmap = np.asarray(heatmap.numpy(), dtype=np.float32)

    original = image.convert("RGB")
    heatmap_image = Image.fromarray(
        np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    )
    heatmap_image = heatmap_image.resize(
        original.size,
        Image.Resampling.BILINEAR,
    )

    heatmap_array = (
        np.asarray(heatmap_image, dtype=np.float32) / 255.0
    )
    colored_heatmap = Image.fromarray(
        (cm.jet(heatmap_array)[:, :, :3] * 255).astype(np.uint8)
    )

    return Image.blend(original, colored_heatmap, alpha=0.40)


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(image, predicted_label):
    """Generate a Grad-CAM overlay for the selected Fresh/Rotten class."""
    input_tensor = preprocess_resnet50(image)

    try:
        grad_model, conv_layer_name = build_gradcam_model(model)
    except Exception as error:
        raise RuntimeError(
            "Could not build a connected Grad-CAM model. "
            f"Freshness prediction is still available. Details: {error}"
        ) from error

    print("Grad-CAM layer:", conv_layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            input_tensor,
            training=False,
        )

        rotten_probability = predictions[:, 0]
        if predicted_label == "Rotten":
            target_score = rotten_probability
        else:
            target_score = 1.0 - rotten_probability

    gradients = tape.gradient(target_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            f"Gradients are disconnected from '{conv_layer_name}'."
        )

    return create_gradcam_overlay(
        image,
        conv_outputs,
        gradients,
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

        A two-stage AI system. MobileNetV2 first checks
        whether an image contains supported produce.
        ResNet50 then classifies accepted produce as
        <b>Fresh</b> or <b>Rotten</b>.

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

        <b>Stage 1:</b>
        MobileNetV2 image validator (3 classes)

        <br><br>

        <b>Stage 2:</b>
        ResNet50 freshness classifier (2 classes)

        <br><br>

        <b>Learning:</b>
        Transfer Learning with ImageNet weights

        <br><br>

        <b>Input:</b>
        224 × 224 RGB pixels

        <br><br>

        <b>Validator threshold:</b>
        0.85

        <br><br>

        <b>Freshness threshold:</b>
        0.50

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

        <b>Food-101 External Rejection</b>

        <br>

        <span style='font-size:2rem;'>
        {format_percentage(FOOD101_EXTERNAL_REJECTION_RATE)}
        </span>

        <br><br>

        <b>Unsupported False-Accept Rate</b>

        <br>

        <span style='font-size:2rem;'>
        {format_percentage(VALIDATOR_FALSE_ACCEPT_RATE)}
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
    🤖 AI Food Freshness Detection System V2.0
    </div>

    <div class='main-subtitle'>
    Two-Stage Computer Vision System — Image Validation and Freshness Classification
    </div>

    <div class='technology-line'>
    🧠 MobileNetV2 • ResNet50 • Transfer Learning • Explainable AI • Batch Inference
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Supported produce: apple, banana, bell pepper, carrot, cucumber, "
    "mango, orange, potato, strawberry and tomato. Other images are rejected."
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
            "png",
            "webp",
            "bmp"
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
            "Validating images with MobileNetV2 and classifying accepted produce with ResNet50..."
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
        "<div class='section-title'>🎯 V2.0 Prediction Results</div>",
        unsafe_allow_html=True
    )

    for result in st.session_state.analysis_results:
        if result["label"] == "Fresh":
            card_class, icon = "result-fresh", "🟢"
            title = "Fresh"
        elif result["label"] == "Rotten":
            card_class, icon = "result-rotten", "🔴"
            title = "Rotten"
        else:
            card_class, icon = "result-rejected", "⛔"
            title = "Image Rejected"

        result_image_col, result_info_col = st.columns([1, 2])

        with result_image_col:
            st.image(
                result["image"],
                caption=result["name"],
                use_container_width=True
            )

        with result_info_col:
            if result["status"] == "Accepted":
                result_details = f"""
                <div class='result-card {card_class}'>
                    <div class='result-title'>{icon} {title}</div>
                    <div class='result-confidence'>
                        Freshness confidence: {result['confidence'] * 100:.2f}%
                    </div>
                    <div class='result-line'>Image validation: Accepted</div>
                    <div class='result-line'>
                        Supported-produce probability: {result['supported_probability'] * 100:.2f}%
                    </div>
                    <div class='result-line'>
                        Fresh probability: {result['fresh_probability'] * 100:.2f}%
                    </div>
                    <div class='result-line'>
                        Rotten probability: {result['rotten_probability'] * 100:.2f}%
                    </div>
                    <div class='result-line'>Image quality: {result['quality']}</div>
                    <div class='result-line'>
                        Approx. inference: {result['inference_ms']:.2f} ms/image
                    </div>
                </div>
                """
            else:
                result_details = f"""
                <div class='result-card result-rejected'>
                    <div class='result-title'>{icon} {title}</div>
                    <div class='result-confidence'>Reason: {result['reason']}</div>
                    <div class='result-line'>
                        Validator prediction: {result['validator_label']}
                    </div>
                    <div class='result-line'>
                        Validator confidence: {result['validator_confidence'] * 100:.2f}%
                    </div>
                    <div class='result-line'>
                        Supported-produce probability: {result['supported_probability'] * 100:.2f}%
                    </div>
                    <div class='result-line'>Image quality: {result['quality']}</div>
                    <div class='result-line'>
                        ResNet50 was not run on this image.
                    </div>
                    <div class='result-line'>
                        Approx. inference: {result['inference_ms']:.2f} ms/image
                    </div>
                </div>
                """

            st.markdown(result_details, unsafe_allow_html=True)

    accepted_count = sum(
        result["status"] == "Accepted"
        for result in st.session_state.analysis_results
    )
    rejected_count = len(st.session_state.analysis_results) - accepted_count
    accepted_col, rejected_col = st.columns(2)
    accepted_col.metric("Accepted produce images", accepted_count)
    rejected_col.metric("Rejected images", rejected_count)


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

        rejected_count = int(
            (
                history_df["prediction"]
                == "Rejected"
            ).sum()
        )

    else:

        fresh_count = 0

        rotten_count = 0
        rejected_count = 0


    if "confidence" in history_df.columns:

        average_confidence = (
            history_df["confidence"]
            .astype(float)
            .mean()
            * 100
        )

    else:

        average_confidence = 0


    analytics_col1, analytics_col2, analytics_col3, analytics_col4, analytics_col5 = (
        st.columns(5)
    )

    with analytics_col1:
        st.metric("Total Detections", total_predictions)

    with analytics_col2:
        st.metric("Fresh", fresh_count)

    with analytics_col3:
        st.metric("Rotten", rotten_count)

    with analytics_col4:
        st.metric("Rejected", rejected_count)

    with analytics_col5:
        st.metric("Average Confidence", f"{average_confidence:.2f}%")

    st.caption(
        "Average confidence combines the freshness confidence for accepted "
        "produce and the validator confidence for rejected images."
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
# MODEL PERFORMANCE — IMPROVED V2.0 VALIDATOR TEST RESULTS
# ============================================================

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='section-title'>📈 Model Performance — Improved V2.0 Validator</div>",
    unsafe_allow_html=True
)

performance_columns = st.columns(4)
performance_metrics = [
    ("Validator 3-Class Accuracy", format_percentage(VALIDATOR_3CLASS_ACCURACY)),
    ("Produce Recall", format_percentage(VALIDATOR_PRODUCE_RECALL)),
    ("Accepted Precision", format_percentage(VALIDATOR_ACCEPTED_PRECISION)),
    ("Unsupported False-Accept Rate", format_percentage(VALIDATOR_FALSE_ACCEPT_RATE)),
]

for column, (label, value) in zip(performance_columns, performance_metrics):
    with column:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown(
    "<div class='subsection-title'>🎯 Validator Three-Class Confusion Matrix</div>",
    unsafe_allow_html=True
)
st.caption(
    "Original held-out validator test set: 5,373 images. "
    "The displayed operating metrics were evaluated at threshold 0.80; "
    "the deployed application now uses a stricter 0.85 acceptance threshold. "
    "The confusion matrix shows three-class argmax predictions."
)
validator_cm_df = pd.DataFrame(
    VALIDATOR_CONFUSION_MATRIX,
    index=["Actual Supported Produce", "Actual Other Food", "Actual Non-Food"],
    columns=["Predicted Supported Produce", "Predicted Other Food", "Predicted Non-Food"]
)
st.dataframe(validator_cm_df, use_container_width=True)

external_cols = st.columns(2)
external_cols[0].metric(
    "Food-101 External Rejection",
    format_percentage(FOOD101_EXTERNAL_REJECTION_RATE)
)
external_cols[1].metric(
    "Food-101 External Test Images",
    f"{FOOD101_EXTERNAL_TEST_IMAGES:,}"
)
st.caption(
    "Food-101 official external test: 2,499 of 2,500 prepared-food images "
    "were rejected; one donut image was incorrectly accepted."
)


# TECHNICAL SPECIFICATIONS
# ============================================================

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='section-title'>⚙️ Technical Specifications</div>",
    unsafe_allow_html=True
)

technical_col1, technical_col2 = st.columns(2)

with technical_col1:
    st.markdown(
        """
        <div class='metric-card'>
            <h3>🧠 Stage 1 — Image Validator</h3>
            <p><b>Architecture:</b> MobileNetV2</p>
            <p><b>Preprocessing:</b> MobileNetV2 preprocess_input</p>
            <p><b>Input:</b> 224 × 224 RGB pixels</p>
            <p><b>Output:</b> 3-class Softmax</p>
            <p><b>Classes:</b> Supported Produce / Other Food / Non-Food</p>
            <p><b>Acceptance rule:</b> Supported Produce class and probability ≥ 0.85</p>
            <p><b>Training optimizer:</b> Adam (learning rate 0.001)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with technical_col2:
    total_size_text = (
        f"{MODEL_SIZE_MB:.2f} MB" if MODEL_SIZE_MB is not None else "—"
    )
    validator_size_text = (
        f"{VALIDATOR_MODEL_SIZE_MB:.2f} MB"
        if VALIDATOR_MODEL_SIZE_MB is not None else "—"
    )
    freshness_size_text = (
        f"{FRESHNESS_MODEL_SIZE_MB:.2f} MB"
        if FRESHNESS_MODEL_SIZE_MB is not None else "—"
    )
    st.markdown(
        f"""
        <div class='metric-card'>
            <h3>🍎 Stage 2 — Freshness Classifier</h3>
            <p><b>Architecture:</b> ResNet50</p>
            <p><b>Preprocessing:</b> ResNet50 preprocess_input</p>
            <p><b>Input:</b> 224 × 224 RGB pixels</p>
            <p><b>Output:</b> 1 sigmoid probability</p>
            <p><b>Classes:</b> Fresh / Rotten</p>
            <p><b>Decision rule:</b> Rotten probability ≥ 0.50 means Rotten</p>
            <p><b>Training optimizer:</b> Adam (learning rate 0.00003)</p>
            <p><b>Fine-tuning:</b> Last 50 backbone layers; dropout 0.20</p>
            <p><b>Validator test set:</b> {VALIDATOR_TEST_IMAGES:,} images</p>
            <p><b>Freshness test set:</b> {FRESHNESS_TEST_IMAGES:,} images</p>
            <p><b>Validator model size:</b> {validator_size_text}</p>
            <p><b>Freshness model size:</b> {freshness_size_text}</p>
            <p><b>Total model size:</b> {total_size_text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# AI PROCESSING PIPELINE
# ============================================================

st.markdown(
    "<div class='section-divider'></div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='section-title'>🔄 V2.0 AI Processing Pipeline</div>",
    unsafe_allow_html=True
)

pipeline_items = [
    ("📷", "Input", "Upload or capture food images"),
    ("📐", "Preprocess", "Resize to 224 × 224 RGB"),
    ("🔍", "Validate", "MobileNetV2 checks 3 image classes"),
    ("✅", "Accept / Reject", "Accept supported produce at threshold 0.85"),
    ("🧠", "Freshness", "ResNet50 predicts Fresh or Rotten"),
    ("📊", "Results", "Confidence, history and Grad-CAM"),
]

pipeline_columns = st.columns(6)
for column, item in zip(pipeline_columns, pipeline_items):
    icon, title, description = item
    with column:
        st.markdown(
            f"""
            <div class='pipeline-card'>
                <div class='pipeline-icon'>{icon}</div>
                <div class='pipeline-title'>{title}</div>
                <div class='pipeline-description'>{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


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
        for result in st.session_state.analysis_results
        if result["label"] in ("Fresh", "Rotten")
    ]


    if not gradcam_options:
        st.info(
            "Grad-CAM is available only for images accepted by the validator. "
            "The current batch contains no accepted produce images."
        )
        selected_gradcam = None
    else:
        selected_gradcam = st.selectbox(
            "Select an accepted produce image to explain",
            gradcam_options
        )


    selected_result = next(
        (
            result
            for result in st.session_state.analysis_results
            if result["name"] == selected_gradcam
        ),
        None
    ) if selected_gradcam is not None else None


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

        The objective of Food Freshness Detection System V2.0
        is to validate food images before performing freshness
        classification.

        <br><br>

        <b>MobileNetV2</b> first classifies an image as
        Supported Produce, Other Food, or Non-Food.
        Only supported-produce images that meet the 0.85
        acceptance threshold proceed to <b>ResNet50</b>,
        which predicts Fresh or Rotten.

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

        <b>Image Validator:</b>
        MobileNetV2 (3-class Softmax)

        <br><br>

        <b>Freshness Classifier:</b>
        ResNet50 (binary Sigmoid)

        <br><br>

        <b>Learning Strategy:</b>
        Transfer Learning

        <br><br>

        <b>Pretrained Weights:</b>
        ImageNet

        <br><br>

        <b>Training Optimizer:</b>
        Adam for the selected V2 models

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

        Supported produce images are combined with external
        Other Food and Non-Food examples for the validator.
        The freshness dataset retains Fresh and Rotten labels.

        <br><br>

        <b>2. Image Preprocessing</b><br>

        Images are resized to 224 × 224 pixels. Each model
        receives its own architecture-specific preprocessing.

        <br><br>

        <b>3. Data Augmentation</b><br>

        Training images use rotation, shifts, zoom and
        horizontal flipping to improve generalization.

        <br><br>

        <b>4. Transfer Learning</b><br>

        MobileNetV2 validates the image category. ResNet50,
        pretrained on ImageNet, classifies freshness for
        accepted produce images.

        <br><br>

        <b>5. Optimizer Screening</b><br>

        Adam, RMSprop and SGD were compared during the
        optimizer screening stage.

        <br><br>

        <b>6. Final Training</b><br>

        The MobileNetV2 validator and ResNet50 freshness
        classifier are trained as separate models with
        their selected configurations.

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

        Both trained models are integrated into Streamlit.
        Unsupported or uncertain images are rejected before
        the freshness classifier is called.

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

    AI Food Freshness Detection System V2.0 |
    MobileNetV2 • ResNet50 • Transfer Learning •
    Image Validation • Batch Inference • Explainable AI • Cloud History

    <br>

    B.Tech Artificial Intelligence & Machine Learning |
    Woxsen University

    </div>
    """,
    unsafe_allow_html=True
)