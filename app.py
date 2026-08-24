import os
from datetime import datetime

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

from PIL import Image
import matplotlib.cm as cm


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Food Freshness Detection",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "best_mobilenetv2.keras"

HISTORY_FILE = "prediction_history.csv"

IMAGE_SIZE = (224, 224)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:

    st.session_state.prediction_history = []


if "last_batch_id" not in st.session_state:

    st.session_state.last_batch_id = None


if "camera_batch" not in st.session_state:

    st.session_state.camera_batch = []


if "camera_names" not in st.session_state:

    st.session_state.camera_names = []


# ============================================================
# LOAD PREDICTION HISTORY
# ============================================================

def load_prediction_history():

    if not os.path.exists(
        HISTORY_FILE
    ):

        return []


    try:

        df = pd.read_csv(
            HISTORY_FILE
        )


        if df.empty:

            return []


        return df.to_dict(
            orient="records"
        )


    except Exception:

        return []


if not st.session_state.prediction_history:

    st.session_state.prediction_history = (
        load_prediction_history()
    )


# ============================================================
# SAVE PREDICTION HISTORY
# ============================================================

def save_prediction_history():

    history = (
        st.session_state.prediction_history
    )


    if not history:

        return


    try:

        df = pd.DataFrame(
            history
        )


        df.to_csv(
            HISTORY_FILE,
            index=False
        )


    except Exception as e:

        st.warning(
            f"Could not save prediction history: {e}"
        )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


try:

    model = load_model()


except Exception as e:

    st.error(
        "❌ Unable to load the trained MobileNetV2 model."
    )


    st.write(
        f"Expected model location: `{MODEL_PATH}`"
    )


    st.exception(e)

    st.stop()


# ============================================================
# SINGLE IMAGE PREDICTION
# ============================================================

def predict_freshness(image):

    img = image.convert(
        "RGB"
    )


    img = img.resize(
        IMAGE_SIZE
    )


    img_array = np.array(
        img,
        dtype=np.float32
    )


    img_array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(
            img_array
        )
    )


    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]


    rotten_probability = float(
        prediction
    )


    fresh_probability = float(
        1.0 - prediction
    )


    fresh_percentage = (
        fresh_probability * 100
    )


    rotten_percentage = (
        rotten_probability * 100
    )


    if prediction >= 0.5:

        label = "Rotten"

        confidence = rotten_percentage


    else:

        label = "Fresh"

        confidence = fresh_percentage


    return {

        "label": label,

        "confidence": confidence,

        "fresh_probability":
            fresh_percentage,

        "rotten_probability":
            rotten_percentage
    }


# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_multiple_images(images):

    image_arrays = []

    valid_images = []


    for image in images:

        try:

            img = image.convert(
                "RGB"
            )


            img = img.resize(
                IMAGE_SIZE
            )


            img_array = np.array(
                img,
                dtype=np.float32
            )


            img_array = (
                tf.keras.applications
                .mobilenet_v2
                .preprocess_input(
                    img_array
                )
            )


            image_arrays.append(
                img_array
            )


            valid_images.append(
                img
            )


        except Exception:

            continue


    if not image_arrays:

        return []


    # --------------------------------------------------------
    # CREATE BATCH
    # --------------------------------------------------------

    batch = np.stack(
        image_arrays,
        axis=0
    )


    # --------------------------------------------------------
    # BATCH INFERENCE
    # --------------------------------------------------------

    predictions = model.predict(
        batch,
        verbose=0
    ).flatten()


    results = []


    # --------------------------------------------------------
    # PROCESS RESULTS
    # --------------------------------------------------------

    for index, prediction in enumerate(
        predictions
    ):

        rotten_probability = float(
            prediction
        )


        fresh_probability = float(
            1.0 - prediction
        )


        fresh_percentage = (
            fresh_probability * 100
        )


        rotten_percentage = (
            rotten_probability * 100
        )


        if prediction >= 0.5:

            label = "Rotten"

            confidence = (
                rotten_percentage
            )


        else:

            label = "Fresh"

            confidence = (
                fresh_percentage
            )


        reliability, _ = (
            get_reliability(
                confidence
            )
        )


        results.append({

            "Image": index + 1,

            "Prediction": label,

            "Confidence (%)": round(
                confidence,
                2
            ),

            "Fresh (%)": round(
                fresh_percentage,
                2
            ),

            "Rotten (%)": round(
                rotten_percentage,
                2
            ),

            "Reliability": reliability

        })


    return results


# ============================================================
# RELIABILITY
# ============================================================

def get_reliability(
    confidence
):

    if confidence >= 90:

        return (
            "Very High",
            "The model is highly confident "
            "in this prediction."
        )


    elif confidence >= 75:

        return (
            "High",
            "The model has strong confidence "
            "in this prediction."
        )


    elif confidence >= 60:

        return (
            "Moderate",
            "The prediction has moderate confidence. "
            "A clearer image may improve reliability."
        )


    else:

        return (
            "Low",
            "The model is uncertain. "
            "Consider providing a clearer image."
        )


# ============================================================
# IMAGE QUALITY ASSESSMENT
# ============================================================

def assess_image_quality(
    image
):

    width, height = image.size


    total_pixels = (
        width * height
    )


    # --------------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------------

    if total_pixels >= 500000:

        resolution_status = "Good"


    elif total_pixels >= 150000:

        resolution_status = "Moderate"


    else:

        resolution_status = "Low"


    # --------------------------------------------------------
    # BRIGHTNESS
    # --------------------------------------------------------

    grayscale = image.convert(
        "L"
    )


    gray_array = np.array(
        grayscale,
        dtype=np.float64
    )


    brightness = float(
        np.mean(
            gray_array
        )
    )


    if brightness < 50:

        brightness_status = "Too Dark"


    elif brightness < 85:

        brightness_status = "Dark"


    elif brightness > 220:

        brightness_status = "Very Bright"


    elif brightness > 190:

        brightness_status = "Bright"


    else:

        brightness_status = "Good"


    # --------------------------------------------------------
    # SHARPNESS
    # --------------------------------------------------------

    laplacian = cv2.Laplacian(
        gray_array,
        cv2.CV_64F
    )


    sharpness = float(
        laplacian.var()
    )


    if sharpness < 50:

        sharpness_status = "Blurry"


    elif sharpness < 150:

        sharpness_status = "Moderate"


    else:

        sharpness_status = "Sharp"


    return {

        "width": width,

        "height": height,

        "brightness": brightness,

        "brightness_status":
            brightness_status,

        "sharpness": sharpness,

        "sharpness_status":
            sharpness_status,

        "resolution_status":
            resolution_status

    }


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(
    current_model
):

    for layer in reversed(
        current_model.layers
    ):

        if isinstance(
            layer,
            tf.keras.Model
        ):

            nested_layer = (
                find_last_conv_layer(
                    layer
                )
            )


            if nested_layer is not None:

                return nested_layer


        try:

            if isinstance(
                layer,
                tf.keras.layers.InputLayer
            ):

                continue


            output_shape = (
                layer.output.shape
            )


            if (
                len(output_shape) == 4
                and
                output_shape[-1] is not None
            ):

                return layer


        except Exception:

            continue


    return None


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam_heatmap(
    image,
    trained_model
):

    img = image.convert(
        "RGB"
    )


    img = img.resize(
        IMAGE_SIZE
    )


    img_array = np.array(
        img,
        dtype=np.float32
    )


    img_array = (
        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(
            img_array
        )
    )


    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    target_layer = (
        find_last_conv_layer(
            trained_model
        )
    )


    if target_layer is None:

        raise ValueError(
            "No suitable convolutional feature "
            "layer was found."
        )


    try:

        grad_model = tf.keras.models.Model(

            inputs=trained_model.inputs,

            outputs=[
                target_layer.output,
                trained_model.output
            ]

        )


    except Exception as e:

        raise ValueError(
            "The trained model structure does not "
            "allow direct Grad-CAM extraction."
        ) from e


    with tf.GradientTape() as tape:

        conv_outputs, predictions = (
            grad_model(
                img_array,
                training=False
            )
        )


        prediction = predictions[:, 0]


        if prediction[0] >= 0.5:

            class_score = prediction


        else:

            class_score = (
                1.0 - prediction
            )


    grads = tape.gradient(
        class_score,
        conv_outputs
    )


    if grads is None:

        raise ValueError(
            "Gradients could not be calculated."
        )


    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )


    conv_outputs = conv_outputs[0]


    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )


    heatmap = tf.maximum(
        heatmap,
        0
    )


    max_value = tf.reduce_max(
        heatmap
    )


    if float(max_value) <= 0:

        raise ValueError(
            "Grad-CAM produced an empty heatmap."
        )


    heatmap = (
        heatmap / max_value
    )


    return heatmap.numpy()


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    heatmap,
    alpha=0.45
):

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )


    colored_heatmap = cm.jet(
        heatmap_uint8
    )[:, :, :3]


    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )


    heatmap_image = Image.fromarray(
        colored_heatmap
    )


    heatmap_image = heatmap_image.resize(
        image.size,
        Image.Resampling.BILINEAR
    )


    original = image.convert(
        "RGB"
    )


    overlay = Image.blend(
        original,
        heatmap_image,
        alpha
    )


    return overlay


# ============================================================
# CREATE REPORT
# ============================================================

def create_prediction_report(
    filename,
    label,
    confidence,
    fresh_probability,
    rotten_probability,
    quality,
    reliability
):

    report = []


    report.append(
        "AI FOOD FRESHNESS DETECTION REPORT"
    )


    report.append(
        "=" * 45
    )


    report.append("")


    report.append(
        "Generated: "
        + datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    report.append(
        f"Input Image: {filename}"
    )


    report.append("")


    report.append(
        "PREDICTION"
    )


    report.append(
        "-" * 20
    )


    report.append(
        f"Classification: {label}"
    )


    report.append(
        f"Confidence: {confidence:.2f}%"
    )


    report.append(
        f"Reliability: {reliability}"
    )


    report.append("")


    report.append(
        "PROBABILITY BREAKDOWN"
    )


    report.append(
        "-" * 20
    )


    report.append(
        f"Fresh: {fresh_probability:.2f}%"
    )


    report.append(
        f"Rotten: {rotten_probability:.2f}%"
    )


    report.append("")


    report.append(
        "IMAGE QUALITY"
    )


    report.append(
        "-" * 20
    )


    report.append(
        f"Resolution: "
        f"{quality['width']} x "
        f"{quality['height']}"
    )


    report.append(
        f"Resolution Status: "
        f"{quality['resolution_status']}"
    )


    report.append(
        f"Brightness: "
        f"{quality['brightness']:.2f}"
    )


    report.append(
        f"Brightness Status: "
        f"{quality['brightness_status']}"
    )


    report.append(
        f"Sharpness: "
        f"{quality['sharpness']:.2f}"
    )


    report.append(
        f"Sharpness Status: "
        f"{quality['sharpness_status']}"
    )


    report.append("")


    report.append(
        "MODEL INFORMATION"
    )


    report.append(
        "-" * 20
    )


    report.append(
        "Architecture: MobileNetV2"
    )


    report.append(
        "Learning: Transfer Learning"
    )


    report.append(
        "Input Size: 224 x 224"
    )


    report.append(
        "Classes: Fresh, Rotten"
    )


    report.append(
        "Test Accuracy: 97.22%"
    )


    report.append(
        "Best Validation Accuracy: 98.11%"
    )


    report.append("")


    report.append(
        "DISCLAIMER"
    )


    report.append(
        "-" * 20
    )


    report.append(
        "This system provides AI-based visual "
        "classification and should not be considered "
        "a substitute for professional food-safety "
        "inspection."
    )


    return "\n".join(
        report
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🤖 AI Food Vision"
    )


    st.divider()


    st.subheader(
        "About the System"
    )


    st.write(
        """
        An AI-powered computer vision system that
        classifies food images as **Fresh** or
        **Rotten** using MobileNetV2 transfer learning.
        """
    )


    st.success(
        "🟢 Fresh"
    )


    st.error(
        "🔴 Rotten"
    )


    st.divider()


    st.subheader(
        "🧠 Model"
    )


    st.write(
        "**Architecture:** MobileNetV2"
    )


    st.write(
        "**Learning:** Transfer Learning"
    )


    st.write(
        "**Pretrained Weights:** ImageNet"
    )


    st.write(
        "**Input:** 224 × 224 pixels"
    )


    st.write(
        "**Classes:** 2"
    )


    st.divider()


    st.subheader(
        "📊 Test Performance"
    )


    st.metric(
        "Test Accuracy",
        "97.22%"
    )


    st.metric(
        "Best Validation Accuracy",
        "98.11%"
    )


    st.divider()


    st.subheader(
        "🏫 Project"
    )


    st.write(
        "**Woxsen University**"
    )


    st.caption(
        "B.Tech Artificial Intelligence & Machine Learning"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🤖 AI Food Freshness Detection"
)


st.subheader(
    "Computer Vision Based Food Quality Classification "
    "using MobileNetV2 Transfer Learning"
)


st.write(
    "🧠 Deep Learning  •  Computer Vision  •  "
    "Artificial Intelligence  •  Explainable AI"
)


st.divider()


# ============================================================
# INPUT METHOD
# ============================================================

st.header(
    "📷 Food Image Input"
)


input_method = st.radio(

    "Select input method",

    [
        "Upload Multiple Images",
        "Capture Multiple Photos"
    ],

    horizontal=True
)


# ============================================================
# MULTIPLE FILE UPLOAD
# ============================================================

if input_method == "Upload Multiple Images":

    uploaded_files = st.file_uploader(

        "Choose one or more food images",

        type=[
            "jpg",
            "jpeg",
            "png"
        ],

        accept_multiple_files=True,

        help=(
            "Upload 10, 20, 50 or more food images "
            "at once."
        )
    )


    if uploaded_files:

        st.success(
            f"📷 {len(uploaded_files)} "
            f"image(s) selected."
        )


# ============================================================
# MULTIPLE CAMERA CAPTURE
# ============================================================

else:

    st.subheader(
        "📸 Camera Batch Capture"
    )


    st.write(
        """
        Take a food photograph, add it to the batch,
        then take another photograph. You can continue
        adding photos until your complete batch is ready.
        """
    )


    camera_file = st.camera_input(
        "Take a food photograph"
    )


    if camera_file is not None:

        # ----------------------------------------------------
        # ADD CURRENT CAMERA PHOTO
        # ----------------------------------------------------

        if st.button(
            "➕ Add Photo to Batch",
            type="primary"
        ):

            try:

                camera_image = Image.open(
                    camera_file
                ).convert(
                    "RGB"
                )


                # Create unique name

                photo_number = (
                    len(
                        st.session_state.camera_batch
                    ) + 1
                )


                camera_name = (
                    f"Camera_Photo_"
                    f"{photo_number}.jpg"
                )


                st.session_state.camera_batch.append(
                    camera_image
                )


                st.session_state.camera_names.append(
                    camera_name
                )


                st.success(
                    f"✅ Photo {photo_number} "
                    f"added to the batch."
                )


            except Exception as e:

                st.error(
                    f"Could not add photo: {e}"
                )


    # --------------------------------------------------------
    # CAMERA BATCH STATUS
    # --------------------------------------------------------

    camera_count = len(
        st.session_state.camera_batch
    )


    if camera_count > 0:

        st.divider()


        st.subheader(
            "📦 Current Camera Batch"
        )


        st.info(
            f"📸 **{camera_count} photo(s)** "
            f"currently in your camera batch."
        )


        # ----------------------------------------------------
        # CAMERA BATCH PREVIEW
        # ----------------------------------------------------

        preview_count = min(
            camera_count,
            12
        )


        preview_columns = st.columns(
            min(
                preview_count,
                4
            )
        )


        for index in range(
            preview_count
        ):

            column = preview_columns[
                index % 4
            ]


            with column:

                st.image(
                    st.session_state.camera_batch[
                        index
                    ],
                    use_container_width=True
                )


                st.caption(
                    st.session_state.camera_names[
                        index
                    ]
                )


        if camera_count > 12:

            st.caption(
                f"+ {camera_count - 12} "
                f"additional photos in the batch."
            )


        # ----------------------------------------------------
        # CAMERA BATCH CONTROLS
        # ----------------------------------------------------

        control1, control2 = st.columns(2)


        with control1:

            analyze_camera = st.button(
                "🚀 Analyze Camera Batch",
                type="primary",
                use_container_width=True
            )


        with control2:

            clear_camera = st.button(
                "🗑️ Clear Camera Batch",
                use_container_width=True
            )


        if clear_camera:

            st.session_state.camera_batch = []

            st.session_state.camera_names = []

            st.session_state.last_batch_id = None

            st.rerun()


        # ----------------------------------------------------
        # SET CAMERA BATCH FOR PROCESSING
        # ----------------------------------------------------

        if analyze_camera:

            st.session_state.active_camera_batch = True

        else:

            if (
                "active_camera_batch"
                not in st.session_state
            ):

                st.session_state.active_camera_batch = False


# ============================================================
# DETERMINE ACTIVE IMAGES
# ============================================================

active_images = []

active_filenames = []


# ------------------------------------------------------------
# UPLOAD MODE
# ------------------------------------------------------------

if input_method == "Upload Multiple Images":

    if uploaded_files:

        for uploaded_file in uploaded_files:

            try:

                img = Image.open(
                    uploaded_file
                ).convert(
                    "RGB"
                )


                active_images.append(
                    img
                )


                active_filenames.append(
                    uploaded_file.name
                )


            except Exception as e:

                st.warning(
                    f"Could not read "
                    f"{uploaded_file.name}: {e}"
                )


# ------------------------------------------------------------
# CAMERA MODE
# ------------------------------------------------------------

else:

    if (
        "active_camera_batch"
        in st.session_state
        and
        st.session_state.active_camera_batch
    ):

        active_images = (
            st.session_state.camera_batch
        )


        active_filenames = (
            st.session_state.camera_names
        )


# ============================================================
# BATCH ANALYSIS
# ============================================================

if active_images:

    st.divider()


    st.header(
        "🔬 Food Freshness Analysis"
    )


    number_of_images = len(
        active_images
    )


    st.info(
        f"📷 **{number_of_images} image(s)** "
        f"ready for AI freshness analysis."
    )


    if number_of_images >= 10:

        st.success(
            f"🚀 Batch inference enabled — "
            f"{number_of_images} images will be "
            f"processed together."
        )


    if number_of_images > 100:

        st.warning(
            """
            You have selected more than 100 images.
            Processing may require additional memory.
            """
        )


    # ========================================================
    # RUN BATCH MODEL
    # ========================================================

    with st.spinner(
        f"Analyzing {number_of_images} images..."
    ):

        results = predict_multiple_images(
            active_images
        )


    if not results:

        st.error(
            "No valid images could be processed."
        )

        st.stop()


    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    results_df["Filename"] = (
        active_filenames[
            :len(results_df)
        ]
    )


    results_df = results_df[
        [
            "Image",
            "Filename",
            "Prediction",
            "Confidence (%)",
            "Fresh (%)",
            "Rotten (%)",
            "Reliability"
        ]
    ]


    # ========================================================
    # BATCH SUMMARY
    # ========================================================

    st.subheader(
        "📊 Batch Analysis Summary"
    )


    total = len(
        results_df
    )


    fresh_count = int(
        (
            results_df["Prediction"]
            == "Fresh"
        ).sum()
    )


    rotten_count = int(
        (
            results_df["Prediction"]
            == "Rotten"
        ).sum()
    )


    average_confidence = float(
        results_df[
            "Confidence (%)"
        ].mean()
    )


    summary1, summary2, summary3, summary4 = (
        st.columns(4)
    )


    with summary1:

        st.metric(
            "📷 Total Images",
            total
        )


    with summary2:

        st.metric(
            "🟢 Fresh",
            fresh_count
        )


    with summary3:

        st.metric(
            "🔴 Rotten",
            rotten_count
        )


    with summary4:

        st.metric(
            "🎯 Avg. Confidence",
            f"{average_confidence:.2f}%"
        )


    # ========================================================
    # FRESH / ROTTEN PERCENTAGE
    # ========================================================

    fresh_ratio = (
        fresh_count / total
    ) * 100


    rotten_ratio = (
        rotten_count / total
    ) * 100


    ratio1, ratio2 = st.columns(2)


    with ratio1:

        st.metric(
            "🟢 Fresh Percentage",
            f"{fresh_ratio:.2f}%"
        )


    with ratio2:

        st.metric(
            "🔴 Rotten Percentage",
            f"{rotten_ratio:.2f}%"
        )


    # ========================================================
    # RESULTS TABLE
    # ========================================================

    st.subheader(
        "📋 Individual Predictions"
    )


    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DISTRIBUTION CHART
    # ========================================================

    st.subheader(
        "📊 Fresh vs Rotten Distribution"
    )


    distribution = pd.DataFrame(

        {
            "Class": [
                "Fresh",
                "Rotten"
            ],

            "Images": [
                fresh_count,
                rotten_count
            ]
        }
    )


    st.bar_chart(
        distribution.set_index(
            "Class"
        )
    )


    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    st.subheader(
        "📥 Export Results"
    )


    csv_data = (
        results_df
        .to_csv(
            index=False
        )
    )


    st.download_button(

        label="📥 Download Batch Results CSV",

        data=csv_data,

        file_name=(
            "food_freshness_batch_results.csv"
        ),

        mime="text/csv"
    )


    # ========================================================
    # VISUAL RESULTS
    # ========================================================

    st.divider()


    st.subheader(
        "🖼️ Visual Results"
    )


    st.write(
        "Every image in the batch is shown with "
        "its individual AI prediction."
    )


    for start in range(
        0,
        len(active_images),
        4
    ):

        row_images = active_images[
            start:start + 4
        ]


        row_filenames = active_filenames[
            start:start + 4
        ]


        row_results = results[
            start:start + 4
        ]


        columns = st.columns(
            len(row_images)
        )


        for col, img, filename, result in zip(
            columns,
            row_images,
            row_filenames,
            row_results
        ):

            with col:

                st.image(
                    img,
                    use_container_width=True
                )


                if (
                    result["Prediction"]
                    == "Fresh"
                ):

                    st.success(
                        "🟢 FRESH"
                    )

                else:

                    st.error(
                        "🔴 ROTTEN"
                    )


                st.write(
                    f"**Confidence:** "
                    f"{result['Confidence (%)']:.2f}%"
                )


                st.caption(
                    filename
                )


    # ========================================================
    # SELECT IMAGE
    # ========================================================

    st.divider()


    st.header(
        "🔎 Detailed Image Analysis"
    )


    selected_index = st.selectbox(

        "Select an image for detailed analysis",

        range(
            len(active_images)
        ),

        format_func=lambda x:
            active_filenames[x]
    )


    selected_image = active_images[
        selected_index
    ]


    selected_result = results[
        selected_index
    ]


    # ========================================================
    # IMAGE QUALITY
    # ========================================================

    st.subheader(
        "🔍 Image Quality Assessment"
    )


    quality = assess_image_quality(
        selected_image
    )


    q1, q2, q3 = st.columns(3)


    with q1:

        if (
            quality["resolution_status"]
            == "Good"
        ):

            st.success(
                "✓ Good Resolution"
            )


        elif (
            quality["resolution_status"]
            == "Moderate"
        ):

            st.warning(
                "⚠ Moderate Resolution"
            )


        else:

            st.error(
                "✕ Low Resolution"
            )


        st.caption(
            f'{quality["width"]} × '
            f'{quality["height"]} pixels'
        )


    with q2:

        if (
            quality["brightness_status"]
            == "Good"
        ):

            st.success(
                "✓ Good Brightness"
            )


        elif (
            quality["brightness_status"]
            in [
                "Dark",
                "Bright"
            ]
        ):

            st.warning(
                "⚠ Moderate Lighting"
            )


        else:

            st.error(
                "✕ Poor Lighting"
            )


        st.caption(
            f'Brightness: '
            f'{quality["brightness"]:.1f}'
        )


    with q3:

        if (
            quality["sharpness_status"]
            == "Sharp"
        ):

            st.success(
                "✓ Sharp Image"
            )


        elif (
            quality["sharpness_status"]
            == "Moderate"
        ):

            st.warning(
                "⚠ Moderate Sharpness"
            )


        else:

            st.error(
                "✕ Image May Be Blurry"
            )


        st.caption(
            f'Sharpness: '
            f'{quality["sharpness"]:.1f}'
        )


    # ========================================================
    # SELECTED IMAGE PREDICTION
    # ========================================================

    detail1, detail2 = st.columns(
        [1.15, 1]
    )


    with detail1:

        st.subheader(
            "📷 Selected Image"
        )


        st.image(
            selected_image,
            use_container_width=True
        )


    with detail2:

        st.subheader(
            "🧠 AI Prediction"
        )


        if (
            selected_result["Prediction"]
            == "Fresh"
        ):

            st.success(
                "🟢 FRESH FOOD"
            )


        else:

            st.error(
                "🔴 ROTTEN FOOD"
            )


        st.metric(
            "Confidence",
            f'{selected_result["Confidence (%)"]:.2f}%'
        )


        st.progress(
            min(
                max(
                    selected_result[
                        "Confidence (%)"
                    ] / 100,
                    0.0
                ),
                1.0
            )
        )


        st.write(
            f"**Fresh probability:** "
            f'{selected_result["Fresh (%)"]:.2f}%'
        )


        st.write(
            f"**Rotten probability:** "
            f'{selected_result["Rotten (%)"]:.2f}%'
        )


        st.write(
            f"**Reliability:** "
            f'{selected_result["Reliability"]}'
        )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.divider()


    st.header(
        "🧠 Explainable AI — Grad-CAM"
    )


    st.write(
        """
        Grad-CAM visualizes the regions of the selected
        food image that contributed to the model's
        classification decision.
        """
    )


    try:

        heatmap = make_gradcam_heatmap(
            selected_image,
            model
        )


        gradcam_image = (
            create_gradcam_overlay(
                selected_image,
                heatmap
            )
        )


        gc1, gc2 = st.columns(2)


        with gc1:

            st.subheader(
                "📷 Original Image"
            )


            st.image(
                selected_image,
                use_container_width=True
            )


        with gc2:

            st.subheader(
                "🔥 Model Attention"
            )


            st.image(
                gradcam_image,
                use_container_width=True
            )


        if (
            selected_result["Prediction"]
            == "Rotten"
        ):

            st.warning(
                """
                🔴 The highlighted regions represent
                areas that contributed strongly to the
                Rotten prediction.
                """
            )


        else:

            st.success(
                """
                🟢 The highlighted regions represent
                areas that contributed strongly to the
                Fresh prediction.
                """
            )


        st.caption(
            """
            Grad-CAM is an interpretability technique.
            Highlighted regions indicate model attention
            and do not independently prove food spoilage.
            """
        )


    except Exception as e:

        st.warning(
            "⚠️ Grad-CAM could not be generated "
            "for this model structure."
        )


        st.caption(
            f"Technical details: {str(e)}"
        )


    # ========================================================
    # SAVE BATCH TO HISTORY
    # ========================================================

    batch_id = (
        "|".join(
            active_filenames
        )
        +
        str(
            [
                r["Prediction"]
                for r in results
            ]
        )
    )


    if (
        st.session_state.last_batch_id
        != batch_id
    ):

        for filename, result in zip(
            active_filenames,
            results
        ):

            record = {

                "Timestamp":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "Input":
                    filename,

                "Prediction":
                    result["Prediction"],

                "Confidence (%)":
                    result["Confidence (%)"],

                "Fresh Probability (%)":
                    result["Fresh (%)"],

                "Rotten Probability (%)":
                    result["Rotten (%)"],

                "Reliability":
                    result["Reliability"]
            }


            st.session_state.prediction_history.append(
                record
            )


        save_prediction_history()


        st.session_state.last_batch_id = (
            batch_id
        )


    # ========================================================
    # SELECTED IMAGE REPORT
    # ========================================================

    st.divider()


    st.subheader(
        "📄 Detailed Prediction Report"
    )


    report = create_prediction_report(

        active_filenames[
            selected_index
        ],

        selected_result[
            "Prediction"
        ],

        selected_result[
            "Confidence (%)"
        ],

        selected_result[
            "Fresh (%)"
        ],

        selected_result[
            "Rotten (%)"
        ],

        quality,

        selected_result[
            "Reliability"
        ]
    )


    st.download_button(

        label="📥 Download Selected Image Report",

        data=report,

        file_name=(
            "food_freshness_report.txt"
        ),

        mime="text/plain"
    )


# ============================================================
# ANALYTICS
# ============================================================

st.divider()


st.header(
    "📊 Prediction Analytics"
)


history = (
    st.session_state.prediction_history
)


if len(history) == 0:

    st.info(
        """
        No predictions have been recorded yet.

        Upload images or capture camera photos
        to begin.
        """
    )


else:

    history_df = pd.DataFrame(
        history
    )


    total_predictions = len(
        history_df
    )


    fresh_count = int(
        (
            history_df["Prediction"]
            == "Fresh"
        ).sum()
    )


    rotten_count = int(
        (
            history_df["Prediction"]
            == "Rotten"
        ).sum()
    )


    average_confidence = float(
        history_df[
            "Confidence (%)"
        ].mean()
    )


    a1, a2, a3, a4 = st.columns(4)


    with a1:

        st.metric(
            "Images Analyzed",
            total_predictions
        )


    with a2:

        st.metric(
            "🟢 Fresh",
            fresh_count
        )


    with a3:

        st.metric(
            "🔴 Rotten",
            rotten_count
        )


    with a4:

        st.metric(
            "Average Confidence",
            f"{average_confidence:.2f}%"
        )


    # ========================================================
    # DISTRIBUTION
    # ========================================================

    st.subheader(
        "📊 Fresh vs Rotten Distribution"
    )


    distribution_df = pd.DataFrame(

        {
            "Class": [
                "Fresh",
                "Rotten"
            ],

            "Count": [
                fresh_count,
                rotten_count
            ]
        }
    )


    st.bar_chart(
        distribution_df.set_index(
            "Class"
        )
    )


    # ========================================================
    # CONFIDENCE TREND
    # ========================================================

    if total_predictions > 1:

        st.subheader(
            "📈 Confidence Trend"
        )


        confidence_chart = (
            history_df[
                [
                    "Timestamp",
                    "Confidence (%)"
                ]
            ]
            .copy()
        )


        confidence_chart = (
            confidence_chart
            .set_index(
                "Timestamp"
            )
        )


        st.line_chart(
            confidence_chart
        )


    # ========================================================
    # HISTORY
    # ========================================================

    st.subheader(
        "📜 Prediction History"
    )


    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DOWNLOAD HISTORY
    # ========================================================

    csv_data = (
        history_df
        .to_csv(
            index=False
        )
    )


    st.download_button(

        label="📥 Download Complete Prediction History",

        data=csv_data,

        file_name=(
            "food_freshness_prediction_history.csv"
        ),

        mime="text/csv"
    )


    # ========================================================
    # CLEAR HISTORY
    # ========================================================

    if st.button(
        "🗑️ Clear Prediction History"
    ):

        st.session_state.prediction_history = []

        st.session_state.last_batch_id = None


        if os.path.exists(
            HISTORY_FILE
        ):

            try:

                os.remove(
                    HISTORY_FILE
                )

            except Exception:

                pass


        st.rerun()


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.divider()


st.header(
    "📈 Model Performance"
)


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "Test Accuracy",
        "97.22%"
    )


with m2:

    st.metric(
        "Best Validation Accuracy",
        "98.11%"
    )


with m3:

    st.metric(
        "Test Images",
        "1,799"
    )


with m4:

    st.metric(
        "Classes",
        "2"
    )


# ============================================================
# CONFUSION MATRIX RESULTS
# ============================================================

st.subheader(
    "🎯 Classification Results"
)


cm1, cm2, cm3, cm4 = st.columns(4)


with cm1:

    st.metric(
        "Fresh Correct",
        "888"
    )


with cm2:

    st.metric(
        "Fresh → Rotten",
        "29"
    )


with cm3:

    st.metric(
        "Rotten Correct",
        "861"
    )


with cm4:

    st.metric(
        "Rotten → Fresh",
        "21"
    )


st.caption(
    """
    These values correspond to the confusion matrix
    obtained during model evaluation.
    """
)


# ============================================================
# TECHNICAL SPECIFICATIONS
# ============================================================

st.divider()


st.header(
    "⚙️ Technical Specifications"
)


tech1, tech2 = st.columns(2)


with tech1:

    with st.container(
        border=True
    ):

        st.subheader(
            "🧠 Deep Learning"
        )


        st.write(
            """
            **Architecture:** MobileNetV2

            **Learning Strategy:** Transfer Learning

            **Pretrained Weights:** ImageNet

            **Input Resolution:** 224 × 224 pixels

            **Output:** Binary Classification

            **Classes:** Fresh / Rotten
            """
        )


with tech2:

    with st.container(
        border=True
    ):

        st.subheader(
            "📊 Evaluation"
        )


        st.write(
            """
            **Training Images:** 8,389

            **Validation Images:** 1,798

            **Testing Images:** 1,799

            **Test Accuracy:** 97.22%

            **Best Validation Accuracy:** 98.11%

            **Loss Function:** Binary Cross-Entropy
            """
        )


# ============================================================
# AI PROCESSING PIPELINE
# ============================================================

st.divider()


st.header(
    "🔄 AI Processing Pipeline"
)


p1, p2, p3, p4, p5, p6 = st.columns(6)


with p1:

    with st.container(
        border=True
    ):

        st.subheader("📷")

        st.write(
            "Input"
        )

        st.caption(
            "Upload or capture multiple images"
        )


with p2:

    with st.container(
        border=True
    ):

        st.subheader("🔍")

        st.write(
            "Quality"
        )

        st.caption(
            "Image assessment"
        )


with p3:

    with st.container(
        border=True
    ):

        st.subheader("⚙️")

        st.write(
            "Preprocess"
        )

        st.caption(
            "224 × 224"
        )


with p4:

    with st.container(
        border=True
    ):

        st.subheader("🧠")

        st.write(
            "MobileNetV2"
        )

        st.caption(
            "Batch inference"
        )


with p5:

    with st.container(
        border=True
    ):

        st.subheader("🎯")

        st.write(
            "Prediction"
        )

        st.caption(
            "Fresh / Rotten"
        )


with p6:

    with st.container(
        border=True
    ):

        st.subheader("🔥")

        st.write(
            "Grad-CAM"
        )

        st.caption(
            "Explain selected image"
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()


st.header(
    "📚 Project Information"
)


tab1, tab2, tab3, tab4 = st.tabs(

    [
        "🎯 Objective",
        "🧠 Technology",
        "🔬 Methodology",
        "🚀 Future Scope"
    ]
)


# ============================================================
# OBJECTIVE
# ============================================================

with tab1:

    st.subheader(
        "Project Objective"
    )


    st.write(
        """
        The objective of this project is to develop an
        AI-powered computer vision system that classifies
        food images as Fresh or Rotten.

        The system uses MobileNetV2 transfer learning to
        extract visual features and perform binary
        classification.

        The dashboard supports both multi-image upload
        and multi-photo camera batch inference.
        """
    )


# ============================================================
# TECHNOLOGY
# ============================================================

with tab2:

    st.subheader(
        "Technologies Used"
    )


    st.write(
        """
        - Python
        - TensorFlow / Keras
        - MobileNetV2
        - Transfer Learning
        - NumPy
        - Pandas
        - Pillow
        - OpenCV
        - Matplotlib
        - Streamlit
        - Computer Vision
        - Deep Learning
        - Explainable AI
        - Grad-CAM
        - Batch Inference
        """
    )


# ============================================================
# METHODOLOGY
# ============================================================

with tab3:

    st.subheader(
        "System Methodology"
    )


    st.write(
        """
        1. The user selects either multiple image upload
           or multiple camera capture.

        2. For uploads, multiple images can be selected
           simultaneously.

        3. For camera mode, the user captures photographs
           one at a time and adds each photo to the batch.

        4. The system stores all selected or captured
           images in memory.

        5. Each image is converted to RGB format.

        6. Images are resized to 224 × 224 pixels.

        7. MobileNetV2 preprocessing is applied.

        8. All preprocessed images are combined into
           a batch tensor.

        9. MobileNetV2 performs batch inference.

        10. Each image receives a Fresh or Rotten
            classification.

        11. Confidence and reliability values are
            calculated for every image.

        12. Results are displayed in a table and visual
            image grid.

        13. The user can select one image for detailed
            analysis and Grad-CAM visualization.

        14. Batch predictions are stored in the
            prediction history.

        15. Results can be exported as a CSV file.
        """
    )


# ============================================================
# FUTURE SCOPE
# ============================================================

with tab4:

    st.subheader(
        "Future Scope"
    )


    st.write(
        """
        ### Multi-Class Food Identification

        A future model can identify specific food
        categories such as apples, bananas, tomatoes,
        carrots and other fruits and vegetables.

        ### Cloud Deployment

        The application can be deployed to a cloud
        platform for remote access.

        ### Mobile Application

        The computer vision model can be integrated
        into a mobile application.

        ### IoT Integration

        Cameras and sensors can be integrated into
        storage environments for continuous monitoring.

        ### Automated Alerts

        The system can generate alerts when food is
        classified as rotten.

        ### Advanced Explainable AI

        Additional interpretability techniques can
        be incorporated.

        ### Large-Scale Batch Processing

        The system can be extended to process large
        food inventories automatically.
        """
    )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()


st.info(
    """
    **Important:** This system performs visual AI-based
    classification. A high-confidence prediction does
    not constitute a laboratory food-safety assessment.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "AI Food Freshness Detection System | "
    "MobileNetV2 • Transfer Learning • "
    "Computer Vision • Batch Inference • "
    "Explainable AI"
)


st.caption(
    "B.Tech Artificial Intelligence & Machine Learning | "
    "Woxsen University"
)
