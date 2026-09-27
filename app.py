import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
import zipfile
import json
import io
import base64
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.json"
COMPARISON_METRICS_PATH = PROJECT_ROOT / "results" / "comparison_metrics.json"
COMPARISON_CHART_PATH = PROJECT_ROOT / "results" / "cnn_vs_svm.png"

DIGIT_WORDS = (
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
)


def load_verified_accuracy():
    try:
        with METRICS_PATH.open(encoding="utf-8") as metrics_file:
            return float(json.load(metrics_file)["test_accuracy"]) * 100
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def as_rgba_array(image):
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGBA"))

    image_array = np.asarray(image)
    if image_array.ndim == 2:
        image_array = np.repeat(image_array[:, :, None], 4, axis=2)
        image_array[:, :, 3] = 255
    elif image_array.ndim == 3 and image_array.shape[2] == 3:
        alpha = np.full((*image_array.shape[:2], 1), 255, dtype=image_array.dtype)
        image_array = np.concatenate((image_array, alpha), axis=2)

    if image_array.ndim != 3 or image_array.shape[2] < 4:
        raise ValueError("Image must contain grayscale, RGB, or RGBA data.")

    return image_array[:, :, :4]


def preprocess_image(image_data):
    image = as_rgba_array(image_data).astype("float32")
    rgb = image[:, :, :3]
    grayscale = np.mean(rgb, axis=2)

    # Detect the canvas/upload background, then convert the foreground to MNIST's
    # white-digit-on-black convention without assuming the input background color.
    border = np.concatenate(
        (grayscale[0, :], grayscale[-1, :], grayscale[:, 0], grayscale[:, -1])
    )
    background_value = float(np.median(border))
    foreground_mask = np.abs(grayscale - background_value) > 15

    if not np.any(foreground_mask):
        return None

    rows = np.where(foreground_mask.any(axis=1))[0]
    cols = np.where(foreground_mask.any(axis=0))[0]
    cropped = grayscale[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    if background_value > 127:
        cropped = 255.0 - cropped

    cropped = np.clip(cropped, 0, 255)
    pil_image = Image.fromarray(cropped.astype(np.uint8), mode="L")
    pil_image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    final_image = Image.new("L", (28, 28), 0)
    final_image.paste(
        pil_image,
        ((28 - pil_image.width) // 2, (28 - pil_image.height) // 2),
    )

    return np.asarray(final_image, dtype="float32")[None, :, :, None] / 255.0


def preprocess_camera_image(image_data):
    """Prepare camera input as white-on-black MNIST data and return debug stages."""
    rgba_image = as_rgba_array(image_data).astype("float32")
    rgb_image = rgba_image[:, :, :3]
    grayscale = (
        0.299 * rgb_image[:, :, 0]
        + 0.587 * rgb_image[:, :, 1]
        + 0.114 * rgb_image[:, :, 2]
    )

    border = np.concatenate(
        (grayscale[0, :], grayscale[-1, :], grayscale[:, 0], grayscale[:, -1])
    )
    background_value = float(np.median(border))
    difference = grayscale - background_value
    foreground = np.abs(difference) > 18

    if not np.any(foreground):
        return None

    rows = np.where(foreground.any(axis=1))[0]
    cols = np.where(foreground.any(axis=0))[0]
    cropped_grayscale = grayscale[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # Match the training pipeline exactly: convert the digit to MNIST-style
    # white foreground on black background while preserving the paper/background
    # polarity detected from the actual camera frame.
    if background_value > 127:
        cropped_grayscale = 255.0 - cropped_grayscale
    cropped_grayscale = np.clip(cropped_grayscale, 0, 255)

    cropped_image = Image.fromarray(cropped_grayscale.astype(np.uint8), mode="L")
    cropped_image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    final_image = Image.new("L", (28, 28), 0)
    final_image.paste(
        cropped_image,
        ((28 - cropped_image.width) // 2, (28 - cropped_image.height) // 2),
    )
    final_array = np.asarray(final_image, dtype="float32")
    processed = final_array[None, :, :, None] / 255.0

    return {
        "processed": processed,
        "original": Image.fromarray(rgb_image.astype(np.uint8), mode="RGB"),
        "grayscale": Image.fromarray(np.clip(grayscale, 0, 255).astype(np.uint8), mode="L"),
        "thresholded": Image.fromarray((foreground.astype(np.uint8) * 255), mode="L"),
        "cropped": cropped_image,
        "final": Image.fromarray(final_array.astype(np.uint8), mode="L"),
    }


def show_camera_debug(camera_stages):
    st.subheader("🔎 Camera Preprocessing Debug")
    debug_columns = st.columns(5)
    for column, label, image_key in zip(
        debug_columns,
        ("Original", "Grayscale", "Thresholded", "Cropped Digit", "Final CNN Input"),
        ("original", "grayscale", "thresholded", "cropped", "final"),
    ):
        column.image(camera_stages[image_key], caption=label, use_container_width=True)
    processed = camera_stages["processed"]
    st.caption(
        f"Final CNN input shape: {processed.shape} | "
        f"Pixel min: {processed.min():.4f} | "
        f"Pixel max: {processed.max():.4f} | "
        f"Pixel mean: {processed.mean():.4f}"
    )


def predict_digit(input_image):
    if model is None:
        raise RuntimeError("The CNN model is unavailable.")

    raw_prediction = np.asarray(model.predict(input_image, verbose=0))
    if raw_prediction.shape != (1, 10):
        raise ValueError(f"Unexpected CNN prediction shape: {raw_prediction.shape}")

    output = raw_prediction[0].astype("float32")
    if np.all(output >= 0) and np.all(output <= 1) and np.isclose(
        output.sum(), 1.0, atol=1e-3
    ):
        probabilities = output
    else:
        shifted_output = output - np.max(output)
        probabilities = np.exp(shifted_output)
        probabilities /= probabilities.sum()

    predicted_digit = int(np.argmax(probabilities))
    return predicted_digit, probabilities


def confidence_message(confidence):
    if confidence >= 90:
        return "✅ The model is highly confident in this prediction.", "high"
    if confidence >= 60:
        return "ℹ️ The model has moderate confidence in this prediction.", "moderate"
    return "⚠️ The model is not highly confident. Try drawing the digit more clearly.", "low"


def render_confidence_donut(confidence, level):
    labels = ["High", "Medium", "Low"]
    colors = ["#22c55e", "#eab308", "#ef4444"]
    category_index = {"high": 0, "moderate": 1, "low": 2}[level]
    values = [0.0, 0.0, 0.0]
    values[category_index] = confidence

    st.subheader("🎯 Prediction Confidence")
    figure, axis = plt.subplots(figsize=(3.2, 3.2))
    axis.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2},
    )
    axis.text(
        0,
        0,
        f"{confidence:.2f}%",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )
    axis.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    axis.set(aspect="equal")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)

    status = {"high": "High Confidence", "moderate": "Medium Confidence", "low": "Low Confidence"}[level]
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Status:** {status}")
    if level == "low":
        st.warning("⚠️ Try drawing the digit more clearly.")


def add_history(source, digit, confidence):
    st.session_state.prediction_history.append(
        {
            "Input": source,
            "Digit": digit,
            "Confidence": round(confidence, 2),
            "Timestamp": __import__("datetime").datetime.now().strftime("%H:%M:%S"),
        }
    )


def render_voice_result(digit, confidence, key):
    spoken_text = (
        f"The predicted digit is {DIGIT_WORDS[digit]}, "
        f"with {confidence:.2f} percent confidence."
    )
    encoded_text = base64.b64encode(spoken_text.encode("utf-8")).decode("ascii")
    components.html(
        f"""
        <button onclick=\"speakResult()\" style=\"padding:0.5rem 1rem;cursor:pointer;\">
            🔊 Speak Result
        </button>
        <script>
        function speakResult() {{
            const text = atob('{encoded_text}');
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
        }}
        </script>
        """,
        height=55,
        scrolling=False,
    )


def render_prediction(source, original_image, processed_image, digit, probabilities, key):
    confidence = float(np.max(probabilities)) * 100
    add_history(source, digit, confidence)
    message, level = confidence_message(confidence)
    st.success(f"✅ {source} recognized successfully!")
    st.markdown(
        f"""
        <div style=\"text-align:center;padding:25px;border-radius:15px;
        background-color:#1e293b;margin-top:20px;\">
            <h2>🧠 CNN Prediction</h2>
            <h1 style=\"font-size:70px;\">{digit}</h1>
            <h3>Confidence: {confidence:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if level == "high":
        st.success(message)
    elif level == "moderate":
        st.info(message)
    else:
        st.warning(message)

    render_confidence_donut(confidence, level)

    st.subheader("📊 Prediction Probabilities")
    for class_digit, probability in enumerate(probabilities):
        st.write(f"**{class_digit}** — {probability * 100:.2f}%")
        st.progress(float(probability))

    st.subheader("🔊 Voice Result")
    render_voice_result(digit, confidence, key)
    render_grad_cam(original_image, processed_image, digit, key)


def render_grad_cam(original_image, processed_image, digit, key):
    try:
        convolutional_layers = [
            layer for layer in model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        if not convolutional_layers:
            st.info("Grad-CAM is unavailable because the model has no convolutional layer.")
            return

        last_conv_layer = convolutional_layers[-1]
        with tf.GradientTape() as tape:
            activation = tf.convert_to_tensor(processed_image)
            conv_outputs = None
            for layer in model.layers:
                activation = layer(activation, training=False)
                if layer is last_conv_layer:
                    conv_outputs = activation
            predictions = activation
            if conv_outputs is None:
                raise ValueError("The selected convolutional layer was not reached.")
            class_score = predictions[:, digit]
        gradients = tape.gradient(class_score, conv_outputs)
        weights = tf.reduce_mean(gradients, axis=(1, 2))
        heatmap = tf.reduce_sum(conv_outputs * weights[:, None, None, :], axis=-1)
        heatmap = tf.maximum(heatmap, 0)[0]
        heatmap /= tf.maximum(tf.reduce_max(heatmap), 1e-8)
        heatmap = tf.image.resize(heatmap[..., None], (28, 28)).numpy()[:, :, 0]

        st.subheader("🧠 Why did the CNN predict this?")
        st.caption(
            "Highlighted regions represent areas that contributed strongly to the model's prediction. "
            "This is an approximate visualization, not a proof of exact reasoning."
        )
        figure, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(original_image, cmap="gray")
        axes[0].set_title("Input")
        axes[1].imshow(processed_image[0, :, :, 0], cmap="gray")
        axes[1].set_title("MNIST input")
        axes[2].imshow(processed_image[0, :, :, 0], cmap="gray")
        axes[2].imshow(heatmap, cmap="jet", alpha=0.45)
        axes[2].set_title(f"Grad-CAM: {last_conv_layer.name}")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        st.pyplot(figure, clear_figure=True)
    except (RuntimeError, ValueError, tf.errors.InvalidArgumentError) as error:
        st.warning(f"Explainability visualization unavailable: {error}")


st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)


# -------------------------------
# LOAD CNN MODEL
# -------------------------------

@st.cache_resource
def load_cnn_model():
    model_path = Path(__file__).resolve().parent / "model" / "digit_cnn.keras"

    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"The CNN model is missing or empty: {model_path}"
        )

    if not zipfile.is_zipfile(model_path):
        raise ValueError(
            f"The CNN model is not a valid .keras archive: {model_path}"
        )

    loaded_model = tf.keras.models.load_model(model_path, compile=False)

    if loaded_model.input_shape != (None, 28, 28, 1):
        raise ValueError(
            f"Unexpected CNN input shape: {loaded_model.input_shape}"
        )

    if loaded_model.output_shape != (None, 10):
        raise ValueError(
            f"Unexpected CNN output shape: {loaded_model.output_shape}"
        )

    return loaded_model


try:
    model = load_cnn_model()
    model_load_error = None
except (FileNotFoundError, OSError, ValueError, ImportError) as error:
    model = None
    model_load_error = error


# -------------------------------
# TITLE
# -------------------------------

st.title("✍️ Handwritten Digit Recognition")

st.write(
    "Draw a handwritten digit and let our Deep Learning CNN model recognize it."
)

st.info(
    "CNN model trained on the MNIST handwritten digit dataset"
)


# -------------------------------
# MODEL STATUS
# -------------------------------

st.subheader("🚀 Project Status")

if model is None:
    st.error(f"CNN model unavailable: {model_load_error}")
else:
    st.success("CNN Model Loaded Successfully")

verified_accuracy = load_verified_accuracy()
if verified_accuracy is None:
    st.write("**Test Accuracy: Not available in project metrics**")
else:
    st.write(f"**Test Accuracy: {verified_accuracy:.2f}% (verified during training)**")


# -------------------------------
# DRAWING AREA
# -------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

st.subheader("✏️ Draw Your Digit")

st.write("Draw a digit from **0 to 9** inside the box below.")


canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=15,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)


# -------------------------------
# RECOGNIZE BUTTON
# -------------------------------

if st.button("🔍 Recognize Digit", width="stretch"):
    if canvas_result.image_data is None:
        st.warning("⚠️ Please draw a digit first.")
    else:
        try:
            processed_image = preprocess_image(canvas_result.image_data)
            if processed_image is None:
                st.warning("⚠️ Please draw a digit first.")
            elif model is None:
                st.error("The CNN model could not be loaded. Check the model file and restart the app.")
            else:
                predicted_digit, probabilities = predict_digit(processed_image)
                render_prediction(
                    "Drawing",
                    canvas_result.image_data[:, :, :3],
                    processed_image,
                    predicted_digit,
                    probabilities,
                    "canvas",
                )
        except (ValueError, RuntimeError, OSError) as error:
            st.error(f"Prediction failed: {error}")

st.divider()

# -------------------------------
# UPLOAD RECOGNITION
# -------------------------------

st.subheader("📷 Upload Handwritten Digit")
uploaded_file = st.file_uploader(
    "Upload a PNG, JPG, or JPEG image",
    type=["png", "jpg", "jpeg"],
    key="digit_upload",
)

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGBA")
        st.image(uploaded_image, caption="Uploaded image", width=280)
        if st.button("🔍 Predict Uploaded Image", key="predict_upload"):
            processed_image = preprocess_image(uploaded_image)
            if processed_image is None:
                st.warning("Please upload an image containing a visible digit.")
            else:
                predicted_digit, probabilities = predict_digit(processed_image)
                render_prediction(
                    "Uploaded image",
                    np.asarray(uploaded_image)[:, :, :3],
                    processed_image,
                    predicted_digit,
                    probabilities,
                    "upload",
                )
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Could not process uploaded image: {error}")

st.divider()

# -------------------------------
# CAMERA RECOGNITION
# -------------------------------

st.subheader("📸 Camera Recognition")
camera_file = st.camera_input("Capture a handwritten digit", key="digit_camera")

if camera_file is not None:
    try:
        camera_image = Image.open(io.BytesIO(camera_file.getvalue())).convert("RGBA")
        st.image(camera_image, caption="Camera image", width=280)
        if st.button("🔍 Predict Camera Image", key="predict_camera"):
            camera_stages = preprocess_camera_image(camera_image)
            if camera_stages is None:
                st.warning("Please capture an image containing a visible digit.")
            else:
                show_camera_debug(camera_stages)
                processed_image = camera_stages["processed"]
                predicted_digit, probabilities = predict_digit(processed_image)
                render_prediction(
                    "Camera image",
                    np.asarray(camera_image)[:, :, :3],
                    processed_image,
                    predicted_digit,
                    probabilities,
                    "camera",
                )
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"Could not process camera image: {error}")

st.divider()

# -------------------------------
# PREDICTION HISTORY
# -------------------------------

st.subheader("📈 Prediction History")
if st.session_state.prediction_history:
    history_frame = __import__("pandas").DataFrame(st.session_state.prediction_history)
    st.dataframe(history_frame, width="stretch", hide_index=True)
    st.line_chart(history_frame.set_index("Timestamp")["Confidence"])
else:
    st.caption("Successful canvas, upload, and camera predictions will appear here.")

if st.button("🗑️ Clear History", key="clear_history"):
    st.session_state.prediction_history = []
    st.rerun()

st.divider()

# -------------------------------
# CNN VS SVM COMPARISON
# -------------------------------

st.subheader("📊 CNN vs SVM Comparison")
if COMPARISON_METRICS_PATH.is_file():
    try:
        with COMPARISON_METRICS_PATH.open(encoding="utf-8") as metrics_file:
            comparison = json.load(metrics_file)
        comparison_rows = [
            {"Metric": metric, "CNN": comparison["cnn"][metric], "SVM": comparison["svm"][metric]}
            for metric in comparison["cnn"]
            if metric in comparison["svm"]
        ]
        st.dataframe(__import__("pandas").DataFrame(comparison_rows), hide_index=True, width="stretch")
        if COMPARISON_CHART_PATH.is_file():
            st.image(str(COMPARISON_CHART_PATH), caption="CNN vs SVM comparison")
        st.markdown(
            "**Why CNN?** CNNs learn spatial patterns and local visual features from images, "
            "which makes them well suited to handwritten digit recognition."
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        st.warning(f"Comparison results are unavailable: {error}")
else:
    st.info("Comparison metrics are not available yet. Run training/compare_svm.py once to generate them.")