
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------
# 1. Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="CNN Fruit Classification",
    page_icon="🍎",
    layout="centered"
)

st.markdown(
    """
    <style>
    :root {
        --lavender: #c4a8f0;
        --lavender-deep: #a985df;
        --lavender-light: #332748;
        --ink: #f8f4ff;
        --muted: #d0c4e2;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 12% 8%, rgba(139, 103, 194, 0.32), transparent 34%),
            radial-gradient(ellipse at 88% 58%, rgba(102, 72, 157, 0.3), transparent 38%),
            linear-gradient(145deg, #20182d, #2d2140 56%, #1b1527);
        background-size: 160% 160%;
        animation: lavender-breath 16s ease-in-out infinite alternate;
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMain"] {
        color: var(--ink);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 780px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        animation: page-arrive 650ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
    }

    .hero {
        position: relative;
        overflow: hidden;
        margin: 0 0 1.25rem;
        padding: 2rem 1.5rem 1.8rem;
        border: 1px solid rgba(205, 178, 245, 0.22);
        border-radius: 26px;
        background: linear-gradient(140deg, rgba(62, 47, 84, 0.97), rgba(43, 32, 61, 0.97));
        box-shadow: 0 18px 48px rgba(8, 4, 16, 0.38), 0 0 30px rgba(164, 126, 219, 0.08);
        text-align: center;
        animation: hero-pop 750ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
    }

    .hero::before,
    .hero::after {
        position: absolute;
        z-index: 0;
        width: 175px;
        height: 175px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(199, 166, 245, 0.3), transparent 70%);
        content: "";
        pointer-events: none;
        animation: glow-drift 8s ease-in-out infinite alternate;
    }

    .hero::before {
        top: -92px;
        left: -45px;
    }

    .hero::after {
        right: -48px;
        bottom: -110px;
        animation-delay: -3s;
    }

    .hero-content {
        position: relative;
        z-index: 1;
    }

    .hero-icon {
        display: inline-grid;
        width: 68px;
        height: 68px;
        margin-bottom: 0.65rem;
        place-items: center;
        border: 1px solid rgba(223, 204, 249, 0.22);
        border-radius: 22px;
        background: linear-gradient(145deg, #715598, #493660);
        box-shadow: 0 10px 26px rgba(8, 4, 16, 0.34), 0 0 24px rgba(191, 159, 235, 0.2);
        font-size: 2rem;
        animation: fruit-float 3.2s ease-in-out infinite;
    }

    .hero h1 {
        margin: 0;
        color: #fffaff;
        font-family: "Trebuchet MS", sans-serif;
        font-size: clamp(1.8rem, 5vw, 2.55rem);
        font-weight: 800;
        letter-spacing: -0.045em;
    }

    .hero p {
        max-width: 480px;
        margin: 0.55rem auto 0;
        color: #d6c9e8;
        font-size: 1rem;
        line-height: 1.65;
    }

    [data-testid="stFileUploader"] {
        padding: 0.9rem;
        border: 1px solid rgba(205, 178, 245, 0.2);
        border-radius: 20px;
        background: rgba(49, 37, 68, 0.96);
        box-shadow: 0 10px 30px rgba(8, 4, 16, 0.28);
        color: var(--ink);
        transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
    }

    [data-testid="stFileUploader"]:hover {
        transform: translateY(-3px);
        border-color: rgba(205, 178, 245, 0.58);
        box-shadow: 0 15px 36px rgba(8, 4, 16, 0.4), 0 0 20px rgba(173, 136, 224, 0.12);
    }

    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #8870a8;
        border-radius: 14px;
        background: rgba(37, 28, 52, 0.88);
    }

    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small {
        color: var(--muted) !important;
    }

    [data-testid="stFileUploader"] button,
    [data-testid="stButton"] button {
        overflow: hidden;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(120deg, #b69ae5, #8060b6, #b69ae5);
        background-size: 200% 100%;
        color: white;
        font-weight: 700;
        box-shadow: 0 7px 18px rgba(8, 4, 16, 0.32);
        transition: transform 180ms ease, box-shadow 180ms ease, background-position 350ms ease;
    }

    [data-testid="stButton"] button {
        min-height: 3rem;
        width: 100%;
        font-size: 1rem;
    }

    [data-testid="stFileUploader"] button:hover,
    [data-testid="stButton"] button:hover {
        transform: translateY(-2px);
        background-position: 100% 0;
        box-shadow: 0 11px 24px rgba(8, 4, 16, 0.4), 0 0 18px rgba(184, 151, 230, 0.2);
    }

    [data-testid="stImage"] {
        overflow: hidden;
        border: 5px solid #49375f;
        border-radius: 20px;
        box-shadow: 0 12px 32px rgba(8, 4, 16, 0.4);
        animation: image-show 550ms ease-out both;
    }

    [data-testid="stMetric"] {
        padding: 1rem 1.2rem;
        border: 1px solid rgba(205, 178, 245, 0.22);
        border-radius: 16px;
        background: rgba(49, 37, 68, 0.96);
        box-shadow: 0 8px 23px rgba(8, 4, 16, 0.28);
    }

    [data-testid="stMetricValue"] {
        color: var(--lavender);
    }

    [data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, #d0b7f3, #9a76cf);
    }

    [data-testid="stAlert"] {
        border-radius: 16px;
        border: 1px solid rgba(205, 178, 245, 0.2);
        background: rgba(49, 37, 68, 0.96);
        color: var(--ink);
    }

    h2, h3, [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"],
    [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }

    @keyframes page-arrive {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes hero-pop {
        from { opacity: 0; transform: translateY(16px) scale(0.985); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    @keyframes lavender-breath {
        from { background-position: 0% 0%; }
        to { background-position: 100% 100%; }
    }

    @keyframes glow-drift {
        from { transform: translate(-8px, 8px) scale(0.9); }
        to { transform: translate(14px, -10px) scale(1.12); }
    }

    @keyframes fruit-float {
        0%, 100% { transform: translateY(0) rotate(-2deg); }
        50% { transform: translateY(-7px) rotate(2deg); }
    }

    @keyframes image-show {
        from { opacity: 0; transform: translateY(12px) scale(0.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# 2. Fruit Classes
# Must match the training class order
# -----------------------------------
CLASS_NAMES = [
    "apple",
    "avocado",
    "banana",
    "cherry",
    "kiwi",
    "mango",
    "orange",
    "pinenapple",
    "strawberries",
    "watermelon"
]

# -----------------------------------
# 3. Load the Trained CNN Model
# -----------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fruit_cnn_model.keras" )


# -----------------------------------
# 4. Application Title
# -----------------------------------
st.markdown(
    """
    <section class="hero">
        <div class="hero-content">
            <div class="hero-icon">🍎</div>
            <h1>Fruit Classification</h1>
            <p>Upload a fruit photo and let our CNN model identify what's fresh.</p>
        </div>
    </section>
    """,
    unsafe_allow_html=True
)

st.info(
    "This model recognizes 10 fruit classes. "
    "For reliable results, upload a clear image of one fruit."
)

# -----------------------------------
# 5. Load Model and Upload Image
# -----------------------------------
try:
    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload a fruit image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded fruit image",
            use_container_width=True
        )

        # -----------------------------------
        # 6. Predict Fruit
        # -----------------------------------
        if st.button("Predict Fruit", type="primary"):

            # Resize to the input size used during training
            resized_image = image.resize((128, 128))

            # Convert image to a NumPy array
            image_array = np.array(
                resized_image,
                dtype=np.float32
            )

            # Normalize pixel values from 0-255 to 0-1
            image_array = image_array / 255.0

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            # Get model predictions
            probabilities = model.predict(
                image_array,
                verbose=0
            )[0]

            # Verify the number of model outputs
            if len(probabilities) != len(CLASS_NAMES):
                st.error(
                    "The number of model outputs does not match "
                    "the number of fruit classes."
                )
            else:
                # Find the most probable class
                predicted_index = int(np.argmax(probabilities))
                predicted_fruit = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index]) * 100

                # Display prediction
                st.success(
                    f"Predicted Fruit: {predicted_fruit.title()}"
                )

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                # Show all class probabilities
                st.subheader("Class Probabilities")

                for index, name in enumerate(CLASS_NAMES):
                    probability = float(probabilities[index]) * 100

                    st.write(
                        f"**{name.title()}** — {probability:.2f}%"
                    )

                    st.progress(
                        min(max(probability / 100.0, 0.0), 1.0)
                    )

                st.caption(
                    "Confidence is the model's estimated probability, "
                    "not a guarantee that the prediction is correct."
                )

except FileNotFoundError:
    st.error(
        "Model file not found. Keep fruit_cnn_model.keras "
        "in the same folder as app.py."
    )

except Exception as error:
    st.error(f"An error occurred: {error}")
