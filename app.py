import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
import io
import os
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

# -----------------------------
# CONFIG & SETUP
# -----------------------------
st.set_page_config(
    page_title="Fashion MNIST Predictor",
    page_icon="👕",
    layout="centered"
)

# Class names mapping
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load model
MODEL_PATH = "./models/fashion_mnist_cnn.h5"
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.error("❌ Model file not found!")
    st.stop()

st.title("👕 Fashion MNIST - Image Classifier")
st.markdown("Upload an image to make predictions or retrain the model with new data.")

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.header("🔮 Make a Prediction")
uploaded_image = st.file_uploader("Upload a 28x28 grayscale image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Read & preprocess image
    img = Image.open(uploaded_image).convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))

    # Prediction
    prediction = model.predict(img_array)
    predicted_label = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.image(img, caption="Uploaded Image", width=150)
    st.success(f"**Prediction:** {predicted_label} ({confidence:.2f}%)")

# -----------------------------
# RETRAINING SECTION
# -----------------------------
st.header("🔄 Retrain Model with New Data")
uploaded_data = st.file_uploader("Upload a CSV file containing pixel values & labels", type=["csv"])

if uploaded_data is not None:
    # Load and preprocess
    df = pd.read_csv(uploaded_data)
    if "label" not in df.columns:
        st.error("CSV must contain a 'label' column!")
    else:
        labels = df["label"].values
        features = df.drop(columns=["label"]).values.reshape(-1, 28, 28, 1) / 255.0

        # Retrain model with metrics tracking
        st.info("Training model on uploaded data...")
        history = model.fit(
            features, labels,
            epochs=3,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        # Evaluate metrics
        loss, accuracy = model.evaluate(features, labels, verbose=0)
        y_pred = np.argmax(model.predict(features), axis=1)
        precision = precision_score(labels, y_pred, average="weighted", zero_division=0)
        recall = recall_score(labels, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(labels, y_pred, average="weighted", zero_division=0)

        # Save updated model
        model.save("./models/fashion_mnist_cnn_updated.h5")

        # Display metrics
        st.success("✅ Model retrained and saved as `fashion_mnist_cnn_updated.h5`")
        st.subheader("📊 Retraining Evaluation Metrics")
        st.write(f"**Loss:** {loss:.4f}")
        st.write(f"**Accuracy:** {accuracy*100:.2f}%")
        st.write(f"**Precision:** {precision*100:.2f}%")
        st.write(f"**Recall:** {recall*100:.2f}%")
        st.write(f"**F1 Score:** {f1*100:.2f}%")

        # Plot training history
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_title('Training & Validation Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        st.pyplot(fig)

# -----------------------------
# DATA INSIGHTS SECTION
# -----------------------------
st.header("📊 Data Insights")
(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test / 255.0
x_test = np.expand_dims(x_test, -1)

preds = model.predict(x_test)
pred_labels = np.argmax(preds, axis=1)

# Class distribution chart
fig, ax = plt.subplots()
pd.Series(pred_labels).value_counts().sort_index().plot(
    kind="bar", ax=ax, color="skyblue", edgecolor="black"
)
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_title("Predicted Class Distribution")
st.pyplot(fig)

st.markdown("---")
