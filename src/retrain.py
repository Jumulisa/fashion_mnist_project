import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from preprocessing import preprocess_data  # your preprocessing function

# Paths
NEW_DATA_DIR = "data/new_data/"
MODEL_PATH = "models/fashion_mnist_model.h5"

def save_uploaded_file(uploaded_file):
    """Save uploaded file to data/new_data/."""
    os.makedirs(NEW_DATA_DIR, exist_ok=True)
    file_path = os.path.join(NEW_DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def retrain_model_from_file(file_path):
    """Retrain model using new uploaded data."""
    # Load existing model
    model = load_model(MODEL_PATH)

    # Load and preprocess new data
    df = pd.read_csv(file_path)
    X, y = preprocess_data(df)

    # Convert labels to one-hot encoding
    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Retrain model with early stopping
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=10, batch_size=32, callbacks=[early_stop])

    # Save updated model
    model.save(MODEL_PATH)

    return model, lb
