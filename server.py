from fastapi import FastAPI
from pydantic import BaseModel
import base64, io
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model("./models/fashion_mnist_cnn.h5")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

class ImageData(BaseModel):
    image: str

@app.post("/predict")
def predict(data: ImageData):
    img_data = base64.b64decode(data.image)
    img = Image.open(io.BytesIO(img_data)).convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    prediction = model.predict(img_array)
    return {
        "class": CLASS_NAMES[np.argmax(prediction)],
        "confidence": float(np.max(prediction))
    }

@app.get("/")
def home():
    return {"message": "Fashion MNIST API is running"}
