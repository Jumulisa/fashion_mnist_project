import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# 1. Load model
model = tf.keras.models.load_model("./models/fashion_mnist_cnn.h5")

# 2. Load test data
(_, _), (x_test, y_test) = fashion_mnist.load_data()

# Normalize the data
x_test = x_test / 255.0
x_test = np.expand_dims(x_test, -1)  # Add channel dimension

# 3. Class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 4. Make predictions for first 5 samples
predictions = model.predict(x_test[:5])

# 5. Plot the first 5 images with predictions
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    predicted_label = np.argmax(predictions[i])
    plt.title(f"P: {class_names[predicted_label]}\nT: {class_names[y_test[i]]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
