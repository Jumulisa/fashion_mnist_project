from locust import HttpUser, task, between
from tensorflow.keras.datasets import fashion_mnist
import base64
import io
from PIL import Image

class FashionMnistUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_image(self):
        # Load one image from Fashion MNIST
        (_, _), (x_test, _) = fashion_mnist.load_data()
        img = Image.fromarray(x_test[0])

        # Convert image to base64
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Send request to API
        self.client.post("/predict", json={"image": encoded_img})

    @task
    def home_page(self):
        self.client.get("/")
