# Fashion MNIST Project

---
title: Fashion MNIST App
emoji: 👕
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.25.0
app_file: app.py
pinned: false
---

# 👕 Fashion MNIST - Image Classifier

This project is an **interactive image classification web app** built using **Streamlit** and **TensorFlow** to classify images from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  
The app allows users to:

- Upload a **28x28 grayscale image** and get a **predicted clothing category**.
- Retrain the model using **custom CSV data**.
- View performance metrics and predicted class distributions.

---

## 🚀 Live Demo

You can try the app here:  
🔗 **[Fashion MNIST App on Hugging Face Spaces](https://huggingface.co/spaces/JollyUmulisa/fashion-mnist-app)**

---

##  Repository Structure

fashion_mnist_project/
│
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation (this file)
├── fashion_mnist_cnn.h5 # Pre-trained CNN model (in root or models/)
├── models/ # Folder for storing models
│ └── fashion_mnist_cnn.h5
├── data/ # Optional dataset storage (not uploaded to repo)
│ └── raw/ # Contains original CSV or IDX files (ignored in push)
└── .gitattributes # Git LFS tracking configuration

yaml
---

##  Installation & Running Locally

To run the app locally on your machine:

### **1. Clone the repository**
```bash
git clone https://huggingface.co/spaces/JollyUmulisa/fashion-mnist-app
cd fashion-mnist-app

2. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

3. Install dependencies
pip install -r requirements.txt
4. Ensure the model file is in place
The app requires fashion_mnist_cnn.h5.
Place it in either:

fashion_mnist_project/fashion_mnist_cnn.h5
or:

fashion_mnist_project/models/fashion_mnist_cnn.h5
If you don’t have the model file, you can train it locally and save it:

python
model.save("fashion_mnist_cnn.h5")

5. Run the Streamlit app
streamlit run app.py

# Usage Instructions (in the app)
Upload an Image – Choose a 28x28 grayscale PNG/JPG image.

Get Predictions – The model will display:

Predicted clothing category.

Confidence score.

Optional Retraining – Upload a CSV with:

label column (0–9 class IDs).

784 pixel columns (flattened 28×28 grayscale values).

View Insights – Click "Generate Insights" to see predicted class distribution.

# Model Information
Architecture: Convolutional Neural Network (CNN)

Framework: TensorFlow / Keras

Training Dataset: Fashion MNIST (60,000 training, 10,000 testing)

Classes:
0. T-shirt/top

Trouser

Pullover

Dress

Coat

Sandal

Shirt

Sneaker

Bag

Ankle boot

🔗 Useful Links

 Hugging Face Space: https://huggingface.co/spaces/JollyUmulisa/fashion-mnist-app

 Fashion MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist

 Streamlit Docs: https://docs.streamlit.io/

 Hugging Face Spaces Docs: https://huggingface.co/docs/hub/spaces Used while Deploying
