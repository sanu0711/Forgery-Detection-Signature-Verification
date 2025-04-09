import onnxruntime as ort
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load ONNX model
session = ort.InferenceSession("forg.onnx")

# Get model input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # e.g., [None, 150, 150, 3]

# Preprocess function
def preprocess_image(img_path, target_size=(150, 150)):
    if os.path.isdir(img_path):
        files = os.listdir(img_path)
        if files:
            img_path = os.path.join(img_path, files[0])
        else:
            raise ValueError("Directory is empty.")

    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array, img

# Paths
img_path = os.path.join(real_image_dir, os.listdir(real_image_dir)[0])

# Preprocess
img_array, img = preprocess_image(img_path)

# Predict using ONNX
prediction = session.run(None, {input_name: img_array})[0]

# Binary classification
result = "Original" if prediction[0][0] >= 0.5 else "Forged"

# Show image with result
# plt.imshow(img)
# plt.title(f'{result} Signature')
# plt.axis('off')
# plt.show()
