import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
model = tf.keras.models.load_model('forg.h5')

# Import the image module from keras.preprocessing

real_image_dir = '/content/Train/Original'
fake_image_dir = '/content/Train/Forged'

# Function to preprocess the image
def preprocess_image(img_path, target_size=(150, 150)): # Change to the original training input size
    # Check if the provided path is a directory
    if os.path.isdir(img_path):
        # If it's a directory, list the files inside
        files = os.listdir(img_path)
        # Select the first image file (you might want to adjust this based on your directory structure)
        if files:
            img_path = os.path.join(img_path, files[0])
        else:
            raise ValueError("The provided directory is empty.")

    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image
    return img_array, img

# Example image path (replace with your actual image path)
img_path = os.path.join(real_image_dir, os.listdir(real_image_dir)[0])
fake_img_path = os.path.join(fake_image_dir, os.listdir(fake_image_dir)[0])
# Preprocess the image
img_array, img = preprocess_image(img_path)
# fimg_array, fimg = preprocess_image(fake_img_path)
# Print the shape of the preprocessed image to verify
# print(img_array.shape)
# print(fimg_array.shape)

# Make predictions
prediction = model.predict(img_array)

# Since it's a binary classification (real or fake), we use a threshold of 0.5
if prediction < 0.5:
    result = "Forged"
else:
    result = "Original"

# Display the image with the classification result
# plt.imshow(img)
# plt.title(f'{result} Signature.')
# plt.axis('off')
# plt.show()