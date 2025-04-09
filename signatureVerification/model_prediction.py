import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path
import io
base_dir = Path(__file__).resolve().parent.parent

def preprocess_image(file, target_size=(150, 150)):
    file.seek(0)
    img = image.load_img(io.BytesIO(file.read()), target_size=target_size)
    # img = image.load_img(file, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array, img

def prediction_score(model_name, file,target_size):
    model_path = os.path.join(base_dir, 'trained_models', model_name)
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist.")
    
    model = tf.keras.models.load_model(model_path)
    img_array, _ = preprocess_image(file, target_size=target_size)
    prediction = model.predict(img_array)
    return prediction[0][0]


# def preprocess_image(img_path, target_size=(150, 150)):
#     img = image.load_img(img_path, target_size=target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  
#     return img_array, img


# def prediction_score(model_name, img_path):
#     model_dir = os.path.join(base_dir, 'trained_models', model_name)
#     if not os.path.exists(model_dir):
#         raise ValueError(f"Model directory {model_dir} does not exist.")
#     model = tf.keras.models.load_model(model_dir)
#     img_array, img = preprocess_image(img_path)
#     prediction = model.predict(img_array)
#     return prediction[0][0]


# img= os.path.join(base_dir, 'static', 'img', 'logo.jpg')
# score = prediction_score('forg.h5', img)
# print(score)
