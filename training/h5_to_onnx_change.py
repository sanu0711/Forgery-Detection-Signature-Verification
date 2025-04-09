import keras2onnx
from tensorflow.keras.models import load_model

model = load_model("forg.h5")
onnx_model = keras2onnx.convert_keras(model, model.name)
keras2onnx.save_model(onnx_model, "forg.onnx")
