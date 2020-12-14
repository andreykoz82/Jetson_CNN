# %%
import tensorflow as tf
import os

os.chdir('DeepLearning/YandexGPU/OCR Keras Jetson')
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('model')  # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
