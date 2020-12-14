# %%
import cv2
import tensorflow as tf
import numpy as np
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt
from scripts.preprocess import preprocess
from time import time
import logging

tf.get_logger().setLevel(logging.ERROR)

# %%
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
               'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Q', 'R',
               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

img_height = 64
img_width = 64

camera = cv2.VideoCapture(0)
GPIO.setmode(GPIO.BOARD)
inputPin = 15
outputPin = 23
GPIO.setup(inputPin, GPIO.IN)
GPIO.setup(outputPin, GPIO.OUT)

while True:
    x = GPIO.input(inputPin)
    ret, frame = camera.read()  # stream from camera

    if x == 1:  # Default value on input pin 15
        GPIO.output(outputPin, 0)  # Led signal off

    if ret and x == 0:  # If button pressed
        start_time = time()
        GPIO.output(outputPin, 1)  # Led signal on
        image = preprocess(frame)
        canny = cv2.Canny(image, 0, 255, 1)

        cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

        min_area = 400
        gtin = []
        sn = []

        plt.imshow(np.array(image))
        plt.show()
        idx = 1
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                ROI = image[y:y + h, x:x + w]
                img_array = cv2.resize(ROI, dsize=(img_height, img_width), interpolation=cv2.INTER_CUBIC)
                img_array = tf.expand_dims(img_array, 0)
                input_data = np.array(img_array, dtype=np.float32)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                predictions = interpreter.get_tensor(output_details[0]['index'])
                score = tf.nn.softmax(predictions[0])

                pred_label = class_names[np.argmax(score)]

                if idx % 2 == 0:
                    sn.append(pred_label)
                else:
                    gtin.append(pred_label)
                idx += 1

                # plt.imshow(np.array(image))
                # plt.title(pred_label)
                # plt.show()

        print(f"Process finished for {round(time() - start_time, 2)} sec.")
        print(f"Found GTIN: {''.join([str(elem) for elem in gtin])}")
        print(f"Found SN: {''.join([str(elem) for elem in sn])}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
