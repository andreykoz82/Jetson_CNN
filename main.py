import time
import cv2
import RPi.GPIO as GPIO
from scripts.helper_functions import preprocess
from scripts.helper_functions import extract_text
from scripts.cnn import ConvNet
import torch
import matplotlib.pyplot as plt

PATH = '/home/andrey/Jetson_HSE/model/pytorch_model.pt'
model = ConvNet()
model.load_state_dict(torch.load(PATH))
model.eval()

camera = cv2.VideoCapture(0)
GPIO.setmode(GPIO.BOARD)
inputPin = 15
outputPin = 23
GPIO.setup(inputPin, GPIO.IN)
GPIO.setup(outputPin, GPIO.OUT)

if __name__ == '__main__':
    while True:
        x = GPIO.input(inputPin)

        if x == 1:  # Default value on input pin 15
            GPIO.output(outputPin, 0)  # Led signal off

        if x == 0:  # If button pressed
            start_time = time.time()
            ret, frame = camera.read()  # stream from camera
            time.sleep(0.2)
            GPIO.output(outputPin, 1)  # Led signal on
            image_gtin, image_sn, image_batch = preprocess(frame)
            gtin = extract_text(image_gtin, model)
            sn = extract_text(image_sn, model)
            batch = extract_text(image_batch, model)

            gtin_true = ''
            sn_true = ''
            batch_true = ''

            print(f"Process finished for {round(time.time() - start_time, 2)} sec.")
            print(f"Found GTIN: {gtin}, result: {gtin_true == gtin}")
            print(f"Found SN: {sn}, result: {sn_true == sn}")
            print(f"Found BATCH: {batch}, result: {batch_true == batch}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
