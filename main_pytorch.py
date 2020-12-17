import cv2
import numpy as np
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt
from scripts.preprocess import preprocess
from torchvision import transforms
import torch
from scripts.convnet_pytorch import ConvNet
import time


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = {
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}
model = ConvNet()
PATH = "model/pytorch_model/pytorch_model.pt"
model.load_state_dict(torch.load(PATH))
model.eval()

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
               'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P','Q', 'R',
               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SLASH']

img_height = 64
img_width = 64

camera = cv2.VideoCapture(0)
GPIO.setmode(GPIO.BOARD)
inputPin = 15
outputPin = 23
GPIO.setup(inputPin, GPIO.IN)
GPIO.setup(outputPin, GPIO.OUT)

def main():
    while True:
        x = GPIO.input(inputPin)
        ret, frame = camera.read()  # stream from camera

        if x == 1:  # Default value on input pin 15
            GPIO.output(outputPin, 0)  # Led signal off

        if ret and x == 0:  # If button pressed
            start_time = time.time()
            GPIO.output(outputPin, 1)  # Led signal on
            image = preprocess(frame)
            canny = cv2.Canny(image, 0, 255, 1)

            cnts, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[1])



            min_area = 5
            data = []

            plt.imshow(np.array(image))
            plt.show()

            for c in cnts:
                area = cv2.contourArea(c)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
                    ROI = image[y:y + h, x:x + w]
                    img_array = data_transforms['val'](ROI)
                    img_array = img_array.view(1, 3, 64, 64)

                    outputs = model(img_array)
                    _, preds = torch.max(outputs, 1)

                    data.append(class_names[preds])

                    # plt.imshow(np.array(image))
                    # plt.title(class_names[preds])
                    # plt.show()

            data = ''.join([str(elem) for elem in data])
            gtin = data[::-1][-14:]
            sn = data[::-1][-27: -14]
            batch = data[::-1][-33: -27]
            expired = data[::-1].replace(gtin, "").replace(sn, "").replace(batch, "")

            print(f"Process finished for {round(time.time() - start_time, 2)} sec.")
            print(f"Found GTIN: {gtin}")
            print(f"Found SN: {sn}")
            print(f"Found BATCH: {batch}")
            print(f"Found EXP: {expired}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()

if __name__ == '__main__':
    main()