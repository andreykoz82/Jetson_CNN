# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt


# %%
def preprocess(image):
    image = image[270:720, 400:1150]
    image, alpha, beta = automatic_brightness_and_contrast(image)
    norm_img = np.zeros((image.shape[0], image.shape[1]))
    image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
    image = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)[1]
    image = cv2.medianBlur(image, ksize=7)
    kernel = np.ones((2, 2), np.uint8)  # the bigger kernel the thinner line
    image = cv2.erode(image, kernel, iterations=1)  # make thinner

    return image


def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


# %%
img = "DeepLearning/YandexGPU/OCR Keras Jetson/data/original/17.png"
image = cv2.imread(img)
img_name = '17'

img = preprocess(image)
canny = cv2.Canny(img, 0, 255, 1)

cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

min_area = 5
image_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
        ROI = img[y:y + h, x:x + w]
        cv2.imwrite(f"DeepLearning/YandexGPU/OCR Keras Jetson/data/raw/roi_{img_name}_{image_number}_v2.png", ROI)
        image_number += 1

cv2.imshow('image', img)
cv2.waitKey(0)
# %%

