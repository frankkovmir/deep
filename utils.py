import cv2
import numpy as np


def pre_processing(image, width, height):
    # Resize the image and apply thresholding
    resized_image = cv2.resize(image, (width, height))
    _, thresholded_image = cv2.threshold(resized_image, 1, 255, cv2.THRESH_BINARY)
    return thresholded_image[None, :, :].astype(np.float32)