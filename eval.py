import numpy as np
import cv2
import matplotlib.image as mpimg


def one_hot_encode(label):
    one_hot_encoded = []

    if label == "red":
        one_hot_encoded = [1, 0, 0]
    elif label == "yellow":
        one_hot_encoded = [0, 1, 0]
    elif label == "green":
        one_hot_encoded = [0, 0, 1]

    return one_hot_encoded


def standardize_input(image):
    standard_im = cv2.resize(image, (6, 12))
    return standard_im


def predict_label(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    height = len(hsv)
    weight = len(hsv[0])

    v = hsv[:, :, 2]

    red_sum = np.sum(v[1:int(height / 3) + 1, 0:weight - 7])
    yellow_sum = np.sum(v[int(height / 3):int(height / 1.5), 0:weight - 7])
    green_sum = np.sum(v[int(height / 1.5):int(height), 0:weight - 7])

    if green_sum > yellow_sum and green_sum > red_sum:
        predicted_label = 'green'
    elif yellow_sum > red_sum and yellow_sum > green_sum:
        predicted_label = 'yellow'
    elif red_sum > yellow_sum and red_sum > green_sum:
        predicted_label = "red"
    else:
        predicted_label = 'red'

    encoded_label = one_hot_encode(predicted_label)

    return encoded_label

