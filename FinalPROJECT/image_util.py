import numpy as np
import cv2
from pkl_file_handler import load
from pkl_file_handler import save


def load_greyscale_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)


def normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / std


def load_training_dataset(filepath):
    training_set = load(filepath)
    # training_set = IntegralImage.integrate_dataset(training_set)
    face_set = []
    non_face_set = []
    for image, label in training_set:
        if label == 1:
            face_set.append((IntegralImage.get_integral_image(image), label))
            # else:
            non_face_set.append((image, label))
    return face_set, non_face_set


class IntegralImage:

    @staticmethod
    def get_integral_image(image):
        cum_row_sum = np.cumsum(image, axis=0)
        integral_sum = np.cumsum(cum_row_sum, axis=1)

        integral_image = np.pad(integral_sum, ((1, 0), (1, 0)))
        return integral_image

    @staticmethod
    def integrate_dataset(dataset):
        integral_images = map(lambda x: (IntegralImage.get_integral_image(x[0]), x[1]), dataset)
        return list(integral_images)

    @staticmethod
    def variance_normalize_dataset(image_dataset):
        normalized_dataset = []
        for image, label in image_dataset:
            if np.std(image) >= 1:
              print("yes")
              normalized_dataset.append(((image - np.mean(image)) / np.std(image), label))
        return normalized_dataset
