import numpy as np

def image_str_to_np(image):
    return np.fromstring(image, dtype=int, sep=' ').reshape(96, 96)