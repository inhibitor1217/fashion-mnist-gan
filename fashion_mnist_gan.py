import numpy as np
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

def normalize_image(img):
    return img / 127.5 - 1.

def load_dataset():
    (train_x, _), _ = fashion_mnist.load_data()
    train_x = np.expand_dims(train_x, axis=-1)
    train_x = normalize_image(train_x)
    return train_x

if __name__ == '__main__':
    train_x = load_dataset()