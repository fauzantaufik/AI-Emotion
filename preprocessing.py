import os
import copy
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    base_data_dir = '.\P96-Section-2-Emotion-AI\Emotion+AI+Dataset\Emotion AI Dataset'
    df = pd.read_csv(
        os.path.join(base_data_dir, 'data.csv')
    )
    return df

def image_str_to_np(image):
    return np.fromstring(image, dtype=int, sep=' ').reshape(96, 96)



def generate_horizontal_filp(df):
    # facial key point columns
    columns = df.columns[:-1]

    df_copy = copy.copy(df)

    # horizontal Flip - flip the images along y axis
    df_copy['Image'] = df_copy['Image'].apply(lambda x : np.flip(x, axis=1))

    # since we are flipping horizontally, y coordinate values would be the same
    # Only x coordiante values would change, all we have to do is to subtract our initial x-coordinate values from width of the image(96)
    for i in range(len(columns)):
        if i%2 == 0:
            df_copy[columns[i]] = df_copy[columns[i]].apply(lambda x: 96. - float(x) )
    return df_copy


def generate_inc_brightness(df):
    # Increase the brightness of the image
    df_copy = copy.copy(df)
    df_copy['Image'] = df_copy['Image'].apply(lambda x : np.clip(random.uniform(1.5, 2)*x, 0, 255.0))
    return df_copy


def normalize_image(df):
    images = df[:, -1]
    images = images/225.
    return images


def x_into_format_batch(df):
    images = df[:, -1]

    # create an empty array of shape (x, 96, 96, 1) to feed the model
    X = np.empty((len(images), 96, 96, 1))

    # iterate ghrough the img list an add image values to the empty array after expanding it's dimension
    for i in range(len(images)):
        X[i,] = np.expand_dims(images[i], axis=2)
        
    X = np.asarray(X).astype(np.float32)
    return X


def y_as_float(df):
    y = df[:, :-1]
    y = np.asarray(y).astype(np.float32)
    return y


def preprocess():
    df = load_data()
    df['Image'] = df['Image'].apply(image_str_to_np)

    # generate data augmentation
    df_horizontal_flip = generate_horizontal_filp(df)
    df_inc_brightness = generate_inc_brightness(df)
    # concatenate the original image with augmented dataframe
    augmented_df = np.concatenate([df, df_horizontal_flip, df_inc_brightness])
    X = x_into_format_batch(augmented_df)
    y = y_as_float(augmented_df)

    # split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess()

