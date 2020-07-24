import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


def get_name(file_path):
    return os.path.basename(file_path)


def import_data_info(path):
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    # print(data.head())
    # print(data['center'][0])
    # print(get_name(data['center'][0]))
    data['center'] = data['center'].apply(get_name)
    print(data.head())
    print("Total Images Imported", data.shape[0])
    return data


def balance_data(data, display=True):
    n_bins = 31
    samples_per_bin = 200
    hist, bins = np.histogram(data['steering'], n_bins)
    print(bins)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        # print(center)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samples_per_bin, samples_per_bin))
        plt.show()
    remove_index_list = []
    for j in range(n_bins):
        bin_data_list = []
        for i in range(len(data['steering'])):
            if bins[j] <= data['steering'][i] <= bins[j + 1]:
                bin_data_list.append(i)
        bin_data_list = shuffle(bin_data_list)
        bin_data_list = bin_data_list[samples_per_bin:]
        remove_index_list.extend(bin_data_list)
    print("Removed Images ", len(remove_index_list))
    data.drop(data.index[remove_index_list], inplace=True)
    print("Remaining Images", len(data))
    if display:
        hist, _ = np.histogram(data['steering'], n_bins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samples_per_bin, samples_per_bin))
        plt.show()

    return data


def load_data(path, data):
    images_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        print(indexed_data)
        images_path.append(os.path.join(path, 'IMG', indexed_data[0]))
        steering.append(float(indexed_data[3]))

    images_path = np.asarray(images_path)
    steering = np.asarray(steering)
    return images_path, steering


def augment_image(image_path, steering):
    img = mpimg.imread(image_path)

    ## PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
        img = pan.augment_image(img)

    ## ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    ## BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.8, 1.2))
        img = brightness.augment_image(img)

    ## FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering

    return img, steering


# img_re, st = augment_image('left_2020_07_23_07_30_08_963.jpg', 0)
# plt.imshow(img_re)
# plt.show()

def preprocessing(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


# img_re= preprocessing(mpimg.imread('left_2020_07_23_07_30_08_963.jpg'))
# plt.imshow(img_re)
# plt.show()

def batch_generator(images_path, steering_list, batch_size, train_flag):
    while True:
        image_batch = []
        steering_batch = []

        for i in range(batch_size):
            index = random.randint(0, len(images_path) - 1)
            if train_flag:
                img, steering = augment_image(images_path[index], steering_list[index])
            else:
                img = mpimg.imread(images_path[index])
                steering = steering_list[index]

            img = preprocessing(img)
            image_batch.append(img)
            steering_batch.append(steering)
        yield (np.asarray(image_batch), np.asarray(steering_batch))


def create_model():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), (2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(48, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.compile(Adam(lr=0.0001), loss='mse')

    return model
