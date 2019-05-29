import os
import sys
import numpy as np
import pandas as pd
import keras
from keras_applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.callbacks import LearningRateScheduler, EarlyStopping
from tqdm import tqdm

def main():
    data_handler = DataHandler('train.csv')
    datagen = data_handler.get_from_columns('id', 'has_cactus')
    img_shape = data_handler.get_shape()
    x_train, y_train = data_handler.get_data_label('train')
    x_test = data_handler.get_data_label('test')





class DataHandler():
    def __init__(self, csv_name):
        self.dataset_path = './dataset'
        self.train_data = pd.read_csv(os.path.join(self.dataset_path, csv_name))

    def get_from_columns(self, *args):
        for column in args:
            yield self.train_data[column]

    def get_only_data(self, folname):
        fol_path = os.path.join(self.dataset_path, folname)
        img_paths = list(map(lambda fname: os.path.join(fname), os.listdir(fol_path)))
        return img_paths

    def get_data_label(self, folname='train'):
        fol_path = os.path.join(self.dataset_path, folname)
        params = self.train_data.columns.values
        datas = []
        labels = []
        if folname == 'train':
            for idx ,row in tqdm(self.train_data.iterrows()):
                for data_or_label, param in enumerate(params):
                    if data_or_label == 0:
                        img_name = row[param]
                        img_data = img_to_array(load_img(os.path.join(fol_path, img_name)))
                        datas.append(img_data)
                    else:
                        labels.append(float(row[param]))
        else:
            img_names = os.listdir(fol_path)
            for img_name in img_names:
                img_data = img_to_array(load_img(os.path.join(fol_path, img_name)))
                datas.append(img_data)
                datas = np.asarray(datas)
                datas /= 255
                return datas

        datas = np.asarray(datas)
        datas /= 255
        labels = np.asarray(labels)
        return (datas, labels)

    def get_shape(self, folname='train'):
        from random import randint
        fol_path = os.path.join(self.dataset_path, folname)
        sample_img_path = os.path.join(fol_path,os.listdir(fol_path)[randint(0,20)])
        img_bin = img_to_array(load_img(sample_img_path)) # return ndarray
        return img_bin.shape


class Train():
    def __init__(self, use_imgnt=False):
        pass

    def build_model(self, img_shape):
        self.model = Sequential()
        self.model.add(Conv2D(filters=32), kernel_size=(5,5), padding='same', activation='relu', input_shape=img_shape)
        self.model.add(Conv2D(filters=32), kernel_size=(5,5), padding='same', activation='relu')
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.1))
        self.model.add(Conv2D(filters=64), kernel_size=(3, 3), padding='same', activation='relu')
        self.model.add(Conv2D(filters=64), kernel_size=(5, 3), padding='same', activation='relu')
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.1))
        self.model.add(Flatten())
        self.model.add(Dense(256))





if __name__ == '__main__':
    main()