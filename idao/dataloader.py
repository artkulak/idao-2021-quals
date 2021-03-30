import numpy as np
import os
import pathlib as path
from PIL import Image
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, batch_size=32, shuffle=True):
        self.images = images
        self.indices = np.arange(len(images))
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.on_epoch_end()

    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.indices[k] for k in index]
    
        X = self.__get_data(batch)
        return X

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        X = np.zeros((self.batch_size, 150, 150, 3))
        for i, idx in enumerate(batch):
            with Image.open(self.images[idx]) as img:
                image = np.asarray(img)
            image = image.reshape(image.shape[0], image.shape[1], 1)
            image = np.concatenate([image, image, image], axis = 2)
            
            center = image.shape[0] // 2
            image = image[center-128//2:center+128//2,center-128//2:center+128//2, :]
            X[i,] = image

        return X / 255.0

