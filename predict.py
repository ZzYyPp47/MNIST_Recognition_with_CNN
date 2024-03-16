# -*- coding:utf-8 -*-
'''
@Author: ZYP
@contact: 3137168510@qq.com
@Time: 2023/7/9 11:30
@version: 1.0
@File: predict.py
'''


import tensorflow as tf
from PIL import Image
import numpy as np
from train import CNN
import os


class Predict(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('./ckpt')
        self.cnn = CNN()
        # 恢复网络权重
        self.cnn.model.load_weights(latest)
    def predict(self, image_path):
        # 以黑白方式读取图片
        img = Image.open(image_path).convert('L')
        img = np.reshape(img, (28, 28, 1)) / 255.
        x = np.array([1 - img])
        y = self.cnn.model.predict(x,verbose=0)
        print(image_path,'-> Predict number is', np.argmax(y[0]))


def get_file_paths(folder_path):
    file_paths = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            file_paths.append(os.path.join(dirpath, filename))
    return file_paths


if __name__ == "__main__":
    app = Predict()
    folder_path = './test_images/'  # 图片路径
    file_paths=get_file_paths(folder_path)
    for file_path in file_paths:
        app.predict(file_path)