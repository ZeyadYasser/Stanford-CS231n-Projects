import struct
import numpy as np

class Mnist:
    def __init__(self, path, one_hot=False):
        print("Loading data .....")
        self.one_hot = one_hot
        self.path = path
        self.train_images = self.train_images()
        self.train_labels = self.train_labels()
        self.test_images = self.test_images()
        self.test_labels = self.test_labels()
        print("Data loaded succesfully.")
    #load train images
    def train_images(self):
        file = open(self.path + '/train-images.idx3-ubyte', 'rb')
        magic_nm, size, rows, cols = struct.unpack('>IIII', file.read(16))
        train_images = np.fromfile(file, dtype=np.dtype('B'))  # read the data into numpy
        train_images = np.reshape(train_images, (size, rows * cols))
        return train_images
    #load train labels
    def train_labels(self):
        with open(self.path + '/train-labels.idx1-ubyte', 'rb') as file:
            data = np.frombuffer(file.read(), np.uint8, offset=8)
        if self.one_hot:
            num_samples = len(data)
            one_hot_lables = np.zeros((num_samples, 10))
            for i in range(num_samples):
                one_hot_lables[i][data[i]]=1
            return one_hot_lables
        return data
    #load test images
    def test_images(self):
        file = open(self.path + '/t10k-images.idx3-ubyte', 'rb')
        magic_nm, size, rows, cols = struct.unpack('>IIII', file.read(16))
        test_images = np.fromfile(file, dtype=np.dtype('B'))  # read the data into numpy
        test_images = np.reshape(test_images, (size, rows * cols))
        return test_images
    #load test labels
    def test_labels(self):
        file = open(self.path + '/t10k-labels.idx1-ubyte', 'rb')
        file.read(8)
        test_labels = np.fromfile(file, dtype=np.dtype('B'))  # read the data into numpy
        if self.one_hot:
            num_samples = len(test_labels)
            one_hot_lables = np.zeros((num_samples, 10))
            for i in range(num_samples):
                one_hot_lables[i][test_labels[i]]=1
            return one_hot_lables
        return test_labels
    

