import numpy as np
from scipy.stats import mode
import load_mnist

class KNN:
    def __init__(self, k):
        self.k = k

    def train(self, X, y):
        # Just memorize the data
        self.X = X
        self.y = y

    def predict(self, X):
        distances = np.sum(np.abs(self.X - X), axis = 1)
        labels = self.y[distances.argsort()[:self.k]]
        return mode(labels)[0][0]

if __name__ == "__main__":
    mnist = load_mnist.Mnist("./mnist")
    train_sz = 5000
    val_sz = 100
    # Convert to black and white binary value
    train_images = (mnist.train_images[:train_sz] > 0) + 0
    val_images = (mnist.test_images[:val_sz] > 0) + 0
    val_labels = mnist.test_labels[:val_sz]
    for k in range(1, 11):
        knn = KNN(k)
        knn.train(train_images, mnist.train_labels)
        cnt_correct = 0
        # TODO: Parallelize testing
        for (image, label) in zip(val_images, val_labels):
            prediction = knn.predict(image)
            if prediction == label:
                cnt_correct += 1

        print("For k={}, acc={}".format(k, cnt_correct / val_sz))