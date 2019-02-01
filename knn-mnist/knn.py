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
    for k in range(1, 6):
        knn = KNN(k)
        knn.train(mnist.train_images, mnist.train_labels)
        test_sz = 50
        cnt_correct = 0
        # TODO: Parallelize testing
        for idx in range(test_sz):
            image = mnist.test_images[idx]
            label = mnist.test_labels[idx]
            if knn.predict(image) == label:
                cnt_correct += 1

        print("For k={}, acc={}".format(k, cnt_correct / test_sz))