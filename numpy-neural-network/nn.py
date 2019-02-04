import numpy as np

x = np.ones((5, 1))
w = np.random.uniform(0,1,(10, 5))
alpha = 0.01

Ytrue = np.ones((10, 5)).dot(x) / 10 # Function to be learned
print("Actual function")
print(Ytrue)

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(10000):
    Ypred = sigmoid(w.dot(x))
    grad = ((Ypred - Ytrue) * Ytrue * (1 - Ytrue)).dot(x.T)
    w -= alpha * grad

err = 0.5 * np.sum(np.square(Ypred - Ytrue))
print("Error: ", err)
print("Learned function")
print(sigmoid(w.dot(x)))