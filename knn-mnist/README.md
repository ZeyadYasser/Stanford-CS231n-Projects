# KNN on Mnist

- download mnist dataset files from http://yann.lecun.com/exdb/mnist/ and place them at /knn-mnist/mnist
    - train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
    - train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
    - t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
    - t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes) 

### Run
```bash
    cd knn-mnist
    python knn.py
```

Change train_sz in knn.py to tune the model (**speed vs accuracy tradeoff**).
For K=8 & train_sz=5000, the model provides 97% accuracy based on 100 samples. 