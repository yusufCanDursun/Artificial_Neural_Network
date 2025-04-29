import numpy as np
import matplotlib.pyplot as plot

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivate(x):
    return sigmoid(x) * (1 - sigmoid(x))

X = np.array([[0,0],[0,1],[1,0],[1,1]]) 

Y = np.array([[0],[0],[0],[1]])

np.random.seed(42)
w = np.random.randn(2,1)
b = np.random.randn(1)

plot_update_frequency = 10

lr = 10
epochs = 1000

for epoch  in range(epochs) :
    z = np.dot(X, w) + b
    y = sigmoid(z)
    error = y - Y 

    dz = error * derivate(z)
    dw = np.dot(X.T , dz)
    db = np.sum(dz)
    
    w -= lr * dw
    b -= lr * db
    

    
print(w)
print(b)






