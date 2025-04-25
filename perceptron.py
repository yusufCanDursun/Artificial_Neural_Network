import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivate(x):
    return sigmoid(x) * (1 - sigmoid(x))

X = np.array([[0,0],[0,1],[1,0],[1,1]]) 

Y = np.array([[0],[0],[0],[1]])

lr = 0.02

w = np.random.randn(2,1)
b = np.random.randn(1)

for epoch  in range(50) :
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






