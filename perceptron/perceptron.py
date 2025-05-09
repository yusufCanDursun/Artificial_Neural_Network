import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([[0], [0], [0], [1]])

np.random.seed(42)
w = np.random.randn(2,1)
b = np.random.randn(1)

lr = 10
epochs = 250
plot_update_frequency = 10

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

for epoch in range(epochs):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)

    error = y_pred - y
    dz = error * sigmoid_derivative(z)
    dw = np.dot(X.T, dz)
    db = np.sum(dz)

    w -= lr * dw
    b -= lr * db

    if epoch % plot_update_frequency == 0:
        ax.cla()

        for i in range(len(X)):
            if y[i] == 0:
                ax.scatter(X[i, 0], X[i, 1], color='red', marker='o', s=100, label='0' if i == 0 else "")
            else:
                ax.scatter(X[i, 0], X[i, 1], color='blue', marker='x', s=100, label='1' if i == 3 else "")

        x1_values = np.linspace(-0.5, 1.5, 100)
        if abs(w[1, 0]) > 1e-6:
            x2_values = (-w[0, 0] * x1_values - b[0]) / w[1, 0]
            ax.plot(x1_values, x2_values, label=f'Karar Sınırı (Epoch {epoch})', color='green')
        else:
            x1_boundary = -b[0] / w[0, 0]
            ax.axvline(x=x1_boundary, label=f'Karar Sınırı (Epoch {epoch})', color='green')

        ax.set_xlabel('Girdi 1 (x1)')
        ax.set_ylabel('Girdi 2 (x2)')
        ax.set_title(f'Eğitim Süreci - Hata: {sum(abs(error))}')
        ax.legend(loc='upper left')
        ax.grid(True)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)

        plt.draw()
        plt.pause(0.2)

print("Eğitim Sonrası Ağırlıklar:")
print(w)
print("Eğitim Sonrası Bias:")
print(b)

print("Eğitim Sonrası Tahminler:")
print(sigmoid(np.dot(X, w) + b).round())

plt.ioff()
plt.show()
