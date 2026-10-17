print('---------------------ЧИСТО ГРАФИК--------------------------------')

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(200, centers=2, cluster_std=1.2, random_state=1)
plt.scatter(X[:, 0], X[:, 1], c=y); plt.show()

print('-----------------------------------------------------')

sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, 2); b = 0.0
def forward(X): return sig(X @ w + b)
print(((forward(X) > .5) == y).mean())

print('-----------------------------------------------------')

def bce(p, y): return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
print(bce(forward(X), y))

print('---------------------ГРАФИК--------------------------------')

lr, hist = 0.1, []
for i in range(500):
    p = forward(X); dz = p - y
    w -= lr * X.T @ dz / len(X); b -= lr * dz.mean()
    hist.append(bce(p, y))
plt.plot(hist); plt.show()
print('accuracy:', ((forward(X) > .5) == y).mean())

print('-----------------------ЧИСТО ГРАФИК------------------------------')

xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200), np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
zz = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, zz, levels=[0, .5, 1], alpha=.3); plt.scatter(X[:,0], X[:,1], c=y); plt.show()

print('----------------ДОМАШНЕЕ ПОВТОРЕНИЕ-------------------------------------')

print('---------------------ЧИСТО ГРАФИК--------------------------------')

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(200, centers=2, cluster_std=1.2, random_state=1)
plt.scatter(X[:, 0], X[:, 1], c=y); plt.show()

print('-----------------------------------------------------')

tanh = lambda z: np.tanh(z)
rng = np.random.default_rng(0)
w = rng.normal(0, 0.1, 2); b = 0.0
def forward(X): return tanh(X @ w + b)
print(((forward(X) > 0) == y).mean())

print('-----------------------------------------------------')

y_tanh = y * 2 - 1 
def loss_func(p, y_t): 
    return np.mean((p - y_t)**2)
print(loss_func(forward(X), y_tanh))



print('---------------------ГРАФИК--------------------------------')

lr, hist = 0.1, []
for i in range(500):
    z = X @ w + b 
    p = tanh(z)
    dz = (p - y_tanh) * (1 - p**2) 
    w -= lr * X.T @ dz / len(X)
    b -= lr * dz.mean()
    hist.append(loss_func(p, y_tanh)) 
plt.plot(hist); plt.show()
print('accuracy:', ((forward(X) > 0) == y).mean())

print('-----------------------ЧИСТО ГРАФИК------------------------------')

xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200), np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
zz = forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, zz, levels=[-1, 0, 1], alpha=.3); plt.scatter(X[:,0], X[:,1], c=y); plt.show()