print('f(x) = (x-3)**2')

import numpy as np, matplotlib.pyplot as plt

f = lambda x:(x-3) **2
df = lambda x: 2 * (x-3)
x, lr, path = 10.0, 0.1, []
for i in range(30):
    path.append(x); x = x - lr * df(x)
print(x)
xs = np.linspace(-2, 12, 100); plt.plot(xs, f(xs)); plt.scatter(path, [f(p) for p in path], c = 'r'); plt.show()

print('-----------------------------------------------------')

rng = np.random.default_rng(0)
X = rng.uniform(0, 5, 50)
y = 2 * X + 1 + rng.normal(0, 0.5, 50)
plt.scatter(X, y); plt.show()

print('-----------------------------------------------------')

def loss(w, b): return np.mean((w * X + b - y)**2)
def grad(w, b):
    p = w * X + b
    return np.mean(2 * (p-y) * X), np.mean(2*(p-y))

print('-----------------------------------------------------')

w, b, lr, hist = 0.0, 0.0, 0.05, []
for i in range(200):
    dw, db = grad(w, b)
    w -= lr*dw
    b -= lr*db 
    hist.append(loss(w, b))
print(w, b)
plt.plot(hist); plt.show()
plt.scatter(X, y); plt.plot(X, w*X+b, 'r'); plt.show()

print('-----------------------------------------------------')
print('ДОМАШНЕЕ ПОВТОРЕНИЕ')

print('f(x1) = a1*x1**2 + b1*x1')

f1 = lambda x1: a*x1**2 + b*x1
df1 = lambda x1: 2*a*x1+b
x1, lr, path = 10.0, 0.1, []
for i in range(30):
    path.append(x1); x1 = x1 - lr * df(x1)
print(x1)
xs1 = np.linspace(-2, 12, 100); plt.plot(xs1, f(xs1)); plt.scatter(path, [f(p) for p in path], c = 'r'); plt.show()
