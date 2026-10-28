import torch, torch.nn as nn, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print('-------------------------------ШАГ 1---------------------------------------')
print()

torch.set_num_threads(4)  # число ядер процессора
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train = datasets.MNIST('data', train=True, download=True, transform=tf)
test = datasets.MNIST('data', train=False, download=True, transform=tf)
img, label = train[0]; print(img.shape, label)
fig, ax = plt.subplots(5, 5, figsize=(6, 6))
for i, a in enumerate(ax.flat): a.imshow(train[i][0][0], cmap='gray'); a.set_title(train[i][1]); a.axis('off')
plt.show()

print('-------------------------------ШАГ 2---------------------------------------')
print()

train_dl = DataLoader(train, batch_size=128, shuffle=True)
test_dl = DataLoader(test, batch_size=512)
x, y = next(iter(train_dl)); print(x.shape, y.shape, len(train_dl))

print('-------------------------------ШАГ 3 ГРАФИК---------------------------------------')
print()

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)

print('-------------------------------ШАГ 4---------------------------------------')
print()

import time
for ep in range(3):
    t0 = time.time(); model.train()
    for x, y in train_dl:
        opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_dl: correct += (model(x).argmax(1) == y).sum().item()
    sched.step()
    print(f'epoch {ep} loss {loss.item():.3f} acc {correct / len(test):.4f} lr {sched.get_last_lr()[0]} time {time.time()-t0:.0f}s')

print('-------------------------------ШАГ 5---------------------------------------')
print()

model.eval()

images, labels = next(iter(test_dl))

with torch.no_grad():
    preds = model(images).argmax(1)

wrong_indices = (preds != labels).nonzero(as_tuple=True)[0]

fig, ax = plt.subplots(4, 4, figsize=(8, 8))
for i, idx in enumerate(wrong_indices[:16]):
    img = images[idx][0]
    true_label = labels[idx].item()
    pred_label = preds[idx].item()
    
    a = ax.flat[i]
    a.imshow(img, cmap='gray')
    a.set_title(f'Предсказано: {pred_label}\nПравда: {true_label}')
    a.axis('off')
plt.tight_layout()
plt.show()

conf_matrix = torch.zeros(10, 10, dtype=torch.int)
for x, y in test_dl:
    with torch.no_grad():
        p = model(x).argmax(1)
    for t, pr in zip(y, p):
        conf_matrix[t, pr] += 1

print("Матрица ошибок (строки - истина, столбцы - предсказание):")
print(conf_matrix)

print(f"Ошибок 4->9: {conf_matrix[4][9].item()}")
print(f"Ошибок 3->5: {conf_matrix[3][5].item()}")


print('-------------------------------ШАГ 6---------------------------------------')
print()

#torch.save(state_dict, 'mnist.pt') #Новая модель, load_state_dict, та же accuracy. Файл около 400 КБ.
torch.save(model.state_dict(), 'mnist.pt')
m2 = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
m2.load_state_dict(torch.load('mnist.pt')); m2.eval()

print('-------------------------------ШАГ 7---------------------------------------')
print()

tf_no_norm = transforms.Compose([transforms.ToTensor()])
train_no_norm = datasets.MNIST('data', train=True, download=True, transform=tf_no_norm)
test_no_norm = datasets.MNIST('data', train=False, download=True, transform=tf_no_norm)

train_dl_nn = DataLoader(train_no_norm, batch_size=128, shuffle=True)
test_dl_nn = DataLoader(test_no_norm, batch_size=512)

model_exp = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
opt_exp = torch.optim.Adam(model_exp.parameters(), lr=1e-3)

model_exp.train()
for x, y in train_dl_nn:
    opt_exp.zero_grad()
    loss = loss_fn(model_exp(x), y)
    loss.backward()
    opt_exp.step()

model_exp.eval()
correct_nn = 0
with torch.no_grad():
    for x, y in test_dl_nn:
        correct_nn += (model_exp(x).argmax(1) == y).sum().item()

print(f'Accuracy без Normalize (1 эпоха): {correct_nn / len(test_no_norm):.4f}')

model_lr = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
opt_lr = torch.optim.Adam(model_lr.parameters(), lr=0.1) # Измененный lr

model_lr.train()
for x, y in train_dl: 
    opt_lr.zero_grad()
    loss = loss_fn(model_lr(x), y)
    loss.backward()
    opt_lr.step()

model_lr.eval()
correct_lr = 0
with torch.no_grad():
    for x, y in test_dl:
        correct_lr += (model_lr(x).argmax(1) == y).sum().item()

print(f'Accuracy с lr=0.1 (1 эпоха): {correct_lr / len(test):.4f}')