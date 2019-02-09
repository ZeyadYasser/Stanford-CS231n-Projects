import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)
start_w1 = np.random.randn(D_in, H)
start_w2 = np.random.randn(H, D_out)

start_w1 = np.random.randn(D_in, H)
start_w2 = np.random.randn(H, D_out)
start_v1 = np.zeros_like(start_w1)
start_v2 = np.zeros_like(start_w2)

def SGD_Momentum(lr, moment):
    losses = []
    w1 = torch.tensor(start_w1, dtype=torch.float32, device=device, requires_grad=True)
    v1 = torch.tensor(start_v1, dtype=torch.float32, device=device)
    w2 = torch.tensor(start_w2, dtype=torch.float32, device=device, requires_grad=True)
    v2 = torch.tensor(start_v2, dtype=torch.float32, device=device)
    for t in range(100):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred - y).pow(2).sum()
        loss.backward()

        with torch.no_grad():
            v1 = moment * v1 + lr * w1.grad
            v2 = moment * v2 + lr * w2.grad
            w1 -= v1
            w2 -= v2

            w1.grad.zero_()
            w2.grad.zero_()
        
        losses.append(loss.item())
    return losses

def SGD(lr):
    losses = []
    w1 = torch.tensor(start_w1, dtype=torch.float32, device=device, requires_grad=True)
    w2 = torch.tensor(start_w2, dtype=torch.float32, device=device, requires_grad=True)
    for t in range(100):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred - y).pow(2).sum()
        loss.backward()

        with torch.no_grad():
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad

            w1.grad.zero_()
            w2.grad.zero_()

        losses.append(loss.item())
    return losses

# After many trials I found those Hyperparameters to be better for SGD+Momentum
momentum_losses = SGD_Momentum(1e-6, 0.8)
# From hyperparameter_optim.py I found lr of 1e-6 to perform well
sgd_losses = SGD(1e-6)

size = len(momentum_losses)
print(size)
for i in range(size):
    print("Iteration: %i, SGD+Momentum Loss=%.6f, SGD Loss=%.6f" \
                % (i+1, momentum_losses[i], sgd_losses[i]))

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(14,5))
# Plot log(loss) instead due to large loss at the beginning
ax1.plot(np.log(momentum_losses), 'b')
ax1.set_title('SGD+Momentum')
ax2.plot(np.log(sgd_losses), 'r')
ax2.set_title('SGD')
plt.ylabel('log(loss)')
plt.show()
