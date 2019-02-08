import torch
import numpy as np

device = torch.device('cpu')
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

start_w1 = np.random.randn(D_in, H)
start_w2 = np.random.randn(H, D_out)

def test_lr_range(st, ed):
    best = []
    for i in range(30):
        pow = np.random.uniform(st, ed)
        learning_rate = 10**pow
        w1 = torch.tensor(start_w1, dtype=torch.float32, device=device, requires_grad=True)
        w2 = torch.tensor(start_w2, dtype=torch.float32, device=device, requires_grad=True)
        for t in range(200):
            y_pred = x.mm(w1).clamp(min=0).mm(w2)
            loss = (y_pred - y).pow(2).sum()
            loss.backward()

            with torch.no_grad():
                w1 -= learning_rate * w1.grad
                w2 -= learning_rate * w2.grad

                w1.grad.zero_()
                w2.grad.zero_()
        
        loss = (y_pred - y).pow(2).sum()
        if not np.isnan(loss.item()): 
            best.append((loss.item(), pow))
    best.sort()
    pows = []
    losses = []
    for i, j in best:
        losses.append(i)
        pows.append(j)
    return pows, losses

pows, losses = test_lr_range(-10, 1)
best_pow, best_loss = pows[0], losses[0]
for i in range(10):
    if losses[0] < best_loss:
        best_pow, best_loss = pows[0], losses[0]
    print("iteration: %i, alpha=1e%.6f, loss=%.6f" % (i+1, pows[0], losses[0]))
    st = pows[0] - 3 * np.std(pows)
    ed = pows[0] + 3 * np.std(pows)
    pows, losses = test_lr_range(st, ed)

print("Best alpha=1e%.6f, Best loss=%.6f" % (best_pow, best_loss))