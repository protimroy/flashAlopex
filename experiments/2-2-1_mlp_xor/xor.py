import os, sys, torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os


# Make local optimizer importable
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python"))
from optimizer import FlashAlopexCUDA

# Where to save plots
out_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(out_dir, exist_ok=True)

class Config:
    device = torch.device("cuda")
    max_epochs = 1500
    lambda_val = 0.5

device = Config.device

# Full-batch XOR (4 samples)
X = torch.tensor([[0.,0],[0,1],[1,0],[1,1]], device=device)
Y = torch.tensor([0.,1.,1.,0.], device=device)  # float for BCEWithLogits

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 1)
    def forward(self, x):
        h = torch.sigmoid(self.l1(x))   # hidden sigmoid
        return self.l2(h)               # linear output (logits)

model = MLP().to(device)
with torch.no_grad():
    # symmetry breaking helps 2-2-1 XOR
    nn.init.orthogonal_(model.l1.weight, gain=1.0)
    model.l1.bias.copy_(torch.tensor([-3.0, 3.0], device=device))
    nn.init.xavier_uniform_(model.l2.weight, gain=1.0)
    nn.init.zeros_(model.l2.bias)

opt = FlashAlopexCUDA(model.parameters(), delta=0.01, seed=42)

# Denominator like the notebook, with a strong floor
denominator = torch.tensor(1e-3, device=device)
lambda_val = 0.1  # match notebook

loss_hist, acc_hist, pflip_hist, denom_hist = [], [], [], []

for epoch in range(Config.max_epochs):
    # update denom (notebook form) + floor
    denominator = (lambda_val - 1.0) * denominator + lambda_val * opt.delta_E.abs()
    denominator = denominator.clamp_min(1e-3)

    # 1) loss_t with MSE (linear outputs)
    with torch.no_grad():
        out_t = model(X).squeeze(-1)
        loss_t = F.mse_loss(out_t, Y)

    # 2) ALOPEX update
    p_flip = opt.step(loss_t, denominator=denominator)

    # 3) loss_{t+1}
    with torch.no_grad():
        out_tp1 = model(X).squeeze(-1)
        loss_tp1 = F.mse_loss(out_tp1, Y)

    # 4) update ΔE
    opt.update_delta_E(loss_tp1)

    # metrics (threshold 0.5 on linear output, sigmoid only for plotting)
    with torch.no_grad():
        preds = (out_tp1 > 0.5).float()
        acc = (preds == Y).float().mean().item()

    loss_hist.append(loss_tp1.item())
    acc_hist.append(acc)
    pflip_hist.append(float(p_flip))
    denom_hist.append(float(denominator.item()))
    print(f"loss {loss_tp1.item():.4f} acc {acc:.3f}")

# Final predictions
with torch.no_grad():
    out = model(X).squeeze(-1)
    preds = (out > 0.5).to(torch.int32)
    for x, o, y, pr in zip(X.tolist(), out.tolist(), Y.tolist(), preds.tolist()):
        print(f"input {x} -> out={o:.3f} pred={pr} target={int(y)}")


# Plots: loss, accuracy, p_flip/denominator, decision boundary
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Loss
axes[0, 0].plot(loss_hist, label="loss")
axes[0, 0].set_title("Loss")
axes[0, 0].set_xlabel("epoch")
axes[0, 0].set_ylabel("BCE")
axes[0, 0].grid(True)

# Accuracy
axes[0, 1].plot(acc_hist, label="acc", color="green")
axes[0, 1].set_title("Accuracy")
axes[0, 1].set_xlabel("epoch")
axes[0, 1].set_ylabel("acc")
axes[0, 1].set_ylim(0, 1.05)
axes[0, 1].grid(True)

# p_flip and denominator
ax3 = axes[1, 0]
ax3.plot(pflip_hist, label="p_flip", color="tab:blue")
ax3.set_title("p_flip and denominator")
ax3.set_xlabel("epoch")
ax3.set_ylabel("p_flip", color="tab:blue")
ax3.tick_params(axis="y", labelcolor="tab:blue")
ax3.grid(True)
ax3b = ax3.twinx()
ax3b.plot(denom_hist, label="denominator", color="tab:red")
ax3b.set_ylabel("denominator", color="tab:red")
ax3b.tick_params(axis="y", labelcolor="tab:red")

# Decision boundary
with torch.no_grad():
    xs = torch.linspace(-0.2, 1.2, 200, device=device)
    ys = torch.linspace(-0.2, 1.2, 200, device=device)
    gx, gy = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)
    pr = torch.sigmoid(model(grid).squeeze(-1)).reshape(gx.shape).cpu().numpy()

im = axes[1, 1].contourf(xs.cpu().numpy(), ys.cpu().numpy(), pr.T, levels=np.linspace(0, 1, 21), cmap="viridis")
axes[1, 1].scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=Y.cpu(), cmap="coolwarm", edgecolors="k", s=80)
axes[1, 1].set_title("Decision boundary (P(class=1))")
axes[1, 1].set_xlim(-0.2, 1.2)
axes[1, 1].set_ylim(-0.2, 1.2)
fig.colorbar(im, ax=axes[1, 1], shrink=0.8)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "xor_metrics.png"), dpi=200, bbox_inches="tight")
plt.show()
