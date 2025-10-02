import numpy as np
import matplotlib.pyplot as plt

base_lr = 0.01
t = np.linspace(0, 1, 1000)  # normalized round
lr = base_lr * (1 - np.cos(np.pi * t)) / 2  # Cosine warmup + decay

plt.plot(t, lr, color="orange", label="Cosine Warmup+Decay (0 → base_lr → 0)")
plt.axhline(base_lr, linestyle="--", color="gray", label="Base LR")
plt.xlabel("Normalized Round (t)")
plt.ylabel("Learning Rate")
plt.title("Cosine Warmup + Decay LR Schedule")
plt.legend()
plt.show()