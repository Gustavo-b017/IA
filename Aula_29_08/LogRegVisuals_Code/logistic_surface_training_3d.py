#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def sigmoid(z): return 1/(1+np.exp(-z))

def main(steps=24, lr=0.1,
         gif_path="logistic_surface_training_3d.gif",
         last_path="logistic_surface_training_3d_last.png"):
    data = load_breast_cancer()
    X_full = data.data
    y = data.target
    feature_names = data.feature_names
    i_radius = list(feature_names).index("mean radius")
    i_texture = list(feature_names).index("mean texture")
    X2 = X_full[:, [i_radius, i_texture]]

    scaler = StandardScaler()
    X = scaler.fit_transform(X2)

    w = np.zeros(2); b = 0.0
    x1_min, x1_max = X[:,0].min()-0.7, X[:,0].max()+0.7
    x2_min, x2_max = X[:,1].min()-0.7, X[:,1].max()+0.7
    x1 = np.linspace(x1_min, x1_max, 60)
    x2 = np.linspace(x2_min, x2_max, 60)
    X1, X2g = np.meshgrid(x1, x2)

    frames = []
    outdir = Path("surface_frames"); outdir.mkdir(parents=True, exist_ok=True)

    for t in range(steps):
        z = X @ w + b
        yhat = sigmoid(z)
        grad_w = (X.T @ (yhat - y)) / len(y)
        grad_b = np.mean(yhat - y)
        w -= lr * grad_w
        b -= lr * grad_b

        Z = sigmoid(w[0]*X1 + w[1]*X2g + b)
        fig = plt.figure(figsize=(7,5.8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X1, X2g, Z, linewidth=0, antialiased=True, alpha=0.95)
        ax.contour(X1, X2g, Z, levels=[0.5], zdir='z', offset=0, linestyles='--', linewidths=1.0)
        ax.set_title(f"Training logistic regression surface — step {t+1}/{steps}\n"
                     f"w=[{w[0]:.2f}, {w[1]:.2f}], b={b:.2f}")
        ax.set_xlabel("mean radius (std)")
        ax.set_ylabel("mean texture (std)")
        ax.set_zlabel("P(y=1|x)")
        ax.set_zlim(0, 1)
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.view_init(elev=25, azim=-60)
        plt.tight_layout()
        fp = outdir / f"frame_{t:03d}.png"
        plt.savefig(fp, dpi=90)
        plt.close()
        frames.append(fp)

    with imageio.get_writer(gif_path, mode="I", duration=0.2) as writer:
        for fp in frames:
            writer.append_data(imageio.imread(fp))
    Path(last_path).write_bytes(Path(frames[-1]).read_bytes())
    print("Saved:", gif_path, last_path)

if __name__ == "__main__":
    main()
