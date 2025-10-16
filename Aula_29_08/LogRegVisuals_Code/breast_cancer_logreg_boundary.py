#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1/(1+np.exp(-z))

def main(lr=0.1, steps=60, gif_path="breast_cancer_logreg_boundary.gif",
         last_frame="breast_cancer_logreg_boundary_last.png"):
    data = load_breast_cancer()
    X_full = data.data
    y = data.target
    feature_names = data.feature_names

    i_radius = list(feature_names).index("mean radius")
    i_texture = list(feature_names).index("mean texture")
    X = X_full[:, [i_radius, i_texture]]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200)

    frames = []
    outdir = Path("bc_frames")
    outdir.mkdir(parents=True, exist_ok=True)

    for t in range(steps):
        z = X @ w + b
        yhat = sigmoid(z)
        grad_w = (X.T @ (yhat - y)) / n
        grad_b = np.mean(yhat - y)
        w -= lr * grad_w
        b -= lr * grad_b

        plt.figure(figsize=(7,6))
        plt.scatter(X[y==0,0], X[y==0,1], alpha=0.8, label="malignant (0)")
        plt.scatter(X[y==1,0], X[y==1,1], alpha=0.8, label="benign (1)")

        if abs(w[1]) > 1e-8:
            ys = -(w[0]/w[1]) * xs - b/w[1]
            plt.plot(xs, ys, linewidth=2, label="decision boundary (z=0)")

        plt.title(f"Breast Cancer (2 features) Logistic Regression — step {t+1}/{steps}\n"
                  f"w=[{w[0]:.2f}, {w[1]:.2f}], b={b:.2f}")
        plt.xlabel("mean radius (standardized)")
        plt.ylabel("mean texture (standardized)")
        plt.legend(loc="best")
        plt.xlim(X[:,0].min()-0.5, X[:,0].max()+0.5)
        plt.ylim(X[:,1].min()-0.5, X[:,1].max()+0.5)
        plt.tight_layout()
        fp = outdir / f"frame_{t:03d}.png"
        plt.savefig(fp, dpi=100)
        plt.close()
        frames.append(fp)

    with imageio.get_writer(gif_path, mode="I", duration=0.12) as writer:
        for fp in frames:
            writer.append_data(imageio.imread(fp))

    Path(last_frame).write_bytes(Path(frames[-1]).read_bytes())

    y_pred = (sigmoid(X @ w + b) >= 0.5).astype(int)
    acc = (y_pred == y).mean()
    print("Final training accuracy:", acc)
    print("Saved:", gif_path, last_frame)

if __name__ == "__main__":
    main()
