#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio
from PIL import Image
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def sigmoid(z): 
    return 1/(1+np.exp(-z))

def main(steps=24, lr=0.1, out_gif="combined_zline_plus_surface.gif", seed=0):
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
    xs_2d = np.linspace(x1_min, x1_max, 300)
    X1g, X2g = np.meshgrid(np.linspace(x1_min, x1_max, 60),
                           np.linspace(x2_min, x2_max, 60))

    outdir = Path("combo_frames"); outdir.mkdir(parents=True, exist_ok=True)
    frames = []
    elev, azim = 30, -60

    for t in range(steps):
        z = X @ w + b
        yhat = sigmoid(z)
        grad_w = (X.T @ (yhat - y)) / len(y)
        grad_b = np.mean(yhat - y)
        w -= lr * grad_w
        b -= lr * grad_b

        # 2D z-line view
        figA = plt.figure(figsize=(6.0,4.5))
        z_line = w[0]*xs_2d + b
        plt.plot(xs_2d, z_line, linewidth=2, label="z = w·x + b (vs x1)")
        plt.scatter(X[y==0,0], (X[y==0]@w + b), alpha=0.85, label="malignant (0): true z_i")
        plt.scatter(X[y==1,0], (X[y==1]@w + b), alpha=0.85, label="benign (1): true z_i")
        plt.axhline(0, linestyle="--")
        if abs(w[0]) > 1e-8:
            xb = -b / w[0]
            plt.axvline(xb, linestyle="--")
        plt.title(f"Raw linear scores z (step {t+1}/{steps}) | w=[{w[0]:.2f},{w[1]:.2f}], b={b:.2f}")
        plt.xlabel("{feature_names[i_radius]} (std)")
        plt.ylabel("score z")
        plt.ylim(-12, 12); plt.xlim(x1_min, x1_max)
        plt.legend(loc="upper left")
        plt.tight_layout()
        pathA = outdir / f"A_{t:03d}.png"
        plt.savefig(pathA, dpi=100)
        plt.close(figA)

        # 3D surface view
        Zsurf = sigmoid(w[0]*X1g + w[1]*X2g + b)
        figB = plt.figure(figsize=(6.0,4.5))
        ax3d = figB.add_subplot(111, projection="3d")
        ax3d.plot_surface(X1g, X2g, Zsurf, linewidth=0, antialiased=True, alpha=0.95)
        ax3d.contour(X1g, X2g, Zsurf, levels=[0.5], zdir='z', offset=0, linestyles='--', linewidths=1.0)
        ax3d.set_title("Probability surface σ(w·x + b)")
        ax3d.set_xlabel("{feature_names[i_radius]} (std)")
        ax3d.set_ylabel("{feature_names[i_texture]} (std)")
        ax3d.set_zlabel("P(y=1|x)")
        ax3d.set_zlim(0,1)
        ax3d.set_xlim(x1_min, x1_max)
        ax3d.set_ylim(x2_min, x2_max)
        ax3d.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        pathB = outdir / f"B_{t:03d}.png"
        plt.savefig(pathB, dpi=100, bbox_inches="tight")
        plt.close(figB)

        # Compose horizontally
        from PIL import Image
        imA = Image.open(pathA).convert("RGB")
        imB = Image.open(pathB).convert("RGB")
        h = max(imA.height, imB.height)
        newA = Image.new("RGB", (imA.width, h), (255,255,255)); newA.paste(imA, (0,(h-imA.height)//2))
        newB = Image.new("RGB", (imB.width, h), (255,255,255)); newB.paste(imB, (0,(h-imB.height)//2))
        combo = Image.new("RGB", (newA.width + newB.width, h), (255,255,255))
        combo.paste(newA, (0,0)); combo.paste(newB, (newA.width,0))
        frame_path = outdir / f"combo_{t:03d}.png"; combo.save(frame_path, "PNG")
        frames.append(frame_path)

    with imageio.get_writer(out_gif, mode="I", duration=0.2) as writer:
        for fp in frames:
            writer.append_data(imageio.imread(fp))
    last = out_gif.replace(".gif","_last.png")
    Path(last).write_bytes(Path(frames[-1]).read_bytes())
    print("Saved:", out_gif, last)

if __name__ == "__main__":
    main()
