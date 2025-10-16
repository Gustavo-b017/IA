#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio
from PIL import Image
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

def sigmoid(z): return 1/(1+np.exp(-z))

def main(steps=24, lr=0.1, out_gif="combined_features_by_z_plus_surface.gif"):
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
    X1g, X2g = np.meshgrid(np.linspace(x1_min, x1_max, 120),
                           np.linspace(x2_min, x2_max, 120))

    outdir = Path("composite_zsep_frames"); outdir.mkdir(parents=True, exist_ok=True)
    frames = []
    elev, azim = 30, -60

    for t in range(steps):
        z = X @ w + b
        yhat = sigmoid(z)
        grad_w = (X.T @ (yhat - y)) / len(y)
        grad_b = np.mean(yhat - y)
        w -= lr * grad_w
        b -= lr * grad_b

        # Left: 2D colored by z + z=0
        figL = plt.figure(figsize=(6.2,4.8))
        z_pts = X @ w + b
        sc = plt.scatter(X[:,0], X[:,1], c=z_pts, alpha=0.9)
        Zgrid = w[0]*X1g + w[1]*X2g + b
        cs = plt.contour(X1g, X2g, Zgrid, levels=[0], linestyles='--', linewidths=1.2)
        plt.clabel(cs, fmt={{0: 'z=0'}}, inline=True)
        plt.colorbar(sc, label="linear score z")
        plt.title(f"Feature space colored by z (step {{t+1}}/{{steps}}) | w=[{{w[0]:.2f}},{{w[1]:.2f}}], b={{b:.2f}}")
        plt.xlabel("{feature_names[i_radius]} (std)")
        plt.ylabel("{feature_names[i_texture]} (std)")
        plt.xlim(x1_min, x1_max); plt.ylim(x2_min, x2_max)
        plt.tight_layout()
        pathL = outdir / f"L_{{t:03d}}.png"
        plt.savefig(pathL, dpi=100, bbox_inches="tight")
        plt.close(figL)

        # Right: 3D surface
        Psurf = sigmoid(Zgrid)
        figR = plt.figure(figsize=(6.2,4.8))
        ax3d = figR.add_subplot(111, projection="3d")
        ax3d.plot_surface(X1g, X2g, Psurf, linewidth=0, antialiased=True, alpha=0.95)
        ax3d.contour(X1g, X2g, Psurf, levels=[0.5], zdir='z', offset=0, linestyles='--', linewidths=1.0)
        ax3d.set_title("Probability surface σ(w·x + b)")
        ax3d.set_xlabel("{feature_names[i_radius]} (std)")
        ax3d.set_ylabel("{feature_names[i_texture]} (std)")
        ax3d.set_zlabel("P(y=1|x)")
        ax3d.set_zlim(0,1)
        ax3d.set_xlim(x1_min, x1_max); ax3d.set_ylim(x2_min, x2_max)
        ax3d.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        pathR = outdir / f"R_{{t:03d}}.png"
        plt.savefig(pathR, dpi=100, bbox_inches="tight")
        plt.close(figR)

        # Compose
        from PIL import Image
        imL = Image.open(pathL).convert("RGB")
        imR = Image.open(pathR).convert("RGB")
        h = max(imL.height, imR.height)
        Lpad = Image.new("RGB", (imL.width, h), (255,255,255)); Lpad.paste(imL, (0,(h-imL.height)//2))
        Rpad = Image.new("RGB", (imR.width, h), (255,255,255)); Rpad.paste(imR, (0,(h-imR.height)//2))
        combo = Image.new("RGB", (Lpad.width + Rpad.width, h), (255,255,255))
        combo.paste(Lpad, (0,0)); combo.paste(Rpad, (Lpad.width,0))
        frame_path = outdir / f"combo_{{t:03d}}.png"; combo.save(frame_path, "PNG")
        frames.append(frame_path)

    with imageio.get_writer(out_gif, mode="I", duration=0.2) as writer:
        for fp in frames:
            writer.append_data(imageio.imread(fp))
    last = out_gif.replace(".gif","_last.png")
    Path(last).write_bytes(Path(frames[-1]).read_bytes())
    print("Saved:", out_gif, last)

if __name__ == "__main__":
    main()
