#!/usr/bin/env python3
"""
Logistic Regression Teaching Animation
--------------------------------------
Generates a single GIF with two stacked panels:
  (top) sigmoid probabilities over x
  (bottom) raw linear scores z = w*x + b with true per-point z_i
Includes "push up / push down" annotations and live (w, b) readout.

Dependencies:
  - numpy
  - matplotlib
  - imageio[v2]

Run:
  python logistic_regression_teaching_gif.py

Outputs:
  - logistic_regression_clean_combined_annotated.gif
  - logistic_regression_clean_combined_annotated_last.png
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path

def sigmoid(z):
    return 1/(1+np.exp(-z))

def make_animation(
    n0=40, n1=40, mu0=-2.0, mu1=2.0, sigma=0.8,
    lr=0.1, steps=24, seed=0,
    gif_path="logistic_regression_clean_combined_annotated.gif",
    last_frame_path="logistic_regression_clean_combined_annotated_last.png",
):
    rng = np.random.default_rng(seed)

    # Dataset: two mostly separable 1D clusters
    x0 = rng.normal(mu0, sigma, size=n0)
    x1 = rng.normal(mu1, sigma, size=n1)
    x = np.concatenate([x0, x1])
    y = np.array([0]*n0 + [1]*n1)

    # Training init
    w, b = 0.0, 0.0
    xs = np.linspace(x.min()-1.0, x.max()+1.0, 300)

    frames = []
    outdir = Path("frames_combined_annotated")
    outdir.mkdir(parents=True, exist_ok=True)

    for t in range(steps):
        # forward + gradients
        z = w * x + b
        yhat = sigmoid(z)
        grad_w = np.mean((yhat - y) * x)
        grad_b = np.mean(yhat - y)

        # gradient descent update
        w -= lr * grad_w
        b -= lr * grad_b

        # --- PLOT ---
        fig, axes = plt.subplots(2,1,figsize=(7,8), sharex=True)

        # Top: probabilities
        ax = axes[0]
        ax.scatter(x[y==0], np.zeros_like(x[y==0]), alpha=0.9, label="class 0")
        ax.scatter(x[y==1], np.ones_like(x[y==1]),  alpha=0.9, label="class 1")
        ys_curve = sigmoid(w*xs + b)
        ax.plot(xs, ys_curve, linewidth=2, label="sigmoid(w·x+b)")
        ax.axhline(0.5, linestyle="--", linewidth=1)
        xb = None
        if abs(w) > 1e-6:
            xb = -b / w
            ax.axvline(xb, linestyle="--", linewidth=1)
        ax.set_title(f"Sigmoid probabilities (step {t+1}/{steps})   |   w={w:.2f}, b={b:.2f}")
        ax.set_ylabel("predicted probability")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc="lower right")

        # Annotations in probability view
        if xb is None or not np.isfinite(xb):
            left_x = np.percentile(xs, 25)
            right_x = np.percentile(xs, 75)
        else:
            left_x = xb - 1.0
            right_x = xb + 1.0
        ax.annotate("push down (class 0)", xy=(left_x, 0.2), xytext=(left_x-0.8, 0.6),
                    arrowprops=dict(arrowstyle="->", lw=1))
        ax.annotate("push up (class 1)",   xy=(right_x, 0.8), xytext=(right_x+0.8, 0.4),
                    arrowprops=dict(arrowstyle="->", lw=1))

        # Bottom: raw linear scores
        ax2 = axes[1]
        z_line = w*xs + b
        ax2.plot(xs, z_line, linewidth=2, label="linear score z=w·x+b")
        ax2.scatter(x[y==0], (w*x[y==0] + b), alpha=0.9, label="class 0 (true z_i)")
        ax2.scatter(x[y==1], (w*x[y==1] + b), alpha=0.9, label="class 1 (true z_i)")
        ax2.axhline(0, linestyle="--")
        if xb is not None and np.isfinite(xb):
            ax2.axvline(xb, linestyle="--")
        ax2.set_title("Raw linear scores (before sigmoid)")
        ax2.set_xlabel("feature x")
        ax2.set_ylabel("score z")
        ax2.set_ylim(-10, 10)
        ax2.legend(loc="upper left")

        # Annotations in z-space
        ax2.annotate("push toward z<0", xy=(left_x, 0.5), xytext=(left_x, 3.5),
                     arrowprops=dict(arrowstyle="->", lw=1))
        ax2.annotate("push toward z>0", xy=(right_x, -0.5), xytext=(right_x, -3.5),
                     arrowprops=dict(arrowstyle="->", lw=1))

        plt.tight_layout()
        frame_path = outdir / f"frame_{t:03d}.png"
        plt.savefig(frame_path, dpi=90)
        plt.close()
        frames.append(frame_path)

    # Save combined GIF and last frame
    with imageio.get_writer(gif_path, mode="I", duration=0.2) as writer:
        for fp in frames:
            writer.append_data(imageio.imread(fp))

    Path(last_frame_path).write_bytes(Path(frames[-1]).read_bytes())

if __name__ == "__main__":
    make_animation()
