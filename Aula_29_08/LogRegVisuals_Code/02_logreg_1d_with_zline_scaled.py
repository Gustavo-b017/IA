import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from pathlib import Path
def sigmoid(z): return 1/(1+np.exp(-z))
def main(steps=70, lr=0.2, seed=7, out="logistic_regression_with_z.gif"):
    rng = np.random.default_rng(seed)
    n = 80
    x = rng.normal(0, 1.2, size=n)
    w_true, b_true = 2.0, -0.3
    p = sigmoid(w_true*x + b_true)
    y = (rng.uniform(0,1,size=n) < p).astype(int)
    w=b=0.0; xs = np.linspace(x.min()-1.0, x.max()+1.0, 400)
    outdir = Path("frames_1d_zline"); outdir.mkdir(exist_ok=True); frames=[]
    for t in range(steps):
        z = w*x + b; yhat = sigmoid(z)
        w -= lr*np.mean((yhat-y)*x); b -= lr*np.mean(yhat-y)
        plt.figure(figsize=(7,4.5))
        plt.scatter(x[y==0], np.zeros_like(x[y==0]), alpha=0.9, label="class 0")
        plt.scatter(x[y==1], np.ones_like(x[y==1]),  alpha=0.9, label="class 1")
        ys_curve = sigmoid(w*xs + b); plt.plot(xs, ys_curve, linewidth=2, label="sigmoid(w·x+b)")
        z_line = (w*xs + b); z_line_norm = (z_line - z_line.min())/(z_line.max()-z_line.min()+1e-9)
        plt.plot(xs, z_line_norm, linestyle="--", label="linear score z (scaled)")
        plt.axhline(0.5, linestyle="--", linewidth=1); 
        if abs(w)>1e-6: plt.axvline(-b/w, linestyle="--", linewidth=1)
        plt.title(f"Training (step {t+1}/{steps})"); plt.xlabel("x"); plt.ylabel("output / scaled z")
        plt.legend(loc="lower right"); plt.ylim(-0.1,1.1); plt.xlim(xs.min(), xs.max())
        fp = outdir/f"frame_{t:03d}.png"; plt.tight_layout(); plt.savefig(fp, dpi=120); plt.close(); frames.append(fp)
    with imageio.get_writer(out, mode="I", duration=0.08) as wtr:
        for f in frames: wtr.append_data(imageio.imread(f))
    Path(out.replace(".gif","_last.png")).write_bytes(Path(frames[-1]).read_bytes())
if __name__=="__main__": main()
