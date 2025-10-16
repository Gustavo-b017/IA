import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from pathlib import Path
def sigmoid(z): return 1/(1+np.exp(-z))
def main(steps=30, lr=0.2, seed=7, out="logistic_regression_sigmoid_plus_z.gif"):
    rng = np.random.default_rng(seed); n=80
    x = rng.normal(0, 1.2, size=n); w_true, b_true=2.0,-0.3
    p = sigmoid(w_true*x+b_true); y=(rng.uniform(0,1,size=n)<p).astype(int)
    w=b=0.0; xs=np.linspace(x.min()-1.0,x.max()+1.0,400)
    outdir=Path("frames_1d_sigplusz"); outdir.mkdir(exist_ok=True); frames=[]
    for t in range(steps):
        z=w*x+b; yhat=sigmoid(z); w-=lr*np.mean((yhat-y)*x); b-=lr*np.mean(yhat-y)
        fig,axes=plt.subplots(2,1,figsize=(7,8),sharex=True)
        ax=axes[0]; ax.scatter(x[y==0],np.zeros_like(x[y==0]),alpha=0.9,label="class 0")
        ax.scatter(x[y==1],np.ones_like(x[y==1]),alpha=0.9,label="class 1")
        ax.plot(xs,sigmoid(w*xs+b),linewidth=2,label="sigmoid(w·x+b)")
        ax.axhline(0.5,ls="--",lw=1); 
        if abs(w)>1e-6: ax.axvline(-b/w,ls="--",lw=1)
        ax.set_title(f"Sigmoid probabilities (step {t+1}/{steps})"); ax.set_ylim(-0.1,1.1); ax.legend(loc="lower right")
        ax2=axes[1]; z_line=w*xs+b; ax2.plot(xs,z_line,linewidth=2,label="z=w·x+b")
        ax2.scatter(x[y==0],w*x[y==0]+b,alpha=0.9,label="class 0 (z_i)")
        ax2.scatter(x[y==1],w*x[y==1]+b,alpha=0.9,label="class 1 (z_i)")
        ax2.axhline(0,ls="--"); 
        if abs(w)>1e-6: ax2.axvline(-b/w,ls="--")
        ax2.set_xlabel("x"); ax2.set_ylabel("z"); ax2.set_ylim(-6,6); ax2.legend(loc="upper left")
        fp=outdir/f"frame_{t:03d}.png"; plt.tight_layout(); plt.savefig(fp,dpi=110); plt.close(); frames.append(fp)
    with imageio.get_writer(out,mode="I",duration=0.15) as wtr:
        for f in frames: wtr.append_data(imageio.imread(f))
    Path(out.replace(".gif","_last.png")).write_bytes(Path(frames[-1]).read_bytes())
if __name__=="__main__": main()
