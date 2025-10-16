import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from pathlib import Path
def sigmoid(z): return 1/(1+np.exp(-z))
def main(n0=40,n1=40,mu0=-2.0,mu1=2.0,sigma=0.8,lr=0.1,steps=24,out="logistic_regression_clean_combined_annotated.gif"):
    rng=np.random.default_rng(0)
    x0=rng.normal(mu0,sigma,size=n0); x1=rng.normal(mu1,sigma,size=n1)
    x=np.concatenate([x0,x1]); y=np.array([0]*n0+[1]*n1)
    w=b=0.0; xs=np.linspace(x.min()-1.0,x.max()+1.0,300)
    outdir=Path("frames_clean_ann"); outdir.mkdir(exist_ok=True); frames=[]
    for t in range(steps):
        z=w*x+b; yhat=sigmoid(z); w-=lr*np.mean((yhat-y)*x); b-=lr*np.mean(yhat-y)
        fig,axes=plt.subplots(2,1,figsize=(7,8),sharex=True)
        ax=axes[0]; ax.scatter(x[y==0],np.zeros_like(x[y==0]),alpha=0.9,label="class 0")
        ax.scatter(x[y==1],np.ones_like(x[y==1]),alpha=0.9,label="class 1")
        ax.plot(xs,sigmoid(w*xs+b),lw=2,label="sigmoid(w·x+b)"); ax.axhline(0.5,ls="--",lw=1)
        xb=None
        if abs(w)>1e-6: xb=-b/w; ax.axvline(xb,ls="--",lw=1)
        ax.set_title(f"Sigmoid probabilities (step {t+1}/{steps})   |   w={w:.2f}, b={b:.2f}")
        ax.set_ylim(-0.1,1.1); ax.legend(loc="lower right")
        if xb is None or not np.isfinite(xb): left_x=np.percentile(xs,25); right_x=np.percentile(xs,75)
        else: left_x=xb-1.0; right_x=xb+1.0
        ax.annotate("push down (class 0)", xy=(left_x,0.2), xytext=(left_x-0.8,0.6), arrowprops=dict(arrowstyle="->", lw=1))
        ax.annotate("push up (class 1)",   xy=(right_x,0.8),xytext=(right_x+0.8,0.4), arrowprops=dict(arrowstyle="->", lw=1))
        ax2=axes[1]; ax2.plot(xs,w*xs+b,lw=2,label="linear score z=w·x+b")
        ax2.scatter(x[y==0],w*x[y==0]+b,alpha=0.9,label="class 0 (z_i)")
        ax2.scatter(x[y==1],w*x[y==1]+b,alpha=0.9,label="class 1 (z_i)")
        ax2.axhline(0,ls="--"); 
        if abs(w)>1e-6: ax2.axvline(xb,ls="--")
        ax2.set_title("Raw linear scores (before sigmoid)"); ax2.set_xlabel("feature x"); ax2.set_ylabel("score z"); ax2.set_ylim(-10,10); ax2.legend(loc="upper left")
        fp=outdir/f"frame_{t:03d}.png"; plt.tight_layout(); plt.savefig(fp,dpi=90); plt.close(); frames.append(fp)
    with imageio.get_writer(out,mode="I",duration=0.2) as wtr:
        for f in frames: wtr.append_data(imageio.imread(f))
    Path(out.replace(".gif","_last.png")).write_bytes(Path(frames[-1]).read_bytes())
if __name__=="__main__": main()
