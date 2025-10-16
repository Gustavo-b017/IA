import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from pathlib import Path
def sigmoid(z): return 1/(1+np.exp(-z))
def main(steps=40, lr=0.2, seed=7, out_probs="sigmoid_probabilities.gif", out_scores="linear_scores.gif"):
    rng=np.random.default_rng(seed); n=80
    x=rng.normal(0,1.2,size=n); w_true,b_true=2.0,-0.3
    p=sigmoid(w_true*x+b_true); y=(rng.uniform(0,1,size=n)<p).astype(int)
    w=b=0.0; xs=np.linspace(x.min()-1.0,x.max()+1.0,400)
    d1=Path("frames_probs"); d1.mkdir(exist_ok=True); d2=Path("frames_scores"); d2.mkdir(exist_ok=True); F1=[];F2=[]
    for t in range(steps):
        z=w*x+b; yhat=sigmoid(z); w-=lr*np.mean((yhat-y)*x); b-=lr*np.mean(yhat-y)
        # probs
        plt.figure(figsize=(7,4.5)); 
        plt.scatter(x[y==0],np.zeros_like(x[y==0]),alpha=0.9,label="class 0")
        plt.scatter(x[y==1],np.ones_like(x[y==1]),alpha=0.9,label="class 1")
        plt.plot(xs,sigmoid(w*xs+b),lw=2,label="sigmoid(w·x+b)")
        plt.axhline(0.5,ls="--",lw=1); 
        if abs(w)>1e-6: plt.axvline(-b/w,ls="--",lw=1)
        plt.title(f"Sigmoid probabilities (step {t+1}/{steps})"); plt.ylim(-0.1,1.1); plt.xlim(xs.min(),xs.max())
        plt.legend(loc="lower right"); fp=d1/f"frame_{t:03d}.png"; plt.tight_layout(); plt.savefig(fp,dpi=110); plt.close(); F1.append(fp)
        # scores
        plt.figure(figsize=(7,4.5)); 
        plt.plot(xs,w*xs+b,lw=2,label="z=w·x+b"); 
        plt.scatter(x[y==0],w*x[y==0]+b,alpha=0.9,label="class 0")
        plt.scatter(x[y==1],w*x[y==1]+b,alpha=0.9,label="class 1")
        plt.axhline(0,ls="--"); 
        if abs(w)>1e-6: plt.axvline(-b/w,ls="--")
        plt.title(f"Raw linear scores (step {t+1}/{steps})"); plt.xlabel("x"); plt.ylabel("z"); plt.ylim(-6,6); plt.xlim(xs.min(),xs.max()); plt.legend(loc="upper left")
        fp2=d2/f"frame_{t:03d}.png"; plt.tight_layout(); plt.savefig(fp2,dpi=110); plt.close(); F2.append(fp2)
    with imageio.get_writer(out_probs,mode="I",duration=0.15) as wtr:
        for f in F1: wtr.append_data(imageio.imread(f))
    with imageio.get_writer(out_scores,mode="I",duration=0.15) as wtr:
        for f in F2: wtr.append_data(imageio.imread(f))
if __name__=="__main__": main()
