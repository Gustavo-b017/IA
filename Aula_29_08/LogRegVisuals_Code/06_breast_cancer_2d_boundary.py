import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
def sigmoid(z): return 1/(1+np.exp(-z))
def main(steps=60, lr=0.1, out="breast_cancer_logreg_boundary.gif"):
    data=load_breast_cancer(); X=data.data; y=data.target; names=data.feature_names
    i1=list(names).index("mean radius"); i2=list(names).index("mean texture")
    X=X[:,[i1,i2]]; X=StandardScaler().fit_transform(X)
    n,d=X.shape; w=np.zeros(d); b=0.0; xs=np.linspace(X[:,0].min()-0.5,X[:,0].max()+0.5,200)
    outdir=Path("bc_frames"); outdir.mkdir(exist_ok=True); frames=[]
    for t in range(steps):
        z=X@w+b; yhat=sigmoid(z); w-=lr*(X.T@(yhat-y))/n; b-=lr*np.mean(yhat-y)
        plt.figure(figsize=(7,6)); plt.scatter(X[y==0,0],X[y==0,1],alpha=0.8,label="malignant (0)")
        plt.scatter(X[y==1,0],X[y==1,1],alpha=0.8,label="benign (1)")
        if abs(w[1])>1e-8: ys=-(w[0]/w[1])*xs - b/w[1]; plt.plot(xs,ys,lw=2,label="decision boundary (z=0)")
        plt.title(f"Breast Cancer (2 features) Logistic Regression — step {t+1}/{steps}\n w=[{w[0]:.2f},{w[1]:.2f}], b={b:.2f}")
        plt.xlabel(f"{names[i1]} (std)"); plt.ylabel(f"{names[i2]} (std)")
        plt.legend(); plt.xlim(xs.min(),xs.max()); plt.ylim(X[:,1].min()-0.5,X[:,1].max()+0.5)
        fp=outdir/f"frame_{t:03d}.png"; plt.tight_layout(); plt.savefig(fp,dpi=100); plt.close(); frames.append(fp)
    with imageio.get_writer(out,mode="I",duration=0.12) as wtr:
        for f in frames: wtr.append_data(imageio.imread(f))
    Path(out.replace(".gif","_last.png")).write_bytes(Path(frames[-1]).read_bytes())
    acc=((sigmoid(X@w+b)>=0.5).astype(int)==y).mean(); print("Final training accuracy:",acc)
if __name__=="__main__": main()
