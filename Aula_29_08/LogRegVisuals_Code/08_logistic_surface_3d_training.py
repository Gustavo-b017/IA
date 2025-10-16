import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
def sigmoid(z): return 1/(1+np.exp(-z))
def main(steps=24, lr=0.1, out="logistic_surface_training_3d.gif"):
    data=load_breast_cancer(); X=data.data; y=data.target; names=data.feature_names
    i1=list(names).index("mean radius"); i2=list(names).index("mean texture")
    X=X[:,[i1,i2]]; X=StandardScaler().fit_transform(X)
    w=np.zeros(2); b=0.0
    x1=np.linspace(X[:,0].min()-0.7,X[:,0].max()+0.7,60); x2=np.linspace(X[:,1].min()-0.7,X[:,1].max()+0.7,60)
    X1,X2=np.meshgrid(x1,x2)
    outdir=Path("surface_frames"); outdir.mkdir(exist_ok=True); frames=[]
    for t in range(steps):
        z=X@w+b; yhat=sigmoid(z); w-=lr*(X.T@(yhat-y))/len(y); b-=lr*np.mean(yhat-y)
        Z=sigmoid(w[0]*X1+w[1]*X2+b)
        fig=plt.figure(figsize=(7,5.8)); ax=fig.add_subplot(111,projection="3d")
        ax.plot_surface(X1,X2,Z,linewidth=0,antialiased=True,alpha=0.95)
        ax.contour(X1,X2,Z,levels=[0.5],zdir='z',offset=0,linestyles='--',linewidths=1.0)
        ax.set_title(f"Training logistic regression surface — step {t+1}/{steps}\n w=[{w[0]:.2f},{w[1]:.2f}], b={b:.2f}")
        ax.set_xlabel(f"{names[i1]} (std)"); ax.set_ylabel(f"{names[i2]} (std)"); ax.set_zlabel("P(y=1|x)")
        ax.set_zlim(0,1); ax.set_xlim(x1.min(),x1.max()); ax.set_ylim(x2.min(),x2.max()); ax.view_init(elev=25,azim=-60)
        fp=outdir/f"frame_{t:03d}.png"; plt.tight_layout(); plt.savefig(fp,dpi=90); plt.close(); frames.append(fp)
    with imageio.get_writer(out,mode="I",duration=0.2) as wtr:
        for f in frames: wtr.append_data(imageio.imread(f))
    Path(out.replace(".gif","_last.png")).write_bytes(Path(frames[-1]).read_bytes())
if __name__=="__main__": main()
