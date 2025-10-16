import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
def sigmoid(z): return 1/(1+np.exp(-z))
def main(steps=24, lr=0.1, out="combined_zline_plus_surface.gif", elev=30, azim=-60):
    data=load_breast_cancer(); X=data.data; y=data.target; names=data.feature_names
    i1=list(names).index("mean radius"); i2=list(names).index("mean texture")
    X=X[:,[i1,i2]]; X=StandardScaler().fit_transform(X)
    w=np.zeros(2); b=0.0
    x1_min,x1_max=X[:,0].min()-0.7,X[:,0].max()+0.7
    x2_min,x2_max=X[:,1].min()-0.7,X[:,1].max()+0.7
    xs_2d=np.linspace(x1_min,x1_max,300)
    X1g,X2g=np.meshgrid(np.linspace(x1_min,x1_max,60), np.linspace(x2_min,x2_max,60))
    outdir=Path("combo_frames"); outdir.mkdir(exist_ok=True); frames=[]
    for t in range(steps):
        z=X@w+b; yhat=sigmoid(z); w-=lr*(X.T@(yhat-y))/len(y); b-=lr*np.mean(yhat-y)
        # Left: z-line vs x1 (points at true z_i)
        figA=plt.figure(figsize=(6.0,4.5))
        plt.plot(xs_2d, w[0]*xs_2d + b, lw=2, label="z vs x1 (approx)")
        plt.scatter(X[y==0,0], X[y==0]@w + b, alpha=0.85, label="malignant (0): z_i")
        plt.scatter(X[y==1,0], X[y==1]@w + b, alpha=0.85, label="benign (1): z_i")
        plt.axhline(0, ls="--"); 
        if abs(w[0])>1e-8: plt.axvline(-b/w[0], ls="--")
        plt.title(f"Raw linear scores z (step {t+1}/{steps}) | w=[{w[0]:.2f},{w[1]:.2f}], b={b:.2f}")
        plt.xlabel(f"{names[i1]} (std)"); plt.ylabel("z"); plt.ylim(-12,12); plt.xlim(x1_min,x1_max); plt.legend(loc="upper left")
        pA=outdir/f"A_{t:03d}.png"; plt.tight_layout(); plt.savefig(pA,dpi=100); plt.close()
        # Right: 3D surface
        Z= sigmoid(w[0]*X1g + w[1]*X2g + b)
        figB=plt.figure(figsize=(6.0,4.5)); ax=figB.add_subplot(111,projection="3d")
        ax.plot_surface(X1g,X2g,Z,linewidth=0,antialiased=True,alpha=0.95)
        ax.contour(X1g,X2g,Z,levels=[0.5],zdir='z',offset=0,linestyles='--',linewidths=1.0)
        ax.set_title("Probability surface σ(w·x + b)")
        ax.set_xlabel(f"{names[i1]} (std)"); ax.set_ylabel(f"{names[i2]} (std)"); ax.set_zlabel("P(y=1|x)")
        ax.set_zlim(0,1); ax.set_xlim(x1_min,x1_max); ax.set_ylim(x2_min,x2_max); ax.view_init(elev=elev,azim=azim)
        pB=outdir/f"B_{t:03d}.png"; plt.tight_layout(); plt.savefig(pB,dpi=100,bbox_inches="tight"); plt.close()
        # Compose
        imA=Image.open(pA).convert("RGB"); imB=Image.open(pB).convert("RGB")
        h=max(imA.height,imB.height)
        La=Image.new("RGB",(imA.width,h),(255,255,255)); La.paste(imA,(0,(h-imA.height)//2))
        Rb=Image.new("RGB",(imB.width,h),(255,255,255)); Rb.paste(imB,(0,(h-imB.height)//2))
        combo=Image.new("RGB",(La.width+Rb.width,h),(255,255,255)); combo.paste(La,(0,0)); combo.paste(Rb,(La.width,0))
        frame=outdir/f"combo_{t:03d}.png"; combo.save(frame,"PNG")
        frames.append(frame)
    with imageio.get_writer(out,mode="I",duration=0.2) as wtr:
        for f in frames: wtr.append_data(imageio.imread(f))
    Path(out.replace(".gif","_last.png")).write_bytes(Path(frames[-1]).read_bytes())
if __name__=="__main__": main()
