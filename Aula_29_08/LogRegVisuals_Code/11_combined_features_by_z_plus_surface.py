import numpy as np, matplotlib.pyplot as plt, imageio.v2 as imageio
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
def sigmoid(z): return 1/(1+np.exp(-z))
def main(steps=24, lr=0.1, out="combined_features_by_z_plus_surface.gif", pad=0.3, elev=30, azim=-60):
    data=load_breast_cancer(); X=data.data; y=data.target; names=data.feature_names
    i1=list(names).index("mean radius"); i2=list(names).index("mean texture")
    X=X[:,[i1,i2]]; X=StandardScaler().fit_transform(X)
    w=np.zeros(2); b=0.0
    x1_min,x1_max=X[:,0].min()-pad,X[:,0].max()+pad
    x2_min,x2_max=X[:,1].min()-pad,X[:,1].max()+pad
    X1g,X2g=np.meshgrid(np.linspace(x1_min,x1_max,120), np.linspace(x2_min,x2_max,120))
    outdir=Path("composite_zsep_frames"); outdir.mkdir(exist_ok=True); frames=[]
    for t in range(steps):
        z=X@w+b; yhat=sigmoid(z); w-=lr*(X.T@(yhat-y))/len(y); b-=lr*np.mean(yhat-y)
        # Left: 2D features colored by z + z=0 line
        figL=plt.figure(figsize=(6.2,4.8)); z_pts=X@w+b
        sc=plt.scatter(X[:,0],X[:,1],c=z_pts,alpha=0.9)
        Zgrid=w[0]*X1g+w[1]*X2g+b; cs=plt.contour(X1g,X2g,Zgrid,levels=[0],linestyles='--',linewidths=1.2)
        plt.clabel(cs, fmt={0:'z=0'}, inline=True); plt.colorbar(sc,label="linear score z")
        plt.title(f"Feature space colored by z (step {t+1}/{steps}) | w=[{w[0]:.2f},{w[1]:.2f}], b={b:.2f}")
        plt.xlabel(f"{names[i1]} (std)"); plt.ylabel(f"{names[i2]} (std)"); plt.xlim(x1_min,x1_max); plt.ylim(x2_min,x2_max)
        pL=outdir/f"L_{t:03d}.png"; plt.tight_layout(); plt.savefig(pL,dpi=100,bbox_inches="tight"); plt.close()
        # Right: 3D surface
        Psurf=sigmoid(Zgrid); figR=plt.figure(figsize=(6.2,4.8)); ax=figR.add_subplot(111,projection="3d")
        ax.plot_surface(X1g,X2g,Psurf,linewidth=0,antialiased=True,alpha=0.95)
        ax.contour(X1g,X2g,Psurf,levels=[0.5],zdir='z',offset=0,linestyles='--',linewidths=1.0)
        ax.set_title("Probability surface σ(w·x + b)"); ax.set_xlabel(f"{names[i1]} (std)"); ax.set_ylabel(f"{names[i2]} (std)"); ax.set_zlabel("P(y=1|x)")
        ax.set_zlim(0,1); ax.set_xlim(x1_min,x1_max); ax.set_ylim(x2_min,x2_max); ax.view_init(elev=elev,azim=azim)
        pR=outdir/f"R_{t:03d}.png"; plt.tight_layout(); plt.savefig(pR,dpi=100,bbox_inches="tight"); plt.close()
        # Compose
        from PIL import Image
        imL=Image.open(pL).convert("RGB"); imR=Image.open(pR).convert("RGB"); h=max(imL.height,imR.height)
        Lpad=Image.new("RGB",(imL.width,h),(255,255,255)); Lpad.paste(imL,(0,(h-imL.height)//2))
        Rpad=Image.new("RGB",(imR.width,h),(255,255,255)); Rpad.paste(imR,(0,(h-imR.height)//2))
        combo=Image.new("RGB",(Lpad.width+Rpad.width,h),(255,255,255)); combo.paste(Lpad,(0,0)); combo.paste(Rpad,(Lpad.width,0))
        frame=outdir/f"combo_{t:03d}.png"; combo.save(frame,"PNG"); frames.append(frame)
    with imageio.get_writer(out,mode="I",duration=0.2) as wtr:
        for f in frames: wtr.append_data(imageio.imread(f))
    Path(out.replace(".gif","_last.png")).write_bytes(Path(frames[-1]).read_bytes())
if __name__=="__main__": main()
